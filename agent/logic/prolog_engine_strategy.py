from io import StringIO
from json import loads
from logging import Logger
from typing import Any, Callable, Optional, Tuple, List
import ast
from agent.logic.engine_strategy import EngineStrategy, SolverOutcome
from agent.logic.logic_py_c_harness_generator import LogicPyCHarnessGenerator
from libcst import MetadataWrapper, Module


_CODE_PREFIX: str =""":- use_module(library(clpfd))."""
# Instructs the model to generate the solution constraints.
_CONSTRAINTS_MESSAGE: str = r"""
You have already defined the data structure of a logic puzzle (variable declarations, grouping, and `all_different` constraints). Now, proceed to the **second step**: write the **logical constraints** and the **output display**.

---

Given the following puzzle description (including all clues), generate the following:

### 1. Constraints Section
- Translate **each clue** into a Prolog constraint syntax.
- For each constraint, **add a comment above it** with the format:
  `% Clue X: <original clue in plain English>`
- Do not skip or merge clues — each clue must have exactly one matching constraint and one comment.
- Use the same variable names defined in the `Vars` and groups (e.g., `Eric`, `Arnold`, `Dog`, etc.).
** Your solver tool supports only these 4 operators `#=`, `#<`, `#>`, `#\=` and only the function `abs/1`.
Here is an exmple to follow:
    % Clue 1: The person whose child is Fred is somewhere to the left of Eric.
    Fred #< Eric,
    % Clue 2: There are two houses between The person whose mother's name is Penny and the person who is short.
    abs(Penny - Short) #= 3,
    % Clue 3: The person whose favorite color is red is in the second house.
    Red #= 2,
    % Clue 4: The rabbit owner is directly left of The person whose mother's name is Aniya.
    Rabbit #= Aniya - 1,

- and finally you shoud label and print the results
labeling([], Vars),
write(Vars), nl. ** this line needs to end with a point '.'

"""

_DATA_STRUCTURE_MESSAGE: str = """Here is the puzzle:
```
{}
```
"""

# Instructs the model to format the solver solution according to the requested format
_FORMAT_MESSAGE: str = """Your logic solver tool produced the following solution:
```
{solution}
```

Now convert it to the expected output format:
```
{output_format}
```

⚠️ Very important instructions:
- If there are words related with an underscore '_' remove it (for exemple "very_short" turn it into "very short")
Except that :
- Do **not** modify or auto-correct any names, values, or strings.
- Keep all names **exactly** as they appear in the solution (no rewriting like changing "mar" into "March", "f150" into "f-150", etc).
- Your task is only to **reformat the output**, not to reinterpret or rename anything.
- Preserve original casing, spelling, and word order.
"""

# Instructs the model on how to generate a solution data structure.
_SYSTEM_PROMPT: str = """
You are an expert puzzle solving agent with access to a propositional logic solver tool a prolog alike that has a convenient interface optimised for you. Your boss has given you a logic puzzle that he thinks can be mapped to your logic solver tool

I will walk you through these steps one by one. Do **not** attempt to solve the puzzle or anticipate future steps unless I explicitly ask you to. We will now begin with the **first step**: define the **data structure** for a valid puzzle solution.

---

Given the following puzzle description (including both entity definitions and logical constraints), extract only the structure of the solution space as follows:

### Your task:
1. create prdicate solve/0 where we are gonna work in
2. **Extract all entities** from the puzzle (e.g., names, pets, colors, objects...).
3. Create a **single flat list** named `Vars` that contains all the variables (e.g., `[Eric, Arnold, Dog, Cat]`).**All variable names must begin with an uppercase letter.**
4. Add the domain constraint:
   `Vars ins 1..N`,
   where `N` is the number of houses or positions.
5. **Group them by logical category** (e.g., Names, Pets, Colors...).
6. For each group, create a variable group declaration like:
   `Names = [Eric, Arnold],
7. For each group, also add a constraint:
   `all_different(Names),`
8. End the last line with a coma ',', do not use a point '.'
9. Do not translate the clues or constraints yet — we will handle that in later steps.
10. **put your prolog code in between ``` ```
---
here is an exemple :
```
solve :-
    % Extract all entities from the puzzle
    Vars = [Alice, Eric, Arnold, Peter,
        Google_Pixel_6, Iphone_13, Oneplus_9, Samsung_Galaxy_S21],

    Vars ins 1..4,
    Names = [Alice, Eric, Arnold, Peter],
    PhoneModels = [Google_Pixel_6, Iphone_13, Oneplus_9, Samsung_Galaxy_S21],

    % Add constraints for each group
    all_different(Names),
    all_different(PhoneModels),
```

Here is the puzzle description:

```txt
{}
"""

_REVISE_PROMPT = r"""
You have previously been asked to convert a logical puzzle into a formal specification using a domain-specific language (DSL). The generated DSL code was incorrect or incomplete, leading to an error or a wrong solution.

Your task is to revise the original natural language prompt so that, if given again to a large language model, it would generate a correct and more robust DSL specification that avoids the observed issues.

You will be given:
- The original prompt that described the puzzle.
- The DSL code that was generated.
- The output from the solver (either an error message or an incorrect result).

**Your goal is to:**
- Identify what aspects of the original prompt may have led to the incorrect or ambiguous interpretation.
- Modify only those parts of the original prompt that contributed to the failure.

This is not about fixing the DSL code directly, but about **improving the natural language prompt** so that it can lead to better DSL code generation on the next attempt — even for complex puzzles.

Think of this as training a language model to generalize better across similar logical tasks, not just to fix one-off syntax errors.

⚠️ **Do not add or imply DSL features that are not supported.** The DSL only allows:
- Exactly four operators: `#=`, `#\=`, `#<`, and `#>`.
- Only one function: `abs`.

Here is the original prompt that was used:
---
{original_prompt}
---

Here is the DSL code that was generated:
---
{generated_dsl_code}
---

Here is the result returned by the solver:
---
{solver_output}
---

Now revise the original prompt **only where needed** to resolve the misunderstanding or error, and return a clean, improved version of the prompt that is ready to be used in a new LLM call. Be minimal, precise, and instructive.

Return only the revised prompt, without commentary or explanation.

"""



class PrologEngineStrategy(EngineStrategy):
    """
    Solves search-based problems using a Prolog.
    """

    def __init__(
        self, logger_factory: Callable[[str], Logger], task: str, output_format: str
    ) -> None:
        """
        Args:
            logger_factory (Callable[[str], Logger]): Log configuration.
            task (str): Search-based problem to solve.
            output_format (str): Output format expected by the user.
        """
        self.__logger: Logger = logger_factory(__name__)
        self.__task: str = task
        self.__output_format: str = output_format
        self.curr_constraints_prompt = _CONSTRAINTS_MESSAGE

    @property
    def constraints_prompt(self) -> list[str]:
        return [self.curr_constraints_prompt]

    def set_initial_constraints_prompt(self, prompt) -> None:
        global _CONSTRAINTS_MESSAGE
        _CONSTRAINTS_MESSAGE = prompt
        self.curr_constraints_prompt = _CONSTRAINTS_MESSAGE

    def set_constraints_prompt(self, prompt) -> None:
        self.curr_constraints_prompt = prompt

    def reset_constraints_prompt(self) -> None :
        self.curr_constraints_prompt = _CONSTRAINTS_MESSAGE

    @property
    def data_structure_prompt(self) -> str:
        return _SYSTEM_PROMPT.format(self.__output_format)

    def data_structure_included(self, constraints: str) -> bool :
        return re.match(r"\s*solve",constraints)

    async def generate_solver_constraints(self, code: str ) -> str:
        variables = PrologEngineStrategy.__extract_variables(code)
        nb_categories = PrologEngineStrategy.__extract_num(code)
        return (code,variables,nb_categories)

    def generate_solver_invocation_command(self, solver_input_file: str) -> list[str]:
        return ["swipl",
                "-s",
                solver_input_file,
                "-g",
                "solve",
                "-t", "halt"]

    def preprocess_code(self, code: str) -> tuple:
        variables = LogicAgent._extract_variables(code)
        nb_categories = LogicAgent._extract_num(code)
        return (code, variables, nb_categories)

    def get_format_prompt(self, solution: str) -> Optional[str]:
        return _FORMAT_MESSAGE.format(
            solution=solution, output_format=self.__output_format
        )
    def get_revise_prompt(slef, prompt: str, generated_dsl: str, output: str) -> str:
        return _REVISE_PROMPT.format(
        original_prompt=prompt, generated_dsl_code = generated_dsl,solver_output=output
        )
    def parse_solver_output(
        self, exit_code: int, stdout: Tuple[str,str,int], stderr: str
    ) -> Tuple[SolverOutcome, Optional[str]]:
        if exit_code == 0:
            ints = ast.literal_eval(stdout[0])
            vars = stdout[1]
            n = stdout[2]
            res = PrologEngineStrategy.__parse_prolog_solution(ints, vars, n)
            return SolverOutcome.SUCCESS, res
        else:
            self.__logger.error(f"Prolog solver failed: {stderr}")
            return SolverOutcome.FATAL, None
    @property
    def python_code_prefix(self) -> str:
        return _CODE_PREFIX

    @property
    def retry_prompt(self) -> str:
        return _UNSAT_MESSAGE

    @property
    def solver_input_file_suffix(self) -> str:
        return ".pl"

    @property
    def system_prompt(self) -> str:
        return _DATA_STRUCTURE_MESSAGE.format(self.__task)

    @staticmethod
    def __parse_prolog_solution(ints: List[int], vars: List[str], n: int) -> str:
        m = len(vars) // n
        res = [["" for _ in range(m)] for _ in range(n)]
        for i in range(len(vars)):
            idx = i // n
            res[ints[i]-1][idx] = vars[i]

        return "\n".join(f"{i+1} - {', '.join(ligne)}" for i, ligne in enumerate(res))

    @staticmethod
    def __extract_variables(text: str) -> Optional[List[str]]:
        # On capture ce qu’il y a entre "Vars = [" et "]"
        match = re.search(r'Vars\s*=\s*\[(.*?)\]', text, re.DOTALL)
        if match:
            content = match.group(1)
            variables = [v.strip() for v in content.split(',') if v.strip()]
            return variables
        else:
            return None

    @staticmethod
    def __extract_num(text: str) -> int:
        match = re.search(r"Vars\s+ins\s+1\.\.(\d)",text, re.DOTALL)
        if match:
            return int(match.group(1))
        return None

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import Logger
from typing import Any, Callable, Optional, Tuple, List
import ast
from agent.logic.engine_strategy import EngineStrategy, SolverOutcome, SolverConstraints
import re


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

In order to automatically validate the puzzle solution, your data structure will eventually need to be converted to JSON in the following format, so keep that in mind when deciding on your data structure:
```
{}
```
"""


# Sent if the solver was unable to find a solution.
_UNSAT_MESSAGE: str = (
    "Your constraints are contradictory and thus the solver could not find a solution. Please review them and try to spot the error, we will go through the step of generating the `def validate(solution: Solution) -> None` function again."
)



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

    @property
    def constraints_prompt(self) -> list[str]:
        return [_CONSTRAINTS_MESSAGE]

    @property
    def data_structure_prompt(self) -> str:
        return _SYSTEM_PROMPT.format(self.__output_format)

    async def generate_solver_constraints(self, code: str ) -> Optional[SolverConstraints]:
        variables = PrologEngineStrategy._extract_variables(code)
        nb_categories = PrologEngineStrategy._extract_num(code)
        if variables and nb_categories :
            return SolverConstraints(code,variables, nb_categories)
        return None

    def generate_solver_invocation_command(self, solver_input_file: str) -> list[str]:
        return ["swipl",
                "-s",
                solver_input_file,
                "-g",
                "solve",
                "-t", "halt"]

    def get_format_prompt(self, solution: str) -> Optional[str]:
        return _FORMAT_MESSAGE.format(
            solution=solution, output_format=self.__output_format
        )

    def parse_solver_output(
            self, exit_code: int, solverSpec: SolverConstraints, stdout: str, stderr: str
    ) -> Tuple[SolverOutcome, Optional[str]]:
        
        if exit_code == 0:
            if solverSpec.variables is None or solverSpec.nb_categories is None:
                self.__logger.error(
                    "SolverConstraints is missing required fields: 'variables' and 'nb_categories' must be set."
                )   
                return SolverOutcome.FATAL, None
            ints = ast.literal_eval(stdout)
            vars : list[str] = solverSpec.variables
            n : int = solverSpec.nb_categories
            res = PrologEngineStrategy._parse_prolog_solution(ints, vars, n)
            return SolverOutcome.SUCCESS, res

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
    def _parse_prolog_solution(ints: List[int], vars: List[str], n: int) -> str:
        """
        Converts the Prolog solver output into a human-readable format by house.

        Args:
            ints (List[int]): List of integers representing the position of each variable within its category.
            vars (List[str]): List of variable names in the order defined in the puzzle.
            n (int): Number of houses (or positions).

        Returns:
            str: A multi-line string where each line corresponds to a house,
                 listing the variables associated with that house separated by commas.

        """
        m = len(vars) // n
        res = [["" for _ in range(m)] for _ in range(n)]
        for i in range(len(vars)):
            idx = i // n
            res[ints[i]-1][idx] = vars[i]

        return "\n".join(f"{i+1} - {', '.join(ligne)}" for i, ligne in enumerate(res))


    @staticmethod
    def _extract_variables(text: str) -> List[str]:
        """
        Extracts the list of variable names from a Prolog puzzle code snippet.

        Args:
            text (str): Prolog code containing a `Vars = [...]` declaration.

        Returns:
            Optional[List[str]]: A list of variable names if found; otherwise None.
        """
        match = re.search(r'Vars\s*=\s*\[(.*?)\]', text, re.DOTALL)
        if not match:
            return []
        return [v.strip() for v in match.group(1).split(',') if v.strip()]

    @staticmethod
    def _extract_num(text: str) -> int:
        """
        Extracts the number of houses (or the upper bound of the domain) from a Prolog puzzle code snippet.

        Args:
            text (str): Prolog code containing a `Vars ins 1..N` declaration.

        Returns:
            int: The upper bound N if found; otherwise None.
        """
        match = re.search(r"Vars\s+ins\s+1\.\.(\d+)", text, re.DOTALL)
        if not match:
            return 0
        return int(match.group(1))

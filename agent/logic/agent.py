# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from logging import Logger
from re import compile, DOTALL, Match, Pattern
from tempfile import gettempdir
from typing import Callable, Optional, Tuple
import time

from agent.logic.engine_strategy import EngineStrategy, SolverOutcome
from agent.logic.solver import Solver
from agent.symex.module_with_type_info_factory import ModuleWithTypeInfoFactory
from aiofiles.tempfile import NamedTemporaryFile
from concurrency.subprocess import Subprocess
from inference.chat_completion import ChatCompletion, Role
from inference.client import InferenceClient

from judge.result_trace import ResultTrace

from libcst import MetadataWrapper, Module, parse_module, ParserSyntaxError


# Sent if an expected code snippet was not found.
NO_CODE_FOUND_MESSAGE: str = """I could not find the requested output code snippet in your last message. Please make sure you mark it as follows:
```
your output code snippet
```

Please send the entire {} again."""

# Pattern extracting code snippets from a model response.
CODE_EXTRACTION_PATTERN: Pattern = compile(r".*```[^\n]*\n+(.*)```.*", DOTALL)

# Code marker pattern. We remove redundant markers when extracting code from concatenated messages due to token limits.
CODE_MARKER_PATTERN: Pattern = compile(r"```[^\n]*")

# Number of times we retry an operation, e.g. extracting code from a response.
RETRY_COUNT: int = 5

# Constraints taking longer to solve than this are likely incorrect
_SOLVER_TIMEOUT: float = 15

ERROR_MESSAGE = r"""
You are an expert in formal verification and static analysis of Python programs extended with domain-specific languages.

I will provide you with two things:
1. A Python code snippet that includes domain-specific annotations.
2. An error output from a formal verification tool.

Your task is to:
- Carefully analyze the code.
- Understand the error messages.
- Provide a clear and concise explanation of each error, including:
  - What the error means.
  - What part of the code is responsible.
  - How to fix the code to resolve the error.

** do not try to solve the problem your self.

Be precise and structured in your explanation.

### Code:
```python
{}
```
### Error:
```
{}
```

Please explain clearly:

Why the verifier reports these errors and What changes are needed in the code to fix them.

i am gonna walk you through an exemple:
Code :

class house:
    name: Unique[Domain[str, "peter", "eric", "arnold"]]
    hobby: Unique[Domain[str, "photography", "cooking", "gardening"]]

class solution:
    houses: list[house, 3]

def validate(solution: solution) -> None:
    # 1. Eric is somewhere to the left of the person who enjoys gardening.
    eric = nondet(solution.houses)
    assume(eric.name == "eric")
    gardener = nondet(solution.houses)
    assume(gardener.hobby == "gardening")
    assert solution.houses.index(eric) < solution.houses.index(gardener)

    # 2. Arnold is not in the third house.
    arnold = nondet(solution.houses)
    assume(arnold.name == "arnold")
    assert solution.houses.index(arnold)!= 2

    # 3. Arnold is not in the second house.
    assert arnold not in solution.houses[1:2]


Erreur:
```
"syntax error before 'not'",
```

Explanation:
- The parser encountered an unexpected keyword `not` at a position where it expected a different syntax structure.
- The error occurs in the line `assert arnold not in solution.houses[1:2]`.
- This might be due to parser limitations or the static analyzer not supporting `not in` directly on slices in assertions involving nondeterministic objects.
-  Use an explicit inequality or alternative expression:
  assert solution.houses.index(arnold) != 1


"""

class LogicAgent(Solver):
    """
    Basic agent prompting LLM to:
    1) Define a data structure to hold the answer of a request/puzzle.
    2) Define constraints for a valid solution.
    3) Generate a valid solution using a solver back-end.
    """

    def __init__(
        self,
        logger_factory: Callable[[str], Logger],
        chat_completion: ChatCompletion,
        engine_strategy: EngineStrategy,
        result_trace: ResultTrace,
        collect_pyre_type_information: bool = False,
        expected_solution: str = None
        ) -> None:
        """
        Initialises inference client with default settings and the provided
        model name.

        Args:
            logger_factory (Callable[[str], Logger]): Logging configuration to
            use.
            model_name (str): Name of model to use in inference client.
            engine_strategy (EngineStrategy): Agent configuration (e.g. CBMC
            search or SMT conclusion check).
            result_trace (ResultTrace): Sink for debug and result output data.
            collect_pyre_type_information (bool): Whether libCST modules should
            be parsed with type information. This incurs a performance overhead,
            but is necessary for the Z3 back-end.
        """
        self.__logger: Logger = logger_factory(__name__)
        self.__engine_strategy: EngineStrategy = engine_strategy
        self.__client = InferenceClient(logger_factory, chat_completion)
        self.__result_trace: ResultTrace = result_trace
        self.__collect_pyre_type_information: bool = collect_pyre_type_information
        self.__expected_solution = expected_solution

    async def solve(self) -> None:
        """
        Solves thetask in the configured engine strategy. This is actually just
        a retry wrapper around `__solve`, where retries are mostly triggered by
        Python or solver syntax errors. In these cases, we trigger a trajectory
        from scratch, to avoid repeating the same sequences.
        """
        attempt: int = 0
        while True:
            attempt_failed: bool
            try:
                attempt_failed = await self.__solve()
            except:
                self.__logger.exception(
                    f"""Unexpected error during solve.
Python Code:
{self.__result_trace.python_code}

Constraints:
{self.__result_trace.solver_constraints}
"""
                )
                attempt_failed = True

            self.__result_trace.messages.extend(self.__client.conversation)
            if not attempt_failed:
                break

            #self.__client.conversation.clear()
            attempt += 1
            if attempt >= RETRY_COUNT:
                break
            self.__logger.warning("Retrying solution finding due to recoverable error.")
            self.__result_trace.num_agent_retries += 1

    async def __solve(self) -> bool:
        """
        Retryable solution attempt. Includes data structure and constraints
        generation, CBMC invocation, CBMC output parsing, and solution
        formatting.

        Returns: `True` if the solution attempt failed due to a flaky error
        that is unlikely to repeat, e.g. syntax errors due to token limits
        spreading Python code across mutltiple messages.
        """
        data_structure: Optional[str] = await self.__generate_data_structure()
        if not data_structure:
            return False

        solution, retry_if_failed = await self.__generate_and_verify_constraints(
            data_structure
        )
        if not solution:
            return retry_if_failed

        await self.__format_solution(solution)
        return False

    async def __generate_data_structure(self) -> Optional[str]:
        """
        Prompts the model to generate the data structure which can contain a
        solution to the puzzle.

        Returns:
            Python data structure that can contain puzzle solutions.
        """
        self.__client.add_message(self.__engine_strategy.system_prompt, Role.SYSTEM)
        self.__client.add_message(
            self.__engine_strategy.data_structure_prompt, Role.USER
        )
        start = time.perf_counter()
        data_structure: Optional[str] = await self.__receive_code_response(
            "data structure"
        )
        end = time.perf_counter()
        self.__result_trace.data_structure_time = end - start
        if data_structure:
            self.__result_trace.python_data_structure = data_structure
            return data_structure
        self.__logger.error("Failed to define solution data structure.")
        return None

    async def __generate_and_verify_constraints(
        self, data_structure: str
    ) -> Tuple[Optional[str], bool]:
        """
        Prompts the model to generate the constraints describing a valid
        solution, then generates a matching solution using CBMC.
        Args:
            data_structure (str): Python data structure for solution type.
        Returns:
            First tuple element will be the solution in the solver's format, if
            it could be successfully generated, otherwise `None`. The second
            tuple element indicates whether we should retry the data structure
            and constraint generation from scratch. This can be useful if the
            model generated Python code with syntax errors.
        """
        attempts: int = 0
        while True:
            all_constraints: list[str] = []
            for constraints_prompt in self.__engine_strategy.constraints_prompt:
                self.__client.add_message(constraints_prompt, Role.USER)
                start = time.perf_counter()
                constraints: Optional[str] = await self.__receive_code_response(
                    "validation function"
                )
                end = time.perf_counter()
                self.__result_trace.constraints_time = end - start
                if constraints is None:
                    self.__logger.error("Failed to define solution constraints.")
                    self.__result_trace.python_code = data_structure
                    return None, False
                all_constraints.append(constraints)

            python_code: str = f"""
{self.__engine_strategy.python_code_prefix}
{data_structure}
{os.linesep.join(all_constraints)}
                              """
            print("PyTHON CODE")
            print(python_code)
            print("\n\n")
            self.__result_trace.python_code = python_code

            module: Module
            metadata: Optional[MetadataWrapper]
            try:
                if self.__collect_pyre_type_information:
                    metadata = await ModuleWithTypeInfoFactory.create_module(
                        python_code
                    )
                    module = metadata.module
                else:
                    start = time.perf_counter()
                    module = parse_module(python_code)
                    metadata = None
                    end = time.perf_counter()
                    self.__result_trace.libcst_time = end - start
            except ParserSyntaxError:
                self.__logger.exception("Parser error when reading constraint")
                self.__result_trace.num_logic_py_syntax_errors += 1
                return None, True
            solver_constraints: str = (
                await self.__engine_strategy.generate_solver_constraints(
                    module, metadata
                )
            )

            print("\n\n")
            print("CONSTRAINT To CBMC : ")
            print(solver_constraints)
            print("\n\n")
            self.__result_trace.solver_constraints = solver_constraints

            solver_input_file_suffix: str = (
                self.__engine_strategy.solver_input_file_suffix
            )
            solver_exit_code: int
            stdout: str
            stderr: str
            async with NamedTemporaryFile(
                mode="w", suffix=solver_input_file_suffix , delete_on_close=False
            ) as file:
                await file.write(solver_constraints)
                await file.close()
                solver_input_file: str = str(file.name)
                print(f"File : {solver_input_file}")
                try:
                    start = time.perf_counter()
                    solver_exit_code, stdout, stderr = await Subprocess.run(
                        *self.__engine_strategy.generate_solver_invocation_command(
                            solver_input_file
                        ),
                        timeout_in_s=_SOLVER_TIMEOUT,
                    )
                    end = time.perf_counter()
                    self.__result_trace.solver_time = end - start
                except TimeoutError:
                    self.__logger.exception(
                        f"""Solver timeout.
Python Code:
{self.__result_trace.python_code}
Constraints:
{self.__result_trace.solver_constraints}
                        """
                    )
                    self.__result_trace.num_solver_timeouts += 1
                    return None, True
                except Exception as e :
                    print("EREEUR Lors de l'execution de cbmc ",e)
                    #finish_reason, ai_response = await self.__chat_completion.create(ERROR_MESSAGE.format(python_code,e))
                    #self.__client.add_message(ai_response)
                    return None, True

            self.__result_trace.solver_output = f"{stdout}{stderr}"
            self.__result_trace.solver_exit_code = solver_exit_code

            #print(f"SOLVER EXIT CODE {solver_exit_code}")
            solver_outcome, output = self.__engine_strategy.parse_solver_output(
                solver_exit_code, stdout, stderr
            )
            match solver_outcome:
                case SolverOutcome.SUCCESS:
                    return output, False
                case SolverOutcome.FATAL:
                    print("Cbmc Erreur")
                    print(stderr)
                    print("\n\n")
                    #finish_reason, ai_response = await self.__chat_completion.create(ERROR_MESSAGE.format(python_code,stderr))
                    #self.__client.add_message(ai_response)
                    #self.__result_trace.num_solver_errors += 1
                    return None, True
                case SolverOutcome.RETRY:
                    attempts += 1
                    if attempts >= RETRY_COUNT:
                        self.__logger.error(
                            "Exceeded retry limit for repairing constraints, giving up."
                        )
                        return None, False

                    self.__result_trace.num_solver_retries += 1
                    self.__client.add_message(
                        self.__engine_strategy.retry_prompt, Role.USER
                    )

    async def __format_solution(self, solution: str) -> None:
        """
        Asks the model to transform the solution provided by the solver, which
        may be expressed as an instance of the data structured defined by the
        model, into the output solution format expected by the benchmark. For
        some engines (e.g. the SMT conclusion check), this step will be skipped,
        because the output by the solver is already the expected output.

        The (potentially reformatted) solution will be written to
        `self.__result_trace.solution`.

        Args:
            solution (str): Solution provided by solver engine, in its format.
        """
        format_prompt: Optional[str] = self.__engine_strategy.get_format_prompt(
            solution
        )
        formatted_solution: Optional[str]
        if format_prompt:
            self.__client.add_message(format_prompt, Role.USER)
            start = time.perf_counter()
            formatted_solution = await self.__receive_code_response(
                "formatted solution"
            )
            end = time.perf_counter()
            self.__result_trace.format_time = end - start
        else:
            formatted_solution = solution

        self.__result_trace.solution = formatted_solution

    async def __receive_code_response(
        self, expected_content_description: str
    ) -> Optional[str]:
        """
        Submits the conversation, and attempts to extract a code snippet from
        the response. Will send a retry message a limited number of times if no
        code snippet was found in the response.

        Args:
            expected_content_description (str): If no code snippet was found, we
            use this description in the retry message. An example would be "data
            structure", where we would tell the model that no "data structure
            code snippet" was found and that it should regenerate it.
        Returns:
            A code snippet response from the model, if found within the retry
            limit.
        """
        attempt: int = 0
        while True:
            response_text: Optional[str] = await self.__client.send()
            if response_text is None:
                return None

            code: Optional[str] = LogicAgent.__extract_code(response_text)
            if code:
                return code

            attempt += 1
            if attempt >= RETRY_COUNT:
                return None
            self.__client.add_message(
                NO_CODE_FOUND_MESSAGE.format(expected_content_description), Role.USER
            )

    @staticmethod
    def __extract_code(response_text: str) -> Optional[str]:
        """
        Extracts code marked with ``` prefix and suffix from a message.

        Args:
            response_text (str): Model response text containing code to extract,
            potentially surrounded by unrelated description text by the model.
        Returns:
            Single code snippet extracted from the response text.
        """
        num_code_markers: int = len(CODE_MARKER_PATTERN.findall(response_text))
        if num_code_markers > 2:
            code_marker: Optional[Match] = CODE_MARKER_PATTERN.search(response_text, 0)
            if code_marker:
                pos: int = code_marker.start() + 1
                for _ in range(2, num_code_markers):
                    code_marker = CODE_MARKER_PATTERN.search(response_text, pos)
                    if not code_marker:
                        break

                    response_text = (
                        response_text[: code_marker.start()]
                        + response_text[code_marker.end() :]
                    )
                    pos = code_marker.start() + 1

        groups: Optional[Match] = CODE_EXTRACTION_PATTERN.match(response_text)
        return groups.group(1) if groups is not None else None

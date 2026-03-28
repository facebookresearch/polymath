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

from agent.logic.engine_strategy import EngineStrategy, SolverOutcome, SolverConstraints
from agent.logic.solver import Solver
from aiofiles.tempfile import NamedTemporaryFile
from concurrency.subprocess import Subprocess
from inference.chat_completion import ChatCompletion, Role
from inference.client import InferenceClient

from judge.result_trace import ResultTrace


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
        """
        self.__logger: Logger = logger_factory(__name__)
        self.__engine_strategy: EngineStrategy = engine_strategy
        self.__client = InferenceClient(logger_factory, chat_completion)
        self.__result_trace: ResultTrace = result_trace

    async def solve(self) -> None:
        """
        Solves the task in the configured engine strategy. This is actually just
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

            self.__client.conversation.clear()
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
        data_structure: Optional[str] = await self.__receive_code_response(
            "data structure"
        )
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
                constraints: Optional[str] = await self.__receive_code_response(
                    "validation function"
                )
                if constraints is None:
                    self.__logger.error("Failed to define solution constraints.")
                    self.__result_trace.python_code = data_structure
                    return None, False
                all_constraints.append(constraints)

            code: str = f"""
{self.__engine_strategy.python_code_prefix}
{data_structure}
{os.linesep.join(all_constraints)}
"""
            self.__result_trace.python_code = code

            try:
                solverSpec: Optional[SolverConstraints] = (
                await self.__engine_strategy.generate_solver_constraints(
                   code
                )
            )

            except Exception as e:
                self.__logger.exception(f"Parser error when reading constraint: {e}")
                self.__result_trace.num_logic_py_syntax_errors += 1
                return None, True

            if not solverSpec:
                return None, False

            solver_constraints = solverSpec.content
            self.__result_trace.solver_constraints = solver_constraints

            solver_input_file_suffix: str = (
                self.__engine_strategy.solver_input_file_suffix
            )
            solver_exit_code: int
            stdout: str
            stderr: str
            async with NamedTemporaryFile(
                mode="w", suffix=solver_input_file_suffix, delete=False
            ) as file:
                await file.write(solver_constraints)
                await file.close()

                solver_input_file: str = str(file.name)
                try:
                    solver_exit_code, stdout, stderr = await Subprocess.run(
                        *self.__engine_strategy.generate_solver_invocation_command(
                            solver_input_file
                        ),
                        timeout_in_s=_SOLVER_TIMEOUT,
                    )
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

            self.__result_trace.solver_output = f"{stdout}{stderr}"
            self.__result_trace.solver_exit_code = solver_exit_code
            if not stdout:
                return None, True

            solver_outcome, output = self.__engine_strategy.parse_solver_output(
                solver_exit_code, solverSpec, stdout, stderr
            )
            match solver_outcome:
                case SolverOutcome.SUCCESS:
                    return output, False
                case SolverOutcome.FATAL:
                    self.__result_trace.num_solver_errors += 1
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
            formatted_solution = await self.__receive_code_response(
                "formatted solution"
            )
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

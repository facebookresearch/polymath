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
        self.__allow_revise_prompt = True
        self.__retrie_with_revised_prompt = False
        self.__enter_revise = False
        self.__num_revise = 0
        self.__max_num_revise = 3
        self.__curr_temperature = 0
        self.__max_temperature = 1



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

            if not self.__retrie_with_revised_prompt:
                attempt += 1
                if attempt >= RETRY_COUNT:
                    break
            else:
                self.__client.conversation.clear()
                self.__num_revise += 1
                self.__retrie_with_revised_prompt = False
                if self.__num_revise >= self.__max_num_revise:
                    if self.__curr_temperature < self.__max_temperature :
                        self.__curr_temperature += self.__max_temperature / 3
                        self.__client.set_temperature(self.__curr_temperature)
                        self.__num_revise = 0
                        self.__engine_strategy.reset_constraints_prompt()
                    else:
                        break

            self.__logger.warning("Retrying solution finding due to recoverable error.")
            self.__result_trace.num_agent_retries += 1

        self.__client.reset_temperature()

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

        self.__result_trace.solution = await self.__format_solution(solution)

        if self.__allow_revise_prompt:
            success, err = LogicAgent.__compare_solutions(self.__expected_solution, self.__result_trace.solution)
            if not success:
                self.__enter_revise = True
                self.__logger.warning("Retrying prompt revising due to incorrect result.")
                prompt = self.__engine_strategy.get_revise_prompt(
                        os.linesep.join(self.__engine_strategy.constraints_prompt),
                        self.__result_trace.python_code,
                        err
                )
                message = Message(Role.USER,prompt)
                finish_reason, ai_response = await self.__client.create(message)
                self.__engine_strategy.set_constraints_prompt(ai_response)
                self.__retrie_with_revised_prompt = True
                return True

            if self.__enter_revise:
                self.__result_trace.revise_success = True
                self.__engine_strategy.set_initial_constraints_prompt(os.linesep.join(self.__engine_strategy.constraints_prompt))


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

            constraints = os.linesep.join(all_constraints)
            if self.__engine_strategy.data_structure_included(contraints):
                python_code: str = f"""
{self.__engine_strategy.python_code_prefix}
{constraints}
"""
            else :
                python_code: str = f"""
{self.__engine_strategy.python_code_prefix}
{data_structure}
{constraints}
"""
            print(f"PYTHON CODE \n{python_code}")

            self.__result_trace.python_code = python_code

            try:
                start = time.perf_counter()
                preprocessed: str = await self.__engine_strategy.generate_solver_constraints(
                    python_code
                )
                end = perf_counter()
                self.__result_trace.libcst_time = end - start
            except ParserSyntaxError:
                self.__logger.exception("Parser error when reading constraint")
                self.__result_trace.num_logic_py_syntax_errors += 1
                return None, True

            solver_constraints = preprocessed[0]
            print(f"CONSTRAINT: {solver_constraints}")

            self.__result_trace.solver_constraints = solver_constraints

            solver_input_file_suffix: str = self.__engine_strategy.solver_input_file_suffix

            solver_exit_code: int
            stdout: str
            stderr: str
            async with NamedTemporaryFile(
                mode="w", suffix=solver_input_file_suffix , delete_on_close=False
            ) as file:
                await file.write(solver_constraints)
                await file.close()
                solver_input_file: str = str(file.name)
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
                    finish_reason, ai_response = await self.__client.create(Message(Role.USER,ERROR_MESSAGE.format(python_code,e)))
                    self.__client.add_message(ai_response,Role.USER)
                    return None, True

            self.__result_trace.solver_output = f"{stdout}\n{stderr}"
            self.__result_trace.solver_exit_code = solver_exit_code

            solver_outcome, output = self.__engine_strategy.parse_solver_output(
                solver_exit_code, (stdout, *preprocessed[1:] ), stderr
            )
            match solver_outcome:
                case SolverOutcome.SUCCESS:
                    return output, False
                case SolverOutcome.FATAL:
                    if self.__allow_revise_prompt:
                        self.__logger.warning(f"Retrying prompt revising due to an Error : \n {stderr}.")
                        prompt = self.__engine_strategy.get_revise_prompt(
                            os.linesep.join(self.__engine_strategy.constraints_prompt),
                            self.__result_trace.python_code,
                            stderr
                        )
                        self.__retrie_with_revised_prompt = True
                    else :
                        prompt = ERROR_MESSAGE.format(python_code,stderr)

                    finish_reason, ai_response = await self.__client.create(Message(Role.USER,prompt))

                    if self.__allow_revise_prompt:
                        self.__engine_strategy.set_constraints_prompt(ai_response)
                    else:
                        self.__client.add_message(ai_response,Role.USER)
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
            start = time.perf_counter()
            formatted_solution = await self.__receive_code_response(
                "formatted solution"
            )
            end = time.perf_counter()
            self.__result_trace.format_time = end - start
        else:
            formatted_solution = solution

        return  formatted_solution

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
        if groups is None :
            try:
                return json.loads(response_text)
            except:
                return None

        return groups.group(1)

    @staticmethod
    def __compare_solutions(solution, outcome):
        """
        Compare deux solutions de puzzle logique et affiche les différences.
        Retourne True si elles sont identiques, False sinon.
        """
        if not outcome:
            return False, ["Erreur : outcome est vide ou None"]

        rows = solution["rows"]
        header = solution["header"]

        # Convertir solution1 en dict comparable
        expected = {}
        for idx, row in enumerate(rows):
            house_id = f"House {idx+1}"
            expected[house_id] = {header[i]: row[i] for i in range(1, len(header))}

        try:
            outcome = json.loads(outcome)
        except:
            return None,None

        success = True
        res = []
        for house_id, expected_attrs in expected.items():
            given_attrs = outcome["solution"][house_id]
            for key, expected_value in expected_attrs.items():
                given_value = given_attrs.get(key)
                if expected_value.replace(" ", "").replace("_", "").lower() != given_value.replace(" ","").replace("_","").lower():
                    res.append(f"❌ Erreur dans {house_id}, champ '{key}': attendu '{expected_value}', obtenu '{given_value}'")
                    success = False

        if success:
            return True , [f"✅ Les deux solutions sont identiques."]

        return success, res

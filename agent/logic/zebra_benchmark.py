# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from asyncio import Lock, run
from io import StringIO
from json import dumps, JSONDecodeError, loads
from logging import Logger
import os
from os import path
from re import compile, fullmatch, Match, Pattern
from types import TracebackType
from typing import Any, Callable, Optional, Tuple
import time
import aiofiles

from agent.logic.agent import LogicAgent
from agent.logic.engine_strategy_factory import EngineStrategyFactory
from agent.logic.engine_strategy_factory import PrologStrategyFactory
from agent.logic.engine_strategy_factory import CbmcStrategyFactory
from agent.logic.engine_strategy import EngineStrategy
from agent.logic.model_only import ModelOnlySolver
from aiofiles.base import AiofilesContextManager
from aiofiles.threadpool.text import AsyncTextIOWrapper
from concurrency.async_pool import AsyncPool

from dotenv import load_dotenv
from inference.chat_completion import Message, Role
from inference.chat_completion import ChatCompletion
from inference.chat_completion_factory import create_chat_completion
from judge.result_trace import ResultTrace
from logger.logger_factory import LoggerFactory
from pyarrow import Table
from pyarrow.parquet import ParquetFile

from training.cbmc_scorer_vc_factory import CBMCScorerVerificationConstraintsFactory
from training.logic_rl_training_dialog import LogicRlTrainingDialog
from training.lowercase_json import LowerCaseJson
from training.sample_output_converter import SampleOutputConverter
from training.sample_output_converter_factory import create_sample_output_converter

from agent.logic.analysis.solution_comparator import SolutionComparator
from agent.logic.analysis.zebra_comparator import ZebraSolutionComparator
from agent.logic.error_handler import ErrorHandler

# Used to extract the house number from a formatted solution.
_ZERO_EVAL_HOUSE_NAME_PATTERN: Pattern = compile("House ([\\d]*)")


class ZebraBenchmark:
    """
    Benchmark runner for Zebra Grid puzzles.
    """

    def __init__(
        self,
        eval_json_file_name: str,
        generator: str,
        model_name: str,
        solver_factory: EngineStrategyFactory,
        enable_stderr_log: bool = True,
        generate_training_data: bool = False,
        zebra_input_dataset_path: Optional[str] = None,
        model_only: bool = False,
        filter_dataset: Callable[[dict[str, Any]], bool] = lambda task: True,
    ) -> None:
        """
        Args:
            eval_json_file_name (str): Result JSON output file.
            generator (str): Generator name used in ZebraLogic leader board.
            model_name (str): Name of model to use in inference client.
            enable_stderr_log (bool): Indicates whether log messages should be
            writting to stderr as well as the result JSON. Disabled for unit
            tests.
            generate_training_data (bool): Indicates whether we are evaluating
            against a Zebra benchmark, or whether we are generating training
            data using Logic.py.
            zebra_input_dataset_path (Optional[str]): Zebra logic dataset used
            as a benchmark or to generate training data. If not set, the
            default ZebraLogicBench dataset is used.
            model_only (bool): Indicates to not use the Logic agent, but instead
            perform a model-only baseline comparison run.
            filter_dataset (Callable[[Any], Any]): Filter to select which tasks
            from the benchmark set to run.
        """
        self.__eval_json_file_name = eval_json_file_name
        self.__output_dataset_lock = Lock()
        self.__generator: str = generator
        self.__solver_factory = solver_factory
        self.__model_name: str = model_name
        self.__model_only: bool = model_only
        self.__enable_stderr_log: bool = enable_stderr_log
        self.__filter_dataset: Callable[[dict[str, Any]], bool] = filter_dataset
        self.__output_dataset_context: Optional[AiofilesContextManager] = (
            aiofiles.open(
                self.__eval_json_file_name.replace(".json", "-dataset.json"), "w"
            )
            if generate_training_data
            else None
        )
        self.__sample_output_converter: SampleOutputConverter = (
            create_sample_output_converter()
        )

        if zebra_input_dataset_path is not None:
            self.__zebra_input_dataset_path: str = zebra_input_dataset_path
        else:
            module_path: str = path.dirname(__file__)
            self.__zebra_input_dataset_path: str = path.join(
                module_path, "../../datasets/grid_mode/test-00000-of-00001.parquet"
            )
        self.__chat_completion_1: ChatCompletion = None
        self.__logger_factory = None
        self.zebra_comparator: SolutionComparator = ZebraSolutionComparator()

    async def __aenter__(self) -> "ZebraBenchmark":
        if self.__output_dataset_context:
            self.__output_dataset: AsyncTextIOWrapper = (
                await self.__output_dataset_context.__aenter__()
            )

        # crée le logger temporaire pour init chat_completion
        log_stream = StringIO()
        self.__logger_factory = LoggerFactory(log_stream, self.__enable_stderr_log)

        self.__chat_completion_1 = await create_chat_completion(
            self.__logger_factory, self.__model_name, gpu_id= 0
        ).__aenter__()

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self.__output_dataset_context:
            await self.__output_dataset_context.__aexit__(exc_type, exc_value, exc_tb)

        if self.__chat_completion_1:
           await self.__chat_completion_1.__aexit__(exc_type, exc_value, exc_tb)

        if self.__logger_factory:
            self.__logger_factory.__exit__(exc_type, exc_value, exc_tb)

    async def run(self) -> None:
        """
        Runs all Zebra puzzles specified in the input dataset JSON file.
        """
        pool = AsyncPool(1)
        eval_json: list[dict[str, Any]] = []
        dataset = ParquetFile(self.__zebra_input_dataset_path)
        for row_group_index in range(dataset.num_row_groups):
            table: Table = dataset.read_row_group(row_group_index)
            rows: list[dict[str, Any]] = table.to_pylist()
            for i, task in enumerate(rows):
                model = self.__chat_completion_1
                if self.__filter_dataset(task):
                    await pool.submit(lambda task=task: self.run_task(eval_json, task, model))
        await pool.gather()

        if not self.__output_dataset_context:
            async with aiofiles.open(self.__eval_json_file_name, "w") as file:
                await file.write(dumps(eval_json, indent=4))

    async def run_task(
        self,
        eval_json: list[dict[str, Any]],
        task: dict[str, Any],
        model: ChatCompletion ,
    ) -> None:
        """
        Executes a single benchmark task and stores the result in `eval_json`.
        """
        task_id: str = task["id"]
        puzzle: str = task["puzzle"]
        expected_solution: Any = task["solution"]
        output_format: str = ZebraBenchmark.get_format(expected_solution)

        result_trace = ResultTrace(task_id)
        engine_strategy : EngineStrategy = self.__solver_factory.create(
            self.__logger_factory, puzzle, output_format
        )
        error_handler = ErrorHandler(engine_strategy, model)
        solver = (
                ModelOnlySolver(
                    self.__logger_factory,
                    model,
                    result_trace,
                    puzzle,
                    output_format,
                )
                if self.__model_only
                else LogicAgent(
                        self.__logger_factory, model, engine_strategy, result_trace,
                        expected_solution = expected_solution,
                        error_handler = error_handler,
                )
        )
        await solver.solve()
        if self.__output_dataset_context:
            await self.write_sample(
                task_id, puzzle, result_trace, expected_solution, self.__logger_factory
            )


        if not self.__output_dataset_context:
            success, nb_err, _ = self.zebra_comparator.compare(expected_solution, result_trace.solution,display=True, task_id=task_id )
            eval_json.append(
                {
                    "session_id": task_id,
                    "chat_history": [message.text for message in result_trace.messages],
                    "model_input": "n/a",
                    "output": [result_trace.solution],
                    "debug_output": [f"{repr(result_trace)}"],
                    "generator": self.__generator,
                    "configs": {},
                    "dataset": "zebra-grid",
                    "id": task_id,
                    "size": task["size"],
                    "puzzle": puzzle,
                    "created_at": task["created_at"],
                    "success":success,
                    "nb_err": nb_err,
                    "better_prompt": engine_strategy.constraints_prompt if result_trace.revise_success else None,
                    "better_tempertature":solver.__curr_temperature if result_trace.revise_success else None ,
                    "polymath_metadata": {
                        "num_agent_retries": result_trace.num_agent_retries,
                        "num_logic_py_syntax_errors": result_trace.num_logic_py_syntax_errors,
                        "num_solver_errors": result_trace.num_solver_errors,
                        "num_solver_retries": result_trace.num_solver_errors,
                        "num_solver_timeouts": result_trace.num_solver_timeouts,
                        "constraints": result_trace.solver_constraints,
                        "python-code":result_trace.python_code,
                    },
                    "time":{
                        "data_structure_time" : result_trace.data_structure_time,
                        "constraints_time" : result_trace.constraints_time,
                        "solver_time" : result_trace.solver_time,
                        "libcst_time" : result_trace.libcst_time,
                        "format_time" : result_trace.format_time,
                    },
                }
            )

    async def write_sample(
        self,
        task_id: str,
        puzzle: str,
        result_trace: ResultTrace,
        expected_solution: Any,
        logger_factory: Callable[[str], Logger],
    ) -> None:
        """
        When generating training data, this method generates a simple RL
        training sample based on a correctly solved benchmark task.

        Args:
            task_id (str): ID of benchmark task on which the generated sample is
            based. This is mostly used for logging.
            puzzle (str): Puzzle being solved. This is used to generate the
            dialog in the training sample.
            result_trace (ResultTrace): Contains the solution data structure,
            the generated solution, and the solver constraints. All of these are
            used to generate scorer VCs.
            expected_solution (Any): Ground truth to filter out incorrect VCs
            from training data.
            logger_factory (Callable[[str], Logger]): Used to write logs from
            various sources
        """
        logger: Logger = logger_factory(__name__)
        try:
            hugging_face_solution: Optional[dict[str, Any]] = (
                ZebraBenchmark.convert_to_reference_solution_format(
                    result_trace.solution
                )
            )
        except JSONDecodeError:
            logger.exception(
                f"""Discarding task `{task_id}` due to JSON parser error in solution:
```
{result_trace.solution}
```"""
            )
            return

        if not LowerCaseJson.are_equal(hugging_face_solution, expected_solution):
            logger.warning(
                f"""Discarding task `{task_id}` due to incorrect solution.
Expected:
```
{dumps(expected_solution)}
```
Actual:
```
{dumps(hugging_face_solution)}
```
"""
            )
            return

        if not result_trace.python_data_structure:
            logger.error(f"Discarding task `{task_id}` due to missing data structure.")
            return

        if not result_trace.solver_constraints:
            logger.error(
                f"Discarding task `{task_id}` due to missing solver synthesis constraints."
            )
            return

        cbmc_scorer_vc_factory = CBMCScorerVerificationConstraintsFactory(
            logger_factory
        )
        scorer_vc: Optional[str] = await cbmc_scorer_vc_factory.convert(
            result_trace.solver_constraints
        )
        if not scorer_vc:
            logger.error(
                f"Discarding task `{task_id}` due to failed scorer VC creation."
            )
            return

        dialog: list[Message] = LogicRlTrainingDialog.create(
            puzzle, result_trace.python_data_structure
        )
        metadata: dict[str, Any] = {"scorer_vc": scorer_vc}
        sample: Any = self.__sample_output_converter.convert(dialog, metadata)
        line: str = f"{dumps(sample)}\n"
        async with self.__output_dataset_lock:
            await self.__output_dataset.write(line)

    @staticmethod
    def get_format(solution_placeholder: dict[str, Any]) -> str:
        """
        Converts a solution placeholder from the Zebra dataset to the correct
        template of the expected output format to pass to the model.
        """
        num_houses: int = len(solution_placeholder["rows"])
        headers: list[str] = solution_placeholder["header"]
        house_prefix: str = headers[0]
        solution_format: dict[str, dict[str, str]] = {}
        for house_number in range(1, num_houses + 1):
            house_format: dict[str, str] = {}
            for header in headers[1:]:
                house_format[header] = "___"
            solution_format[f"{house_prefix} {house_number}"] = house_format

        solution_container: dict[str, dict[str, dict[str, str]]] = {}
        solution_container["solution"] = solution_format
        return dumps(solution_container, indent=4)

    @staticmethod
    def convert_to_reference_solution_format(
        solution: Optional[str],
    ) -> Optional[dict[str, Any]]:
        """
        ZebraLogicBench has two solution formats: The HuggingFace dataset is
        formatted in what we call the "reference" solution format, and the
        ZeroEval benchmark evaluation tool accepts another format. This class
        produces the latter, since we need to evaluate using ZeroEval. However,
        when checking whether our scorer VCs are correct, we compare against the
        ground truth in the HuggingFace dataset directly. To do so, we use this
        function to convert the ZeroEval solution.

        Args:
            solution (str): Solution in ZeroEval format.
        Returns:
            solution in HuggingFace ZebraLogicBench format.
        """
        if not solution:
            return None

        header: list[str] = ["House"]
        is_first_house: bool = True
        zero_eval_solution: Any = loads(solution)["solution"]
        num_houses: int = len(zero_eval_solution)
        rows: list[list[str]] = [[] for _ in range(num_houses)]
        for house_name, house in zero_eval_solution.items():
            parsed: Optional[Match] = fullmatch(
                _ZERO_EVAL_HOUSE_NAME_PATTERN, house_name
            )
            if not parsed:
                return None

            house_number: str = parsed.group(1)
            house_index: int = int(house_number) - 1
            row: list[str] = rows[house_index]
            row.append(house_number)
            for name, value in house.items():
                row.append(value)
                if is_first_house:
                    header.append(name)
            is_first_house = False
        return {
            "header": header,
            "rows": rows,
        }


async def main():
    load_dotenv()
    module_path: str = path.dirname(__file__)
    base_path: str = path.abspath(
        path.join(module_path, "../../lib/ZeroEval/result_dirs/zebra-grid/")
    )
    models: list[Tuple[str, str, str]] = [
        # (
        #     path.join(base_path, "Meta-Llama-4-Polymath@model-only.json"),
        #     "meta-llama/Meta-Llama-4-Polymath@model-only",
        #     "llama4-polymath",
        # ),
        # (
        #     path.join(base_path, "Meta-Llama-4-Polymath-Syntactic@model-only.json"),
        #     "meta-llama/Meta-Llama-4-Polymath-Syntactic@model-only",
        #     "llama4-polymath-syntactic",
        # ),
        # (
        #     path.join(base_path, "Meta-Llama-4-Base@model-only.json"),
        #     "meta-llama/Meta-Llama-4-Base@model-only",
        #     "llama4-base",
        # ),
        # (
        #     path.join(base_path, "Meta-Llama-3.1-70B-Instruct@reasoning.json"),
        #     "meta-llama/Meta-Llama-3.1-70B-Instruct@reasoning",
        #     "llama3.1-70b-instruct",
        # ),
        # (
        #     path.join(base_path, "Claude-3.5-Sonnet-20241022@reasoning.json"),
        #     "anthropic/Claude-3.5-Sonnet-20241022@reasoning",
        #     "claude-3-5-sonnet-20241022",
        # ),
        # (
        #     path.join(base_path, "gpt-o3-mini@reasoning.json"),
        #     "openai/GPT-o3-mini@reasoning",
        #     "gpt-o3-mini",
        # ),
        #(
        #    path.join(base_path, "gpt-4o@reasoning-debug.json"),
        #    "openai/GPT-4o@reasoning-debug",
        #    "gpt-4o-evals2",
        #),

############
##        ##
##  groq  ##
##        ##
############
          #(
          #  path.join(base_path, "deepseek@reasoning.json"),
          #  "deepseek-r1-distill-llama-70b@reasoning",
          #  "deepseek-r1-distill-llama-70b",
          #),
         #(
         #   path.join(base_path, "LLaMA-3-70B@reasoning.json"),
         #   "meta-llama/LLaMA-3-70B@reasoning",
         #   "llama3-70b-8192",
         #),
         #(
         #    path.join(base_path, "LLaMA-3-8B_23_06@reasoning.json"),
         #    "meta-llama/LLaMA-3.1-8B@reasoning",
         #    "llama-3.1-8b-instant",
         #),
        #(
        #  path.join(module_path, "LLaMA-3.3-70B-Versatile-5*6-26@reasoning.json"),
        # "metadata-llama/LLaMA-3.3-70B-Versatile@reasoning",
        #   "llama-3.3-70b-versatile",
        #),

## Mistral Local
	#(
        #  path.join(module_path, "Mistral-7B-v0.3-local@Instruct.json"),
        # "mistral_models/Mistral-7B-v0.3@Instruct",
        # "/srv/data/mistral_models/7B-Instruct-v0.3",
        #),

##Llama Local
        (
          path.join(module_path, "Llama-3.3-70B@Instruct-Review.json"),
         "Llama_models/Llama-70B-v3.3@Instruct",
         "/srv/data/Llama-3.3-70B-Instruct",
       ),
    ]
    Solver_factory: EngineStrategyFactory = PrologStrategyFactory()
    for model in models:
        async with ZebraBenchmark(
                model[0],
                model[1],
                model[2],
                solver_factory = Solver_factory,
                zebra_input_dataset_path = path.join(module_path,"dataset/test-00000-of-00001.parquet")
        ) as benchmark:
            await benchmark.run()


if __name__ == "__main__":
    run(main())

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import os
from dataclasses import asdict, dataclass, field
from io import StringIO
from json import dumps, loads
from typing import Any, Callable, Optional

import aiofiles

from agent.logic.agent import LogicAgent
from agent.logic.engine_strategy import EngineStrategy
from agent.logic.z3_conclusion_check_engine_strategy import (
    Z3ConclusionCheckEngineStrategy,
)
from concurrency.async_pool import AsyncPool
from dotenv import load_dotenv
from inference.chat_completion_factory import create_chat_completion
from judge.result_trace import ResultTrace
from logger.logger_factory import LoggerFactory


@dataclass
class FOLIOResultTrace:
    """
    Tuple of FOLIO benchmark task and the result trace our engine produced.
    """

    # Original FOLIO benchmark task JSON.
    task: Any

    # Result trace produced by Logic.py engine.
    result_trace: Optional[ResultTrace]


@dataclass
class FOLIOBenchmarkResult:
    """
    Contains all individual task results and final benchmark score.
    """

    # All engine result traces per FOLIO benchmark.
    traces: list[FOLIOResultTrace] = field(default_factory=list)

    # Share of solutions matching the expected ground truth.
    score: float = 0.0


class FOLIOBenchmark:
    """
    Benchmark runner for FOLIO dataset.
    """

    def __init__(
        self,
        input_dataset_path: str,
        model_name: str,
        enable_stderr_log: bool = True,
        filter_dataset: Callable[[list[Any]], list[Any]] = lambda dataset: dataset,
    ) -> None:
        """
        Args:
            input_dataset_path (str): Path to FOLIO benchmark tasks in `.jsonl`
            format.
            model_name (str): Name of model to use in inference client.
            enable_stderr_log (bool): Indicates whether log messages should be
            writting to stderr as well as the result JSON. Disabled for unit
            tests.
            filter_dataset (Callable[[Any], Any]): Filter to select which tasks
            from the benchmark set to run.
        """
        self.__enable_stderr_log: bool = enable_stderr_log
        self.__filter_dataset: Callable[[list[Any]], list[Any]] = filter_dataset
        self.__input_dataset_path: str = input_dataset_path
        self.__model_name: str = model_name

    async def generate(self) -> FOLIOBenchmarkResult:
        """
        Runs the logic agent on all configured benchmark tasks.

        Returns:
            Unsocred benchmark results containing result traces per task.
        """

        folio_input_dataset: list[Any] = []
        async with aiofiles.open(self.__input_dataset_path, "r") as file:
            async for line in file:
                folio_input_dataset.append(loads(line))

        pool = AsyncPool(10)
        folio_input_dataset = self.__filter_dataset(folio_input_dataset)
        traces: dict[int, ResultTrace] = {}
        for task in folio_input_dataset:
            await pool.submit(lambda task=task: self.run_task(traces, task))
        await pool.gather()

        result = FOLIOBenchmarkResult()
        for task in folio_input_dataset:
            id: int = FOLIOBenchmark.get_task_id(task)
            result.traces.append(FOLIOResultTrace(task, traces.get(id)))
        return result

    def evaluate(self, result: FOLIOBenchmarkResult) -> None:
        """
        Calculates and sets the `score` field of the given FOLIO benchmark
        result.

        Args:
            result (FOLIOBenchmarkResult): Result file to score.
        """
        number_of_correct_answers: int = 0
        for trace in result.traces:
            label: str = trace.task["label"]
            result_trace: Optional[ResultTrace] = trace.result_trace
            if result_trace and result_trace.solution == label:
                number_of_correct_answers += 1

        result.score = number_of_correct_answers / len(result.traces)

    async def run_task(self, traces: dict[int, ResultTrace], task: Any) -> None:
        """
        Runs a single FOLIO benchmark task.

        Args:
            traces (dict[int, ResultTrace]): Result output parameter into which
            to insert the result of this task run.
            task (Any): FOLIO benchmark JSON task to run.
        """
        task_id: int = FOLIOBenchmark.get_task_id(task)
        premise: str = task["premises"]
        conclusion: str = task["conclusion"]

        log_stream = StringIO()
        result_trace = ResultTrace(task_id)
        with LoggerFactory(log_stream, self.__enable_stderr_log) as logger_factory:
            engine_strategy: EngineStrategy = Z3ConclusionCheckEngineStrategy(
                logger_factory, premise, conclusion
            )
            async with create_chat_completion(
                logger_factory, self.__model_name
            ) as chat_completion:
                agent = LogicAgent(
                    logger_factory, chat_completion, engine_strategy, result_trace, True
                )
                await agent.solve()
        traces[task_id] = result_trace

    @staticmethod
    def get_task_id(folio_task: Any) -> int:
        """
        Helper to get a unique ID for a FOLIO task.

        Returns:
            Unique ID for FOLIO benchmark task.
        """
        return folio_task["example_id"]


async def main():
    load_dotenv()
    module_path: str = os.path.dirname(__file__)
    input_dataset_path: str = os.path.join(
        module_path, "../../datasets/folio_v2_validation.jsonl"
    )
    models: list[str] = [
        "llama3.1-70b-instruct",
        # "gpt-4o-evals2",
        # "claude-3-5-sonnet-20241022",
        # "claude-3-5-sonnet-20241022-genai",
        # "claude-3-7-sonnet-20250219-us-genai",
    ]
    for model in models:
        benchmark = FOLIOBenchmark(
            input_dataset_path,
            model,
        )
        result: FOLIOBenchmarkResult = await benchmark.generate()
        benchmark.evaluate(result)
        json_result: dict[str, Any] = asdict(result)
        output_text: str = dumps(json_result, indent=4)

        async with aiofiles.open(f"folio-results-{model}.json", "w") as file:
            await file.write(output_text)


if __name__ == "__main__":
    asyncio.run(main())

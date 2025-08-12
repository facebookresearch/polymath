# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional
from unittest import IsolatedAsyncioTestCase

from agent.logic.folio_benchmark import FOLIOBenchmark, FOLIOBenchmarkResult

from dotenv import load_dotenv

from judge.result_trace import ResultTrace


load_dotenv()


class TestFOLIOBenchmark(IsolatedAsyncioTestCase):

    async def test_end_to_end_bird_false(self) -> None:
        await self.__test_end_to_end(736, "False", "claude-3-5-sonnet-20241022")

    async def test_end_to_end_cat_uncertain(self) -> None:
        await self.__test_end_to_end(700, "Uncertain", "claude-3-5-sonnet-20241022")

    async def test_end_to_end_lunch_uncertain(self) -> None:
        await self.__test_end_to_end(1208, "Uncertain", "claude-3-5-sonnet-20241022")

    async def test_end_to_end_talent_true(self) -> None:
        await self.__test_end_to_end(1015, "True", "claude-3-5-sonnet-20241022")

    async def test_end_to_end_monkeypox_true(self) -> None:
        await self.__test_end_to_end(563, "True", "claude-3-5-sonnet-20241022")

    async def test_end_to_end_square_true(self) -> None:
        await self.__test_end_to_end(726, "True", "claude-3-5-sonnet-20241022")

    async def test_end_to_end_wrestling_true(self) -> None:
        await self.__test_end_to_end(289, "True", "claude-3-5-sonnet-20241022")

    async def test_end_to_end_china_true(self) -> None:
        await self.__test_end_to_end(657, "True", "claude-3-5-sonnet-20241022")

    async def __test_end_to_end(
        self, example_id: int, expected_solution: str, model: str
    ) -> None:
        module_path: str = os.path.dirname(__file__)
        input_dataset_path: str = os.path.join(
            module_path, "../../../datasets/folio_v2_validation.jsonl"
        )
        benchmark = FOLIOBenchmark(
            input_dataset_path,
            model,
            True,
            lambda dataset: [
                task for task in dataset if task["example_id"] == example_id
            ],
        )
        results: FOLIOBenchmarkResult = await benchmark.generate()
        trace: Optional[ResultTrace] = next(
            (
                trace.result_trace
                for trace in results.traces
                if FOLIOBenchmark.get_task_id(trace.task) == example_id
            ),
        )

        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(expected_solution, trace.solution)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from unittest import TestCase

from agent.logic.model_only import ModelOnlySolver


EXPECTED_JSON: str = """
{
  "name": "Jane"
}
"""


class TestModelOnlySolver(TestCase):

    def test_parse_scorer_response_regular(self) -> None:
        self.__test_parse_scorer_response(
            f"""# Thought
Whatever...

# Solution
{EXPECTED_JSON}""",
        )

    def test_parse_scorer_response_no_thought(self) -> None:
        self.__test_parse_scorer_response(
            f"""# Solution
{EXPECTED_JSON}""",
        )

    def test_parse_scorer_response_highlighting(self) -> None:
        self.__test_parse_scorer_response(
            f"""# Solution
```json{EXPECTED_JSON}```""",
        )

    def test_parse_scorer_response_json_only(self) -> None:
        self.__test_parse_scorer_response(EXPECTED_JSON)

    def test_parse_scorer_response_json_only_highlighting(self) -> None:
        self.__test_parse_scorer_response(f"```json{EXPECTED_JSON}```")

    def test_parse_scorer_response_reasoning_thinking(self) -> None:
        self.__test_parse_scorer_response(
            f"""<|reasoning_thinking_start|>Reasoning, apprently...<|reasoning_thinking_end|># Solution
```json{EXPECTED_JSON}```"""
        )

    def test_parse_scorer_response_mismatch(self) -> None:
        self.__test_parse_scorer_response(
            f"""# Not a solution
{EXPECTED_JSON}""",
            None,
        )

    def test_parse_scorer_response_reasoning_thinking_raw_json(self) -> None:
        self.__test_parse_scorer_response(
            f"<|reasoning_thinking_start|>Reasoning, apprently...<|reasoning_thinking_end|>{EXPECTED_JSON}"
        )

    def test_parse_scorer_response_mismatch(self) -> None:
        self.__test_parse_scorer_response(
            f"""# Not a solution
{EXPECTED_JSON}""",
            None,
        )

    def __test_parse_scorer_response(
        self, response: str, expected_json: Optional[str] = EXPECTED_JSON
    ) -> None:
        actual_json: str = ModelOnlySolver._ModelOnlySolver__parse_scorer_response(
            response
        )
        self.assertEqual(expected_json, actual_json)

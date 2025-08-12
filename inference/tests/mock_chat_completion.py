# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from types import TracebackType
from typing import Optional, Tuple

from inference.chat_completion import ChatCompletion
from inference.finish_reason import FinishReason


class MockChatCompletion(ChatCompletion):

    def __init__(self, answers: list[str]) -> None:
        self.__answers: list[str] = answers
        self.__answer_index: int = 0
        self.__conversations: list[list[dict[str, str]]] = []

    async def __aenter__(self) -> "MockChatCompletion":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        pass

    async def create(
        self, conversation: list[dict[str, str]]
    ) -> Tuple[FinishReason, Optional[str]]:
        """
        Sends the given conversation to chat completions inference back-end.
        This is just a dummy implementation. You should create your own
        implementation with access to your specific LLM inference back-end. We
        will provide some default implementations for this class in the future,
        e.g. for the OpenAI API.

        Returns: Tuple of finish reason and LLM response text.
        """
        self.__conversations.append(conversation)
        answer: str = self.__answers[self.__answer_index]
        self.__answer_index += 1
        return FinishReason.STOPPED, answer

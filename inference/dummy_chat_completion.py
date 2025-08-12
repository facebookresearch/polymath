# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import Logger
from types import TracebackType
from typing import Callable, Optional, Tuple

from inference.chat_completion import ChatCompletion, Message
from inference.finish_reason import FinishReason


class DummyChatCompletion(ChatCompletion):
    """
    Dummy implementation of ChatCompletion. You should create your own
    implementation with access to your specific LLM inference back-end. We will
    provide some default implementations for this class in the future, e.g. for
    the OpenAI API.
    """

    def __init__(
        self,
        logger_factory: Callable[[str], Logger],
        model_name: str,
        max_gen_tokens: int,
        max_tokens: int,
        temperature: float,
    ) -> None:
        self.__logger: Logger = logger_factory(__name__)
        self.__model_name = model_name
        self.__max_gen_tokens = max_gen_tokens
        self.__max_tokens = max_tokens
        self.__temperature = temperature

    async def __aenter__(self) -> "DummyChatCompletion":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        pass

    async def create(
        self, conversation: list[Message]
    ) -> Tuple[FinishReason, Optional[str]]:
        return FinishReason.STOPPED, "Hello! I am very intelligent!"

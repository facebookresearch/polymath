# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import auto, StrEnum
from types import TracebackType
from typing import Optional

from inference.finish_reason import FinishReason


class Role(StrEnum):
    """
    Possible author roles of messages in a chat completions dialog.
    """

    # AI or assistant role
    AI = auto()

    # System prompt role
    SYSTEM = auto()

    # User role
    USER = auto()


@dataclass
class Message:
    """
    Represents a single message in a chat completions API.
    """

    # Role of author of message.
    role: Role

    # Content of message.
    text: str


@dataclass
class ChatCompletionResult:
    """
    Result of a single chat completion request.
    """

    # Reason the completion finished.
    finish_reason: FinishReason

    # Response text from the model, if any.
    text: Optional[str] = None

    # Seconds to wait before retrying, if suggested by the API.
    retry_after: float = 0.0


class ChatCompletion(ABC):
    """
    Basic LLM chat completion API, to be implemented by concrete inference
    providers.
    """

    @abstractmethod
    async def __aenter__(self) -> "ChatCompletion": ...

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None: ...

    @abstractmethod
    async def create(
        self, conversation: list[Message]
    ) -> "ChatCompletionResult":
        """
        Sends the given conversation to chat completions inference back-end.
        Each conversation message should contain a "role" and "text" property.

        Args:
            conversation (list[Message]): Messages to send to model for
            completion.
        Returns: Result containing finish reason, response text, and optional
            retry metadata.
        """
        ...

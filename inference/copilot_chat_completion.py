# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import Logger
from os import environ
from random import uniform
from types import TracebackType
from typing import Callable, Optional

from aiohttp import ClientSession

from inference.chat_completion import ChatCompletion, ChatCompletionResult, Message, Role
from inference.finish_reason import FinishReason


# GitHub Models inference endpoint.
_BASE_URL: str = "https://models.inference.ai.azure.com"

# Environment variable name for the GitHub token.
_TOKEN_ENV_VAR: str = "GITHUB_TOKEN"

# Default retry-after duration range in seconds when server doesn't specify.
_DEFAULT_RETRY_AFTER_MIN: float = 30.0
_DEFAULT_RETRY_AFTER_MAX: float = 45.0

# Role mapping from internal Role enum to OpenAI API role strings.
_ROLE_MAP: dict[Role, str] = {
    Role.SYSTEM: "system",
    Role.USER: "user",
    Role.AI: "assistant",
}


class CopilotChatCompletion(ChatCompletion):
    """
    ChatCompletion implementation using the GitHub Models API. Requires a
    GitHub token set in the GITHUB_TOKEN environment variable.
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
        self.__temperature = temperature
        self.__session: Optional[ClientSession] = None

    async def __aenter__(self) -> "CopilotChatCompletion":
        token: Optional[str] = environ.get(_TOKEN_ENV_VAR)
        if not token:
            raise RuntimeError(
                f"{_TOKEN_ENV_VAR} environment variable is required for Copilot API access"
            )
        self.__session = ClientSession(
            base_url=_BASE_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self.__session:
            await self.__session.close()

    async def create(
        self, conversation: list[Message]
    ) -> ChatCompletionResult:
        assert self.__session is not None, "Must be used as async context manager"

        body: dict = {
            "model": self.__model_name,
            "messages": [
                {"role": _ROLE_MAP[m.role], "content": m.text} for m in conversation
            ],
            "temperature": self.__temperature,
        }
        if self.__max_gen_tokens:
            body["max_tokens"] = self.__max_gen_tokens

        async with self.__session.post("/chat/completions", json=body) as response:
            if response.status == 429:
                retry_after = self.__parse_retry_after(
                    response.headers.get("Retry-After")
                )
                self.__logger.warning(
                    f"Rate limited, retry after {retry_after:.1f}s"
                )
                return ChatCompletionResult(
                    FinishReason.RETRYABLE_ERROR, retry_after=retry_after
                )

            if response.status >= 500:
                retry_after = self.__parse_retry_after(
                    response.headers.get("Retry-After")
                )
                self.__logger.warning(
                    f"Server error {response.status}, retry after {retry_after:.1f}s"
                )
                return ChatCompletionResult(
                    FinishReason.RETRYABLE_ERROR, retry_after=retry_after
                )

            if not response.ok:
                body = await response.text()
                self.__logger.error(
                    f"API error {response.status}: {body}"
                )
            response.raise_for_status()
            data: dict = await response.json()

        choice: dict = data["choices"][0]
        finish_reason: str = choice.get("finish_reason", "stop")
        text: Optional[str] = choice["message"].get("content")

        if finish_reason == "length":
            return ChatCompletionResult(FinishReason.MAX_OUTPUT_TOKENS, text)

        return ChatCompletionResult(FinishReason.STOPPED, text)

    @staticmethod
    def __parse_retry_after(header_value: Optional[str]) -> float:
        if header_value:
            try:
                return float(header_value)
            except ValueError:
                pass
        return uniform(_DEFAULT_RETRY_AFTER_MIN, _DEFAULT_RETRY_AFTER_MAX)

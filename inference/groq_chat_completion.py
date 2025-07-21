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

from groq import Groq
import os
import aiohttp
import json

class GroqChatCompletion(ChatCompletion):
    """
    Groq implementation of ChatCompletion. You should create your own
    implementation with access to your specific LLM inference back-end
    """

    def __init__(
        self,
        logger_factory: Callable[[str], Logger],
        model_name: str,
        max_gen_tokens: int,
        temperature: float,
    ) -> None:
        self.__logger: Logger = logger_factory(__name__)
        self.__model_name = model_name
        self.__max_gen_tokens = max_gen_tokens
        self.__temperature = temperature

        self.__api_url = "https://api.groq.com/openai/v1/chat/completions"

    async def __aenter__(self) -> "GroqChatCompletion":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        pass

    def set_chat_model_name(self,model_name:str) -> None:
        self.__model_name = model_name

    async def create(
        self, conversation: list[Message]) -> Tuple[FinishReason, Optional[str]]:

        headers = {
            "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
            "Content-Type": "application/json",
        }

        messages = [
            {"role": msg.role.lower(), "content": msg.text}
            for msg in conversation
        ]

        payload = {
            "model": self.__model_name,
            "messages": messages,
            "temperature": self.__temperature,
            "max_tokens": self.__max_gen_tokens,
        }
        #print(f"URL: {self.__api_url}")
        #print(f"Headers: {headers}")
        #print(f"Payload: {json.dumps(payload, indent=2)}")


        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.__api_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        self.__logger.error(f"Groq API Error: {response.status}")
                        print(await response.json())
                        return FinishReason.RETRYABLE_ERROR, None

                    data = await response.json()

                    # Garde pour debug
                    self.__logger.debug(f"Groq API raw response: {data}")
                    #print(f"Groq API raw response: {data}")

                    finish_reason = data["choices"][0]["finish_reason"]
                    content = data["choices"][0]["message"]["content"]

                    if finish_reason == "length":
                        return FinishReason.MAX_OUTPUT_TOKENS, content
                    return FinishReason.STOPPED, content

            except Exception as e:
                self.__logger.error(f"Exception in Groq API call: {e}")
                return FinishReason.RETRYABLE_ERROR, None

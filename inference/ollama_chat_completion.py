import asyncio
from logging import Logger
from types import TracebackType
from typing import Callable, List, Optional, Tuple

from inference.chat_completion import ChatCompletion, Message
from inference.finish_reason import FinishReason
import ollama

class OllamaChatCompletion(ChatCompletion):
    def __init__(
        self,
        logger_factory: Callable[[str], Logger],
        model_path: str,
        max_gen_tokens: int,
        temperature: float,
        gpu_id = 0,
    ) -> None:
        self.__logger: Logger = logger_factory(__name__)
        self.__model_path = "gemma3:4b"
        self.__max_gen_tokens = max_gen_tokens
        self.__temperature = temperature
        self.__initial_temperature = temperature

    async def __aenter__(self) -> "OllamaChatCompletion":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        pass

    async def create(
        self, conversation: List[Message]
    ) -> Tuple[FinishReason, Optional[str]]:

        conversation= [{"role":msg.role, "content":msg.text} for msg in conversation]
        loop = asyncio.get_running_loop()

        def generate_sync():
            return ollama.chat(model=self.__model_path, messages=conversation)
        try:
            result = await loop.run_in_executor(None, generate_sync)
            print(result['message']['content'])
            return FinishReason.STOPPED, result['message']['content']
        except RuntimeError as e:
            self.__logger.error(f"RuntimeError during generation: {e}")
            return FinishReason.RETRYABLE_ERROR, None
        except Exception as e:
            self.__logger.error(f"Unexpected error during generation: {e}")
            return FinishReason.RETRYABLE_ERROR, None

    def set_temperature(self, temperature: float) -> None:
        self.__temperature = temperature

    def reset_temperature(self) -> None:
        self.__temperature = self.__initial_temperature

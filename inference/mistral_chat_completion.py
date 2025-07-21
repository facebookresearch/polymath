import asyncio
import functools
from logging import Logger
from types import TracebackType
from typing import Callable, List, Optional, Tuple

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
#from mistral_common.protocol.instruct.messages import Message as MistralMessage
from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage, SystemMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

from inference.chat_completion import ChatCompletion, Message
from inference.finish_reason import FinishReason


class MistralChatCompletion(ChatCompletion):
    def __init__(
        self,
        logger_factory: Callable[[str], Logger],
        model_path: str,
        max_gen_tokens: int,
        temperature: float,
    ) -> None:
        self.__logger: Logger = logger_factory(__name__)
        self.__model_path = model_path
        self.__max_gen_tokens = max_gen_tokens
        self.__temperature = temperature

        self.__model = None  # (tokenizer, model)
        self.__lock = asyncio.Lock()

    async def __aenter__(self) -> "MistralChatCompletion":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        # Pas de ressources particulières à libérer ici
        pass

    async def _load_model(self):
        if self.__model is not None:
            return self.__model

        async with self.__lock:
            if self.__model is not None:
                return self.__model

            loop = asyncio.get_running_loop()

            def load():
                tokenizer = MistralTokenizer.from_file(f"{self.__model_path}/tokenizer.model.v3")
                model = Transformer.from_folder(self.__model_path)
                return tokenizer, model

            self.__model = await loop.run_in_executor(None, load)
            self.__logger.info(f"Model loaded from {self.__model_path}")
            return self.__model

    def _convert_conversation(self, conversation: List[Message]) -> ChatCompletionRequest:
        # Convertit la liste inference.chat_completion.Message en mistral_common.protocol.instruct.request.ChatCompletionRequest
        mistral_msgs = []
        for msg in conversation:
                match msg.role:
                        case "user":
                                mistral_msgs.append(UserMessage(content=msg.text))
                        case "assistant":
                                mistral_msgs.append(SystemMessage(content=msg.text))
                        case "system":
                                mistral_msgs.append(AssistantMessage(content=msg.text))

        return ChatCompletionRequest(messages=mistral_msgs)

    async def create(
        self, conversation: List[Message]
    ) -> Tuple[FinishReason, Optional[str]]:
        tokenizer, model = await self._load_model()

        #print("PAYLOAD")
        #for msg in conversation :
        #        print(f"{msg.role}: {msg.text}")
        #print("\n\n")
        request = self._convert_conversation(conversation)

        loop = asyncio.get_running_loop()

        def generate_sync():
            tokens = tokenizer.encode_chat_completion(request).tokens
            out_tokens, _ = generate(
                [tokens],
                model,
                max_tokens=self.__max_gen_tokens,
                temperature=self.__temperature,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )
            result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
            return result

        try:
            result = await loop.run_in_executor(None, generate_sync)
            #print("\nRESULT\n")
            #if result:
            #        print(result)
            # On suppose que la génération s'arrête naturellement ou par token eos
            return FinishReason.STOPPED, result
        except RuntimeError as e:
            self.__logger.error(f"RuntimeError during generation: {e}")
            return FinishReason.RETRYABLE_ERROR, None
        except Exception as e:
            self.__logger.error(f"Unexpected error during generation: {e}")
            return FinishReason.RETRYABLE_ERROR, None


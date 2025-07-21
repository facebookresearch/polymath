import asyncio
import functools
from logging import Logger
from types import TracebackType
from typing import Callable, List, Optional, Tuple, Dict

from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import re
from inference.chat_completion import ChatCompletion, Message
from inference.finish_reason import FinishReason
from concurrent.futures import ThreadPoolExecutor


class LlamaChatCompletion(ChatCompletion):
    def __init__(
        self,
        logger_factory: Callable[[str], Logger],
        model_path: str,
        max_gen_tokens: int,
        temperature: float,
        gpu_id = 0,
    ) -> None:
        self.__logger: Logger = logger_factory(__name__)
        self.__model_path = model_path
        self.__max_gen_tokens = max_gen_tokens
        self.__temperature = temperature
        self.__initial_temperature = temperature
        print(f"model name : {self.__model_path}")
        self.__gpu_id = {"" : f"cuda:{gpu_id}"} if gpu_id != -1 else "auto"
        self.__model = None  # (Chat, tokenizer, model)
        self.__lock = asyncio.Lock()
        self.__executor = ThreadPoolExecutor(max_workers=1)

    async def __aenter__(self) -> "LlamaChatCompletion":
        await self._load_model()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        pass

    async def _load_model(self):
        if self.__model is not None:
            return self.__model

        async with self.__lock:
            if self.__model is not None:
                return self.__model

            loop = asyncio.get_running_loop()

            def load():
                tokenizer = AutoTokenizer.from_pretrained(self.__model_path)
                tokenizer.padding_side = 'left'
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.__model_path,
                    device_map=self.__gpu_id,
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16
                )

                print(model.hf_device_map)
                chat = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer
                )
                return chat, tokenizer, model

            self.__model = await loop.run_in_executor(None, load)
            self.__logger.info(f"Model loaded from {self.__model_path}")

            return self.__model

    def _convert_conversation(self, conversation: List[Message]) -> List[Dict[str, str]]:
        # Convertit la liste inference.chat_completion.Message en mistral_common.protocol.instruct.request.ChatCompletionRequest
        llama_msgs = []
        for msg in conversation:
            llama_msgs.append({"role":msg.role,"content":msg.text})
        return llama_msgs

    def extract_assistant_code(self,text: str) -> str:
        matches = re.findall(r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)(?=<\|start_header_id\|>|<\|eot_id\|>|$)",text,re.DOTALL)
        #print(matches)
        match = matches[len(matches)-1]

        code = match.strip()
        # Détection de l'indentation commune pour nettoyage
        lines = code.splitlines()
        if lines:
            indent = min((len(line) - len(line.lstrip())) for line in lines if line.strip())
            cleaned = "\n".join(line[indent:] if line.strip() else "" for line in lines)
        else:
            cleaned = code
        blocks = cleaned + "\n"
        return blocks

    async def create(
        self, conversation: List[Message]
    ) -> Tuple[FinishReason, Optional[str]]:
        chat, tokenizer, model  = self.__model
        
        conversation= self._convert_conversation(conversation)
        loop = asyncio.get_running_loop()

        def generate_sync():
            # Formatage avec le template chat LLaMA
            prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

            # Génération optimisée
            outputs = chat(
                prompt,
                max_new_tokens=self.__max_gen_tokens,
                #return_full_text=False,
                do_sample=True,
                temperature=self.__temperature,
                top_p = 0.9
            )
            return outputs
        try:
            result = await loop.run_in_executor(self.__executor, generate_sync)
            print(f"\033[91m Result \033[00m : {result[0]["generated_text"]}")
            return FinishReason.STOPPED, self.extract_assistant_code(result[0]["generated_text"])
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


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from json import loads
from logging import Logger
from re import compile, DOTALL, IGNORECASE, Match, MULTILINE, Pattern
from typing import Callable, Optional
import time
from agent.logic.agent import RETRY_COUNT
from agent.logic.solver import Solver
from inference.chat_completion import ChatCompletion, Message, Role
from inference.finish_reason import FinishReason
from judge.result_trace import ResultTrace

# Instructs the model to separate any comments and thinking tokens into a
# dedicated section in order to simplify parsing.
_SYSTEM_PROMPT_FORMAT: str = """You are a helpful logic assistant tasked with solving a Zebra puzzle. It is imperative that you provide the solution in the following JSON format, without any additional comments, text, or highlighting:
```
{}
```

Now answer the user's puzzle in this format!
"""


# Puzzle and expected output format, as explained in _SYSTEM_PROMPT.
_USER_MESSAGE_FORMAT: str = """
Here is the puzzle:
```
{}
```
"""


# Code highlighting pattern using three ticks. This is an alternative way how
# we allow the model to specify its solution.
_JSON_HIGHLIGHTING_PATTERN: str = r"^\s*```[^\n]*$(.*)^```$"


# Llama 4 models may lead with a reasoning phase in markup that we skip.
_REASONING_THINKING_END: str = "<|reasoning_thinking_end|>"


# Extracts the solution JSON text from a response formatted according to
# _SYSTEM_PROMPT.
_SOLUTION_TITLE_PATTERN: Pattern = compile(
    rf"^\s*#\s*solution\s*$((\n{_JSON_HIGHLIGHTING_PATTERN})|(.*))|{_JSON_HIGHLIGHTING_PATTERN}",
    DOTALL | IGNORECASE | MULTILINE,
)


class ModelOnlySolver(Solver):
    """
    Zebra puzzle solver uses a model behind a chat completions API to solve the
    puzzle.
    """

    def __init__(
        self,
        logger_factory: Callable[[str], Logger],
        chat_completion: ChatCompletion,
        result_trace: ResultTrace,
        puzzle: str,
        output_format: str,
    ) -> None:
        self.__logger: Logger = logger_factory(__name__)
        self.__chat_completion: ChatCompletion = chat_completion
        self.__result_trace: ResultTrace = result_trace
        self.__puzzle: str = puzzle
        self.__output_format: str = output_format

    async def solve(self) -> None:
        system_prompt: str = _SYSTEM_PROMPT_FORMAT.format(self.__output_format)
        user_message: str = _USER_MESSAGE_FORMAT.format(self.__puzzle)
        self.__logger.info(f"Sending puzzle to model: {user_message}")
        messages: list[Message] = [
            Message(Role.SYSTEM, system_prompt),
            Message(Role.USER, user_message),
        ]

        assistant_message: str = ""
        for _ in range(RETRY_COUNT):
            start = time.perf_counter()
            finish_reason, response = await self.__chat_completion.create(messages)
            end = time.perf_counter()
            self.__result_trace.format_time = end - start 
            if response is not None:
                assistant_message += response
                messages.append(Message(Role.AI, assistant_message))

            match finish_reason:
                case FinishReason.MAX_OUTPUT_TOKENS:
                    self.__logger.warning(
                        "Encountered token limit, asking model to continue..."
                    )
                    messages.append(
                        Message(
                            Role.USER,
                            "You encountered a token limit, please continue exactly where you left off, without repeating any characters.",
                        )
                    )
                case FinishReason.STOPPED:
                    solution: Optional[str] = ModelOnlySolver.__parse_scorer_response(
                        assistant_message
                    )
                    if solution is not None:
                        try:
                            loads(solution)
                            self.__result_trace.solution = solution
                            self.__result_trace.messages = messages
                            return
                        except ValueError:
                            self.__logger.exception(
                                f"Received invalid JSON: {solution}"
                            )
                    else:
                        self.__logger.error(
                            f"Could not extract solution from model response: {assistant_message}"
                        )

                    self.__logger.info("Resetting messages to retry from scratch.")
                    del messages[2:]
                case FinishReason.RETRYABLE_ERROR:
                    self.__logger.warning("Encountered Chat Completions API error.")

        self.__logger.error("Exceeded retry limited.")

    @staticmethod
    def __parse_scorer_response(response: str) -> Optional[str]:
        """
        Retrieves the JSON solution from the model response.

        Args:
            response (str): Model response
        Returns:
            JSON solution in model response.
        """
        thinking_end_offset: int = response.rfind(_REASONING_THINKING_END)
        if thinking_end_offset != -1:
            thinking_end_offset += len(_REASONING_THINKING_END)
            response = response[thinking_end_offset:]

        title: Match[str] | None = _SOLUTION_TITLE_PATTERN.search(response)
        if title is not None:
            return title.group(5) or title.group(4) or title.group(3)

        try:
            loads(response)
        except ValueError:
            return None
        return response

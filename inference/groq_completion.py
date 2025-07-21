# inference/groq_llama_completion.py
from __future__ import annotations

import asyncio
import os
from types import TracebackType
from typing import Optional, Tuple, List

from dotenv import load_dotenv
from groq import AsyncGroq

from inference.chat_completion import ChatCompletion, Message, Role
from inference.finish_reason import FinishReason


###############################################################################
# ── single-process Groq client cache ──────────────────────────────────────── #
###############################################################################
_CLIENT: Optional[AsyncGroq] = None           # holds the actual client
_CLIENT_LOCK = asyncio.Lock()                 # protects first-time init


async def _get_client() -> AsyncGroq:
    """
    Lazily create (or return) the singleton AsyncGroq client.
    """
    global _CLIENT
    if _CLIENT is None:                       # fast path
        async with _CLIENT_LOCK:              # only one task may enter here
            if _CLIENT is None:               # re-check inside the lock
                load_dotenv()
                _CLIENT = AsyncGroq(
                    api_key=os.getenv("GROQ_API_KEY"),
                )
    return _CLIENT


###############################################################################
# ── ChatCompletion implementation ─────────────────────────────────────────── #
###############################################################################
class LlamaCompletion(ChatCompletion):
    """
    Groq chat-completion backend for the project’s ChatCompletion interface.
    A single Groq client instance is shared across the Python process.
    """

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.2,
        **generation_kwargs,
    ) -> None:
        self._model_name = model_name
        self._temperature = temperature
        self._generation_kwargs = generation_kwargs
        self._client: Optional[AsyncGroq] = None   # bound in __aenter__

    # --------------------------------------------------------------------- #
    # async context-manager helpers                                         #
    # --------------------------------------------------------------------- #
    async def __aenter__(self) -> "LlamaCompletion":
        self._client = await _get_client()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """
        We deliberately do **not** close the shared client here; other
        ChatCompletion instances may still be using it.  If you really need
        to tear it down at process exit, do it once in your application’s
        shutdown code.
        """
        self._client = None  # just release our local reference

    # --------------------------------------------------------------------- #
    # public API required by InferenceClient                                #
    # --------------------------------------------------------------------- #
    async def create(
        self,
        conversation: List[Message],
    ) -> Tuple[FinishReason, Optional[str]]:
        if self._client is None:  # misuse: create() called outside “async with”
            raise RuntimeError("LlamaCompletion used without entering context")

        # 1. marshal conversation into the format Groq expects
        messages = [
            {"role": (msg.role.value if isinstance(msg.role, Role) else msg.role),
             "content": msg.text}
            for msg in conversation
        ]

        # 2. hit Groq
        completion = await self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=self._temperature,
            **self._generation_kwargs,
        )

        # 3. map Groq’s finish_reason → project enum
        choice = completion.choices[0]
        finish_reason = self._map_finish_reason(choice.finish_reason)

        # 4. success → return text; otherwise signal retry/max-tokens
        content = getattr(choice.message, "content", None) if finish_reason is FinishReason.STOPPED else None
        return finish_reason, content

    # --------------------------------------------------------------------- #
    # helpers                                                               #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _map_finish_reason(reason: str | None) -> FinishReason:
        """
        Translate Groq’s finish_reason strings to the project’s FinishReason.
        """
        if reason == "stop":
            return FinishReason.STOPPED
        if reason == "length":
            return FinishReason.MAX_OUTPUT_TOKENS
        # Groq currently uses "error" for transient server errors.
        return FinishReason.RETRYABLE_ERROR
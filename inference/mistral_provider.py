"""inference/hf_chat_completion.py
Hugging Face implementation of the ``ChatCompletion`` interface that is fully
compatible with your **InferenceClient** _and_ the Transformers chat‑template
style returned by models such as *mistralai/Mistral‑7B‑Instruct‑v0.3*.

Key features
------------
* **Process‑wide pipeline cache** → model is loaded once per Python process.
* **Async‑friendly** → the heavyweight generation runs in a thread executor so
  the event‑loop stays responsive.
* **Local‑files‑only & device config** → optional kwargs let you force offline
  mode, 4‑bit loading, etc.
* **Robust output parsing** → works whether the pipeline returns plain text or
  a structured list of messages (the newer chat template format).

Usage:
```
from inference.hf_chat_completion import ChatCompletion

async with ChatCompletion("mistralai/Mistral-7B-Instruct-v0.3",
                         local_files_only=False) as chat:
    finish, reply = await chat.create(conv)
```
"""

from __future__ import annotations

import asyncio
import functools
from typing import List, Optional, Tuple, Callable, Dict, Any
from logging import Logger
from random import randrange

from transformers import pipeline, Pipeline

# Local imports provided by the host code‑base
from inference.chat_completion import ChatCompletion, Message, Role
from inference.finish_reason import FinishReason

# ---------------------------------------------------------------------------
# Process‑wide cache ---------------------------------------------------------
# ---------------------------------------------------------------------------
_PIPE: Pipeline | None = None
_LOCK = asyncio.Lock()

async def _get_pipeline(task: str, model_id: str, **pipe_kwargs) -> Pipeline:
    """Load the HF pipeline once (thread‑safe, async‑aware)."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    async with _LOCK:  # first waiter loads the model
        if _PIPE is not None:  # another waiter might have won the race
            return _PIPE

        loop = asyncio.get_running_loop()
        _PIPE = await loop.run_in_executor(  # heavy load off the event loop
            None,
            functools.partial(pipeline, task=task, model=model_id, **pipe_kwargs),
        )
        return _PIPE

# ---------------------------------------------------------------------------
# Main class ----------------------------------------------------------------
# ---------------------------------------------------------------------------
class MistralCompletion(ChatCompletion):
    """HuggingFace backend for the project‑specific ``ChatCompletion`` proto."""

    def __init__(
        self,
        model_id: str,
        *,
        local_files_only: bool = True,
        **pipe_kwargs: Any,
    ) -> None:
        self._model_id = model_id
        self._local_files_only = local_files_only
        self._pipe_kwargs = pipe_kwargs

    # ~~~~~~~~~~~~~~~~~~~~~ context‑manager hooks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    async def __aenter__(self) -> "ChatCompletion":
        await _get_pipeline(
            task="text-generation",
            model_id=self._model_id,
            local_files_only=self._local_files_only,
            **self._pipe_kwargs,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):  # noqa: D401 (not a contextlib CM)
        # We deliberately *do not* unload the model so it remains cached.
        return False  # propagate exceptions

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ main entry point ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    async def create(
        self, conversation: List[Message]
    ) -> Tuple[FinishReason, Optional[str]]:
        """Run inference and return (finish_reason, text)."""
        pipe = await _get_pipeline("text-generation", self._model_id)

        # The HF chat template expects a list[dict[{role,content}]]
        prompt: List[Dict[str, str]] = self._format_conversation(conversation)

        loop = asyncio.get_running_loop()
        try:
            out = await loop.run_in_executor(
                None,
                functools.partial(
                    pipe,
                    prompt,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=1,
                    return_full_text=False,
                ),
            )
        except RuntimeError as err:  # OOM etc. – treat as retryable
            return FinishReason.RETRYABLE_ERROR, None

        # HF pipeline returns a list with one item per sequence
        generated = out[0]["generated_text"]

        # Handle two possible shapes:
        # 1. Plain string (older default)
        if isinstance(generated, str):
            return FinishReason.STOPPED, generated.strip()

        # 2. List[{role,content}] – new chat‑template format
        if isinstance(generated, list):
            for msg in reversed(generated):
                if msg.get("role") == "assistant":
                    return FinishReason.STOPPED, msg.get("content", "").strip()
            # Fallback – no assistant message found
            return FinishReason.STOPPED, generated[-1].get("content", "").strip()

        # Unexpected shape ⇒ treat as token limit (forces InferenceClient retry)
        return FinishReason.MAX_OUTPUT_TOKENS, None

    # ---------------------------------------------------------------------
    @staticmethod
    def _format_conversation(conversation: List[Message]) -> List[Dict[str, str]]:
        """Map project Message objects → HF chat‑template dictionaries."""
        formatted: List[Dict[str, str]] = []
        for msg in conversation:
            role = (
                msg.role.value
                if hasattr(msg.role, "value") else
                msg.role.name.lower() if hasattr(msg.role, "name") else
                str(msg.role).lower()
            )
            formatted.append({"role": role, "content": msg.text})

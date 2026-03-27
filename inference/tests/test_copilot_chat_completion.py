# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import Logger, getLogger
from os import environ
from unittest import IsolatedAsyncioTestCase, skipUnless
from unittest.mock import AsyncMock, MagicMock, patch

from dotenv import load_dotenv

from inference.chat_completion import ChatCompletionResult, Message, Role
from inference.copilot_chat_completion import CopilotChatCompletion
from inference.finish_reason import FinishReason


load_dotenv()


def _silent_logger(name: str) -> Logger:
    logger = getLogger(name)
    logger.disabled = True
    return logger


class TestCopilotChatCompletion(IsolatedAsyncioTestCase):

    def _make_completion(self) -> CopilotChatCompletion:
        return CopilotChatCompletion(
            logger_factory=_silent_logger,
            model_name="gpt-4o",
            max_gen_tokens=1024,
            max_tokens=4096,
            temperature=0.5,
        )

    async def test_create_returns_stopped(self) -> None:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {"content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
            cc = self._make_completion()
            async with cc:
                with patch.object(
                    cc._CopilotChatCompletion__session,
                    "post",
                    return_value=mock_response,
                ):
                    result: ChatCompletionResult = await cc.create(
                        [Message(Role.USER, "Hi")]
                    )

        self.assertEqual(FinishReason.STOPPED, result.finish_reason)
        self.assertEqual("Hello!", result.text)

    async def test_create_returns_max_output_tokens_on_length(self) -> None:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {"content": "partial..."},
                        "finish_reason": "length",
                    }
                ]
            }
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
            cc = self._make_completion()
            async with cc:
                with patch.object(
                    cc._CopilotChatCompletion__session,
                    "post",
                    return_value=mock_response,
                ):
                    result: ChatCompletionResult = await cc.create(
                        [Message(Role.USER, "Hi")]
                    )

        self.assertEqual(FinishReason.MAX_OUTPUT_TOKENS, result.finish_reason)
        self.assertEqual("partial...", result.text)

    async def test_rate_limit_returns_retryable_error(self) -> None:
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.headers = {"Retry-After": "10"}
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
            cc = self._make_completion()
            async with cc:
                with patch.object(
                    cc._CopilotChatCompletion__session,
                    "post",
                    return_value=mock_response,
                ):
                    result: ChatCompletionResult = await cc.create(
                        [Message(Role.USER, "Hi")]
                    )

        self.assertEqual(FinishReason.RETRYABLE_ERROR, result.finish_reason)
        self.assertEqual(10.0, result.retry_after)

    async def test_missing_token_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            cc = self._make_completion()
            with self.assertRaises(RuntimeError):
                async with cc:
                    pass

    @skipUnless(environ.get("GITHUB_TOKEN"), "GITHUB_TOKEN not set")
    async def test_integration(self) -> None:
        cc = self._make_completion()
        async with cc:
            result: ChatCompletionResult = await cc.create(
                [Message(Role.USER, "Reply with exactly: Hello World")]
            )
        self.assertEqual(FinishReason.STOPPED, result.finish_reason)
        self.assertIsNotNone(result.text)
        self.assertIn("Hello World", result.text)

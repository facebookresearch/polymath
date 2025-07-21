# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger
from unittest import IsolatedAsyncioTestCase

from dotenv import load_dotenv
from inference.chat_completion import Message, Role
from inference.client import InferenceClient
from inference.tests.mock_chat_completion import MockChatCompletion


load_dotenv()


class TestInferenceClient(IsolatedAsyncioTestCase):

    async def test_single(self) -> None:
        async with MockChatCompletion(["Hello, I am an AI."]) as chat_completion:
            client = InferenceClient(getLogger, chat_completion)
            client.add_message("Say hello, please!", Role.USER)
            self.assertEqual("Hello, I am an AI.", await client.send())
            self.assertEqual(
                [
                    Message(Role.USER, "Say hello, please!"),
                    Message(Role.AI, "Hello, I am an AI."),
                ],
                client.conversation,
            )

    async def test_conversation(self) -> None:
        async with MockChatCompletion(["Hi Mate, I am an AI."]) as chat_completion:
            client = InferenceClient(getLogger, chat_completion)
            client.add_message("Address the user as `mate`.", Role.SYSTEM)
            client.add_message("Hi! How are you?", Role.USER)
            self.assertEqual("Hi Mate, I am an AI.", await client.send())
            self.assertEqual(
                [
                    Message(Role.SYSTEM, "Address the user as `mate`."),
                    Message(Role.USER, "Hi! How are you?"),
                    Message(Role.AI, "Hi Mate, I am an AI."),
                ],
                client.conversation,
            )

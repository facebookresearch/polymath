# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from asyncio import create_task, FIRST_COMPLETED, gather, Task, wait
from typing import Callable, Coroutine


class AsyncPool:
    """
    Helper to invoke `async` tasks concurrently, with a configurable maximum
    concurrency level.
    """

    def __init__(self, max_concurrency: int):
        """
        Args:
            max_concurrency (int): Maximum number of concurrent tasks to
            execute.
        """
        self.__max_concurrency = max_concurrency
        self.__tasks: set[Task] = set()

    async def submit(self, task: Callable[[], Coroutine]):
        """
        Accepts a factory for an async task which is executed in a pool with
        configured maximum concurrency. This method is not concurrency safe, it
        is expected that it is invoked from one central management task/thread.
        """
        while len(self.__tasks) >= self.__max_concurrency:
            _, self.__tasks = await wait(self.__tasks, return_when=FIRST_COMPLETED)
        self.__tasks.add(create_task(task()))

    async def gather(self):
        """
        Wait until all tasks scheduled via `submit` are completed. No further
        invocations of `submit` are allowed once this method is invoked.
        """
        if self.__tasks:
            await gather(*self.__tasks)

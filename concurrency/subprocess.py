# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from asyncio import create_subprocess_exec, wait_for
from asyncio.subprocess import PIPE, Process
from typing import Optional, Tuple

from encoding.encoding import Encoding


class Subprocess:
    """
    Helper to invoke a subprocess and retrieve its exit code and stdout/stderr.
    """

    @staticmethod
    async def run(
        program: str, *args: str, timeout_in_s: Optional[float | int] = None
    ) -> Tuple[int, str, str]:
        """
        Invokes the given program with the provided arguments, optionally
        accepting a maximum timeout to wait for until the process completes.

        Args:
            program: Executable to run.
            args: Arguments to pass to execuable.
            timeout_in_s: Timeout in secods to wait for process to complete.
        Returns:
            Process exit code as well as its stdout and stderr as strings.
        """
        process: Process = await create_subprocess_exec(
            program,
            *args,
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout_bytes: bytes
        stderr_bytes: bytes
        stdout_bytes, stderr_bytes = await wait_for(process.communicate(), timeout_in_s)

        exit_code: int = process.returncode or 0
        stdout: str = Encoding.decode_process_output(stdout_bytes)
        stderr: str = Encoding.decode_process_output(stderr_bytes)
        return exit_code, stdout, stderr

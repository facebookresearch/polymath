# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from asyncio import create_subprocess_exec
from asyncio.subprocess import PIPE, Process
from types import TracebackType
from typing import Optional

import aiofiles

from aiofiles.tempfile import TemporaryDirectory

from encoding.encoding import Encoding
from libcst import MetadataWrapper, Module, parse_module
from libcst.metadata.full_repo_manager import FullRepoManager
from libcst.metadata.type_inference_provider import TypeInferenceProvider


_MODULE_NAME: str = "module.py"


class PyreServer:

    def __init__(self, project_directory: str) -> None:
        self.__project_directory: str = project_directory

    async def __aenter__(self) -> "PyreServer":
        await self.__run_pyre_command("init", b"\n\n")
        await self.__run_pyre_command("start")
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.__run_pyre_command("stop")

    async def __run_pyre_command(self, command: str, stdin: Optional[bytes] = None):
        process: Process = await create_subprocess_exec(
            "pyre",
            command,
            stdin=PIPE if stdin else None,
            stdout=PIPE,
            stderr=PIPE,
            cwd=self.__project_directory,
        )
        stdout_bytes, stderr_bytes = await process.communicate(stdin)
        stdout: str = Encoding.decode_process_output(stdout_bytes)
        stderr: str = Encoding.decode_process_output(stderr_bytes)
        if process.returncode != 0:
            raise RuntimeError(
                f"""`pyre {command}` failed
{stdout}
{stderr}
"""
            )


class ModuleWithTypeInfoFactory:
    """
    Helper to create libCST modules with pyre type information.
    """

    @staticmethod
    async def create_module(code: str) -> MetadataWrapper:
        """
        Creates a libCST module with pyre type information. We replace the
        `inclusive_or` operator, which `pyre` cannot parse, by an `or` operator,
        then copy over the metadata information into a mdoule that we manually
        parse.

        Args:
            code (str): Module code to parse.
        Returns:
            libCST module with type information.
        """
        async with TemporaryDirectory() as project_directory:
            module_file_name: str = os.path.join(project_directory, _MODULE_NAME)
            async with aiofiles.open(module_file_name, mode="w") as file:
                await file.write(code.replace(" inclusive_or ", " or        "))

            async with PyreServer(project_directory):
                full_repo_manager = FullRepoManager(
                    project_directory,
                    [_MODULE_NAME],
                    [TypeInferenceProvider],
                )
                module: Module = parse_module(code)
                return MetadataWrapper(
                    module, True, full_repo_manager.get_cache_for_path(_MODULE_NAME)
                )

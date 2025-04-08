# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from asyncio import run
from json import dumps
from logging import getLogger
from typing import Any, Optional

from agent.logic.logic_py_c_harness_generator import LogicPyCHarnessGenerator

from aiofiles import open
from dotenv import load_dotenv
from inference.chat_completion import Message
from libcst import Module, parse_module
from training.cbmc_scorer_vc_factory import CBMCScorerVerificationConstraintsFactory
from training.logic_rl_training_dialog import LogicRlTrainingDialog
from training.sample_output_converter import SampleOutputConverter
from training.sample_output_converter_factory import create_sample_output_converter
from training.scorer_vc_sample_parser import ScorerVCSampleParser
from training.scorer_vc_sample_parser_factory import create_scorer_vc_sample_parser


async def main() -> None:
    load_dotenv()
    constraint_factory = CBMCScorerVerificationConstraintsFactory(getLogger)
    parser: ScorerVCSampleParser = create_scorer_vc_sample_parser()
    sample_output_converter: SampleOutputConverter = create_sample_output_converter()
    input_file_path: str = (
        "/mnt/wsfuse/mpchen/atlas/puzzles/zebra_puzzles/zebra_puzzles_tudor_train.jsonl"
    )
    output_file_path: str = (
        "/mnt/wsfuse/users/pkesseli/reasoning/zebra_puzzles_tudor_train.jsonl"
    )
    async with open(output_file_path, "w") as output_file:
        async with open(input_file_path, "r") as input_file:
            async for line in input_file:
                puzzle, data_structure, constraints = parser.parse(line)
                python_code: str = f"""{data_structure}
    
{constraints}
"""
                module: Module = parse_module(python_code)
                c_code: str = LogicPyCHarnessGenerator.generate(module)

                scorer_vc: Optional[str] = await constraint_factory.convert(c_code)
                if scorer_vc is None:
                    raise ValueError(f"Failed to convert sample: {line}")
                dialog: list[Message] = LogicRlTrainingDialog.create(
                    puzzle, data_structure
                )
                metadata: dict[str, Any] = {"scorer_vc": scorer_vc}
                sample: Any = sample_output_converter.convert(dialog, metadata)
                output_line: str = f"{dumps(sample)}\n"
                await output_file.write(output_line)
                break


if __name__ == "__main__":
    run(main())

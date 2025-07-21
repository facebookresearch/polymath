# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC


class Solver(ABC):
    """
    Implementations solve Zebra puzzles using various strategies, and write the
    outcome to result traces.
    """

    async def solve(self) -> None:
        """
        Solves the configured benchmark task, in this context always a Zebra
        puzzle.
        """
        ...

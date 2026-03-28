# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from agent.logic.prolog_engine_strategy import PrologEngineStrategy
from agent.logic.cbmc_search_engine_strategy import CBMCSearchEngineStrategy
from agent.logic.z3_conclusion_check_engine_strategy import Z3ConclusionCheckEngineStrategy
from agent.logic.engine_strategy import EngineStrategy
from logging import Logger
from typing import Callable

class EngineStrategyFactory(ABC):
    """
    Abstract factory interface for creating EngineStrategy instances.
    """
    @abstractmethod
    def create(self,logger_factory: Callable[[str], Logger], puzzle: str, output_format: str) -> EngineStrategy:
        """
        Creates an instance of an EngineStrategy suitable for solving the given puzzle.

        Args:
            logger_factory (Callable[[str], Logger]): A callable that produces loggers for the given module name.
            puzzle (str): A textual representation of the logic puzzle to solve.
            output_format (str): The expected output format (JSON).

        Returns:
            EngineStrategy: A concrete instance implementing the logic for solving the specified puzzle.
        """
        ...

class PrologStrategyFactory(EngineStrategyFactory):
    def create(self,logger_factory, puzzle: str, output_format: str) -> EngineStrategy:
        return PrologEngineStrategy(logger_factory, puzzle, output_format)

class CbmcStrategyFactory(EngineStrategyFactory):
    def create(self,logger_factory, puzzle: str, output_format: str) -> EngineStrategy:
        return CBMCSearchEngineStrategy(logger_factory, puzzle, output_format)

class Z3StrategyFactory(EngineStrategyFactory):
    def create(self,logger_factory, puzzle: str, output_format: str) -> EngineStrategy:
        return Z3ConclusionCheckEngineStrategy(logger_factory, puzzle, output_format)

from abc import ABC, abstractmethod
from agent.logic.prolog_engine_strategy import PrologEngineStrategy
from agent.logic.cbmc_search_engine_strategy import CBMCSearchEngineStrategy
from agent.logic.engine_strategy import EngineStrategy
from logging import logger

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
            output_format (str): The expected output format (e.g., table, list, JSON).

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

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

class SolutionComparator(ABC):
    @abstractmethod
    def compare(
        self,
        solution: Dict[str, Any],
        outcome: Optional[str],
        display: bool = False,
        task_id: Optional[str] = None
    ) -> Tuple[bool, int, str]:
        """
        Abstract method for comparing a target solution with the model's output.

        Parameters:
        - solution (Dict[str, Any]): The ground truth or expected solution.
        - outcome (Optional[str]): The model-generated output, typically a JSON string.
        - display (bool): If True, prints detailed comparison messages to the console.
        - task_id (Optional[str]): Optional identifier used for logging context.

        Returns:
        - Tuple[bool, int, str]:
            - bool: True if the outcome matches the solution, False otherwise.
            - int: Number of mismatches (0 if fully correct).
            - str: Text summary of the result (success or explanation of mismatches).
        """

# agent/logic/zebra_comparator.py

from typing import Dict, Any, Optional, Tuple
import os
from json import loads
from agent.logic.solution_comparator import SolutionComparator

class ZebraSolutionComparator(SolutionComparator):
    """
        Compares the expected Zebra puzzle solution with the LLM-generated outcome.

        The method:
        - Normalizes both expected and actual values (case, spaces, underscores).
        - Iterates over each house and attribute to verify correctness.
        - Supports logging mismatches with context-specific error messages.
    """
    def compare_zebra_solutions(
        self,
        solution: Dict[str, Any],
        outcome: Optional[str],
        display: bool = False,
        task_id: Optional[str] = None
    ) -> Tuple[bool, int, str]:
        if not outcome:
            return False, 0, "Error: outcome is empty or None"

        rows = solution["rows"]
        header = solution["header"]

        expected = {}
        for idx, row in enumerate(rows):
            house_id = f"House {idx+1}"
            expected[house_id] = {header[i]: row[i] for i in range(1, len(header))}

        try:
            outcome = loads(outcome)
        except Exception:
            return False, 0, "Error: outcome is not valid JSON"

        errors = []
        for house_id, expected_attrs in expected.items():
            given_attrs = outcome["solution"].get(house_id, {})
            for key, expected_value in expected_attrs.items():
                given_value = given_attrs.get(key)
                ev = expected_value.replace(" ", "").replace("_", "").lower()
                gv = str(given_value).replace(" ", "").replace("_", "").lower()

                if ev != gv:
                    errors.append(
                        f"❌ [{task_id or ''}] Mismatch in {house_id}, field '{key}': "
                        f"expected '{expected_value}', got '{given_value}'"
                    )

        success = len(errors) == 0
        if display:
            if success:
                print(f"✅ [{task_id or ''}] Solutions match.")
            else:
                print(os.linesep.join(errors))

        return success, len(errors), (
            "✅ Solutions match." if success else os.linesep.join(errors)
        )

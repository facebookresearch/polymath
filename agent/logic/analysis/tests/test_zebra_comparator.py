# tests/test_zebra_comparator.py

import unittest
from agent.logic.zebra_comparator import ZebraSolutionComparator
import json

class TestZebraSolutionComparator(unittest.TestCase):
    def setUp(self):
        self.comparator = ZebraSolutionComparator()
        self.solution = {
            "header": ["House", "Color", "Nationality"],
            "rows": [
                ["1", "Red", "English"],
                ["2", "Green", "Spanish"],
                ["3", "Blue", "Japanese"]
            ]
        }

    def test_matching_solution(self):
        outcome = {
            "solution": {
                "House 1": {"Color": "Red", "Nationality": "English"},
                "House 2": {"Color": "Green", "Nationality": "Spanish"},
                "House 3": {"Color": "Blue", "Nationality": "Japanese"},
            }
        }
        success, err_count, msg = self.comparator.compare_zebra_solutions(self.solution, json.dumps(outcome))
        self.assertTrue(success)
        self.assertEqual(err_count, 0)
        self.assertIn("Solutions match", msg)

    def test_mismatched_solution(self):
        outcome = {
            "solution": {
                "House 1": {"Color": "Red", "Nationality": "French"},
                "House 2": {"Color": "Green", "Nationality": "Spanish"},
                "House 3": {"Color": "Blue", "Nationality": "Japanese"},
            }
        }
        success, err_count, msg = self.comparator.compare_zebra_solutions(self.solution, json.dumps(outcome))
        self.assertFalse(success)
        self.assertEqual(err_count, 1)
        self.assertIn("Mismatch in House 1", msg)

    def test_invalid_json(self):
        success, err_count, msg = self.comparator.compare_zebra_solutions(self.solution, "not a json")
        self.assertFalse(success)
        self.assertEqual(err_count, 0)
        self.assertIn("not valid JSON", msg)

    def test_none_outcome(self):
        success, err_count, msg = self.comparator.compare_zebra_solutions(self.solution, None)
        self.assertFalse(success)
        self.assertEqual(err_count, 0)
        self.assertIn("outcome is empty or None", msg)

if __name__ == '__main__':
    unittest.main()

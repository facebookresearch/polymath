from typing import Dict, List, Any
from collections import defaultdict
import math
from utils import compare_zebra_solutions, difficulty, find_solution
from solution_comparator import SolutionComparator


def analyze_all_categories(comparator: SolutionComparator,
                           data: List[Dict[str, Any]],
                           expected_data: List[Dict[str, Any]]
                           ) -> Dict[str, Any]:
    categories = {"Small": [], "Medium": [], "Large": [], "X-Large": []}
    for d in data:
        level = difficulty(d["size"])
        categories[level].append(d)

    results = {}
    temps_total_global = 0.0
    success_total = 0
    failure_total = 0
    succes_per_cell_total: float = 0
    n_data = len(expected_data)

    for level, items in categories.items():
        time_totals = defaultdict(float)
        time_counts = defaultdict(int)
        success_count = 0
        failure_count = 0
        success_per_cell_count: float = 0

        for d in items:
            op = d["size"][1]
            n, m = map(int, d["size"].split(op))
            error_nb = 0

            if d.get("success"):
                success_count += 1
            elif d.get("success") is None:
                if d.get("output")[0] or not d.get("chat_history"):
                    failure_count += 1
                else:
                    id = d.get("session_id")
                    solution = find_solution(id, expected_data)
                    real_success, error_nb, _ = comparator.compare(solution, d.get("chat_history")[-1])
                    if real_success:
                        success_count += 1
                    else:
                        print(f"Error : {id}")
                        failure_count += 1
                        error_nb = n * m
            else:
                id = d.get("session_id")
                solution = find_solution(id, expected_data)
                real_success, error_nb, _ = comparator.compare(solution, d.get("output")[0])
                if real_success:
                    success_count += 1
                else:
                    print(f"False : {id}")
                    failure_count += 1

            for key, val in d["time"].items():
                time_totals[key] += val
                time_counts[key] += 1
                temps_total_global += val

            success_per_cell_count += ((n * m) - error_nb) / (n * m)

        averages = {k: (time_totals[k] / time_counts[k]) if time_counts[k] else 0 for k in time_totals}

        results[level] = {
            "average_times": averages,
            "total": sum(averages.values()) / 60,
            "successes": success_count,
            "failures": failure_count,
            "success_per_cell": success_per_cell_count,
        }

        success_total += success_count
        failure_total += failure_count
        succes_per_cell_total += success_per_cell_count

    return {
        "categories": results,
        "total_time_across_all_puzzles": temps_total_global / 3600,
        "total_succes": success_total,
        "total_failure": failure_total,
        "Succes_per_cell": (succes_per_cell_total / n_data) * 100,
        "Result": (success_total / n_data) * 100,
    }

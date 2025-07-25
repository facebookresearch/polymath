from typing import Dict, Tuple, Optional, List, Any
import math
import json
from json import loads
import os

def difficulty(size: str) -> str:
    op = size[1]
    n, m = map(int, size.split(op))
    res = math.factorial(n) ** m
    if res < 10**3:
        return "Small"
    if res < 10**6:
        return "Medium"
    if res < 10**10:
        return "Large"
    return "X-Large"

def find_solution(session_id: str, expected_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return next((d["solution"] for d in expected_data if d["id"] == session_id), None)

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

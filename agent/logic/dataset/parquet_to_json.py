import sys
import json
import pandas as pd
from json import loads
from typing import Any

def parquet_to_json(parquet_file: str, output_file: str) -> None:
    """Convert a Parquet file to a JSON file."""
    df = pd.read_parquet(parquet_file)
    raw_dict = loads(df.to_json())

    keys = list(raw_dict.keys())
    result: list[dict[str, Any]] = []

    for row_id in raw_dict[keys[0]].keys():
        row = {k: raw_dict[k][row_id] for k in keys}
        result.append(row)

    with open(output_file, "w") as outfile:
        json.dump(result, outfile, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 parquet_to_json.py file_name.parquet")
        sys.exit(1)

    parquet_path = sys.argv[1]
    json_output = parquet_path.replace(".parquet", ".json")

    parquet_to_json(parquet_path, json_output)
    print(f"JSON file written to: {json_output}")

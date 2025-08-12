import json
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Any

def json_to_parquet(input_file: str, output_file: str) -> None:
    """Convert a JSON file to a Parquet file."""
    with open(input_file, "r", encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 json_to_parquet.py input_file.json")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = input_path.replace(".json", ".parquet")

    json_to_parquet(input_path, output_path)
    print(f"Parquet file written to: {output_path}")

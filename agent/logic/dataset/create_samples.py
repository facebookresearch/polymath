import json
import random
import argparse
import os
from typing import Any

def create_samples(count: int, output_file: str) -> None:
    """
    Create a sample dataset by extracting a fixed number of entries from multiple files.
    """
    prefix = "test-00000-of-00001-"
    files = {
        "Small":  prefix + "small.json",
        "Medium": prefix + "medium.json",
        "Large":  prefix + "large.json",
        "X-Large":prefix + "xlarge.json"
    }

    final_samples: list[dict[str, Any]] = []

    for level, filename in files.items():
        if not os.path.exists(filename):
            print(f"File {filename} not found, skipping.")
            continue

        with open(filename, 'r') as f:
            data = json.load(f)

        nb_available = len(data)
        nb_to_sample = min(count, nb_available)
        sampled = random.sample(data, nb_to_sample)
        final_samples.extend(sampled)

        print(f"{level}: requested {count}, found {nb_available}, used {nb_to_sample}")

    with open(output_file, 'w') as f:
        json.dump(final_samples, f, indent=2)

    print(f"Combined sample written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30, help="Number of samples per category")
    parser.add_argument("--output", type=str, default="samples.json", help="Output file name")
    args = parser.parse_args()

    create_samples(args.count, args.output)

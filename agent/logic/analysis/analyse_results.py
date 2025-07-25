import sys
from analysis import analyze_all_categories
from utils import load_json
import json
from zebra_comparator import ZebraSolutionComparator
if len(sys.argv) != 4:
    print("Usage : python main.py data.json input_file.json output_file.json")
    sys.exit(1)

data_file = sys.argv[1]
input_file = sys.argv[2]
output_file = sys.argv[3]

expected_data = load_json(data_file)
data = load_json(input_file)

comparator = ZebraSolutionComparator()
final_output = analyze_all_categories(comparator, data, expected_data)

print(json.dumps(final_output, indent=2))

with open(output_file, "w") as f:
    json.dump(final_output, f, indent=2)

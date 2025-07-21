import json
import random
import argparse
import os

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--count", type=int, default=30)
parser.add_argument("--output", type=str, default="samples.json")
args = parser.parse_args()

count = args.count
output_file = args.output

# Fichiers d'entrée
prefix = "test-00000-of-00001-"
files = {
    "Small":  prefix + "small.json",
    "Medium": prefix + "medium.json",
    "Large":  prefix + "large.json",
    "X-Large":prefix + "xlarge.json"
}

final_samples = []

for level, filename in files.items():
    if not os.path.exists(filename):
        print(f"Fichier {filename} non trouvé, ignoré.")
        continue

    with open(filename, 'r') as f:
        data = json.load(f)

    nb_available = len(data)
    nb_to_sample = min(count, nb_available)
    sampled = random.sample(data, nb_to_sample)
    final_samples.extend(sampled)

    print(f"{level} : demandé {count}, trouvé {nb_available}, utilisé {nb_to_sample}")

# Sauvegarde du fichier final
with open(output_file, 'w') as f:
    json.dump(final_samples, f, indent=2)

print(f"Fichier combiné écrit dans {output_file}")

import json
import math
from collections import defaultdict
import sys

if len(sys.argv) != 3:
    print("Usage : python time_analysis.py input_file.json output_file.json")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# Fonction de difficulté
def difficulty(size: str) -> str:
    n, m = map(int, size.split('*'))
    res = math.factorial(n) ** m
    if res < 10**3: return "Small"
    if res < 10**6: return "Medium"
    if res < 10**10: return "Large"
    return "X-Large"

# Charger les données depuis un fichier
with open(input_file, "r") as f:
    data = json.load(f)

# Initialiser les catégories
categories = {
    "Small": [],
    "Medium": [],
    "Large": [],
    "X-Large": []
}

# Classer les puzzles par difficulté
for d in data:
    level = difficulty(d["size"])
    categories[level].append(d)

# Calculs
results = {}
temps_total_global = 0.0  # Somme de tous les temps des puzzles

for level, items in categories.items():
    time_totals = defaultdict(float)
    time_counts = defaultdict(int)
    success_count = 0
    failure_count = 0

    for d in items:
        # Succès ou échec
        if d.get("success"):
            success_count += 1
        else:
            failure_count += 1

        # Accumuler les temps
        for key, val in d["time"].items():
            time_totals[key] += val
            time_counts[key] += 1
            temps_total_global += val  # Ajout au temps total global

    # Moyennes par champ de temps
    averages = {k: (time_totals[k] / time_counts[k]) if time_counts[k] else 0 for k in time_totals}

    results[level] = {
        "average_times": averages,
        "successes": success_count,
        "failures": failure_count,
    }

# Résumé final
final_output = {
    "categories": results,
    "total_time_across_all_puzzles": temps_total_global
}

# Affichage
print(json.dumps(final_output, indent=2))

# Écriture dans un fichier
with open(output_file, "w") as f:
    json.dump(final_output, f, indent=2)

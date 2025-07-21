import json
from json import loads
import math
from collections import defaultdict
import sys

if len(sys.argv) != 4:
    print("Usage : python time_analysis.py data.json input_file.json output_file.json")
    sys.exit(1)

data_file = sys.argv[1]
input_file = sys.argv[2]
output_file = sys.argv[3]

# Fonction de difficulté
def difficulty(size: str) -> str:
    op = size[1]
    n, m = map(int, size.split(op))
    res = math.factorial(n) ** m
    if res < 10**3: return "Small"
    if res < 10**6: return "Medium"
    if res < 10**10: return "Large"
    return "X-Large"

def compare_solutions(solution, outcome):
    
    if not outcome:
        print("Erreur : outcome est vide ou None")
        return None, None

    rows = solution["rows"]
    header = solution["header"]

    # Convertir solution en dict comparable
    expected = {}
    for idx, row in enumerate(rows):
        house_id = f"House {idx+1}"
        expected[house_id] = {header[i]: row[i] for i in range(1, len(header))}
    try :
        outcome = loads(outcome)
    except:
        return False,0
    error_nb = 0
    for house_id, expected_attrs in expected.items():
        given_attrs = outcome["solution"][house_id]
        for key, expected_value in expected_attrs.items():
            given_value = given_attrs.get(key)
            if expected_value.replace(" ", "").replace("_", "").lower() != given_value.replace(" ","").replace("_","").lower():
                error_nb +=1

    return error_nb == 0, error_nb

with open(data_file, "r") as f:
    expected_data = json.load(f)

def find_sol(id , data):
    for dict in data :
        if dict["id"] == id:
            return dict["solution"]

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

n_data = len(expected_data)    

# Calculs
results = {}
temps_total_global = 0.0  # Somme de tous les temps des puzzles
success_total = 0
failure_total = 0
succes_per_cell_total : float= 0

for level, items in categories.items():
    time_totals = defaultdict(float)
    time_counts = defaultdict(int)
    success_count = 0
    failure_count = 0
    success_per_cell_count: float = 0

    for d in items:
        op = d["size"][1]
        n, m = map(int, d["size"].split(op))

        # Succès ou échec
        if d.get("success"):
            success_count += 1
            error_nb = 0
        elif d.get("success") is None:
            if d.get("output")[0] :failure_count += 1
            elif not d.get("chat_history"):
                failure_count += 1
            else:
                id = d.get("session_id")
                print(f"{id=}")
                solution = find_sol(id, expected_data)
                real_succes, error_nb = compare_solutions(solution, d.get("chat_history")[-1])
                if real_succes : success_count +=1
                else :
                    print(f"Error : {d.get('session_id')}")
                    failure_count +=1
                    error_nb = n*m
        else:
            id = d.get("session_id")
            print(f"{id=}")
            solution = find_sol(id, expected_data)
            real_succes, error_nb = compare_solutions(solution, d.get("output")[0])
            if real_succes : success_count += 1
            else :
                print(f"False : {d.get('session_id')}")
                failure_count += 1
                    

        # Accumuler les temps
        for key, val in d["time"].items():
            time_totals[key] += val
            time_counts[key] += 1
            temps_total_global += val  # Ajout au temps total global

        success_per_cell_count += ((n*m) - error_nb) / (n*m)

    # Moyennes par champ de temps
    averages = {k: (time_totals[k] / time_counts[k]) if time_counts[k] else 0 for k in time_totals}

    results[level] = {
        "average_times": averages,
        "total": sum(averages.values())/60,
        "successes": success_count,
        "failures": failure_count,
        "success_per_cell" : success_per_cell_count,
    }
    success_total += success_count
    failure_total += failure_count
    succes_per_cell_total +=success_per_cell_count

# Résumé final
final_output = {
    "categories": results,
    "total_time_across_all_puzzles": temps_total_global / 3600,
    "total_succes": success_total,
    "total_failure": failure_total,
    "Succes_per_cell" : ( succes_per_cell_total / n_data ) * 100, 
    "Result" : (success_total / n_data ) * 100,
    }

# Affichage
print(json.dumps(final_output, indent=2))

# Écriture dans un fichier
with open(output_file, "w") as f:
    json.dump(final_output, f, indent=2)

import json
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import sys

if len(sys.argv) != 3 :
    print("Usage : python3 json_to_parquet.py input_file output_file")
    sys.exit(1)


input_file = sys.argv[1]
output_file = sys.argv[2]

# Charger le JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convertir en Table PyArrow
df = pd.DataFrame(data)
table = pa.Table.from_pandas(df)

# Écrire le fichier Parquet
pq.write_table(table, output_file)

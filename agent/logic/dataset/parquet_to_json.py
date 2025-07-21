import sys
import pyarrow.parquet as pq
import pandas as pd
import json
from json import loads

if len(sys.argv) != 2:
    print("Usage : python3 parquet_to_json.py file_name")
    sys.exit(1)

parquet_file = sys.argv[1]
file = parquet_file.split(".parquet")[0]

# Read parquet file
df = pd.read_parquet(parquet_file)

dico = loads(df.to_json())

"""
{
"id":{"1":"__","2":"__"}
"size":{"1":"__","2":"__"}
"puzzle":{"1":"__","2":"__"}
"solution":{"1":"__","2":"__"}
}
"""
"""
[
{"id":__, "size",__}
{"id":__, "size",__}
{"id":__, "size",__}
]
"""
keys = [ k for k in dico.keys()]
res = []
for n in dico[keys[0]].keys():
    dict = {}
    for k in keys:
        dict[k] = dico[k][n]
    res.append(dict)

#print(res[:5])

# Serializing json
json_object = json.dumps(res, indent=4)

# Writing to sample.json
json_file = file + ".json"
with open(json_file, "w") as outfile:
    outfile.write(json_object)

print("[DONE]")

import sys
import pyarrow.parquet as pq

if len(sys.argv) != 2:
    print("Usage : python3 lire_parquet.py file_name")
    sys.exit(1)

file_name = sys.argv[1]

pf = pq.ParquetFile(file_name)
print("Nombre de row groups :", pf.num_row_groups)
print("Nombre total de lignes :", pf.metadata.num_rows)

table = pf.read_row_group(0)
print(table)
print("#" * 50)
print(table.to_pylist())

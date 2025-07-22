import sys
import pyarrow.parquet as pq

def read_parquet_file(file_name: str) -> None:
    """Prints metadata and first row group of a Parquet file."""
    pf = pq.ParquetFile(file_name)
    print("Number of row groups:", pf.num_row_groups)
    print("Total number of rows:", pf.metadata.num_rows)

    table = pf.read_row_group(0)
    print(table)
    print("#" * 50)
    print(table.to_pylist())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 lire_parquet.py file_name.parquet")
        sys.exit(1)

    read_parquet_file(sys.argv[1])

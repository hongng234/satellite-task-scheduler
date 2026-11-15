"""
Preview the generated dataset: first and last N rows of each Parquet file.

Usage:
    python -m src.preview_dataset                          # 200 head + 200 tail rows of every file
    python -m src.preview_dataset --rows 50                # fewer rows
    python -m src.preview_dataset --file requests          # only one table
    python -m src.preview_dataset --save-csv               # also write output/preview/<name>_head.csv / _tail.csv
    python -m src.preview_dataset --data-dir data/benchmark/small
"""
from __future__ import annotations     # Lets type hints like `Path | None` work on every Python 3 version.

import argparse                         # Reads command-line options (--rows, --file, ...).
from pathlib import Path                # File paths that work on Windows and Linux.

import pandas as pd                     # Reads the Parquet files and prints them as tables.


## Key point: this is a viewer. It reads the dataset, prints it, and writes nothing back to it.
## Parquet is a binary format that a text editor cannot open, so this script is the way to look at the data.


# Table name -> Parquet file name inside --data-dir.
# Ex: "requests" -> data/processed/requests.parquet
TABLES = {
    "satellites": "satellites.parquet",
    "requests": "requests.parquet",
    "opportunities": "opportunities.parquet",
}


def configure_pandas() -> None:
    """Make pandas print every column and every requested row, on wide lines."""
    pd.set_option("display.max_columns", None)      # Show all columns.
    pd.set_option("display.width", 250)             # Lines up to 250 characters wide before wrapping.
    pd.set_option("display.max_colwidth", 40)       # Cut cell text longer than 40 characters.
    pd.set_option("display.max_rows", None)         # Print every requested row. pandas would otherwise replace the middle with '...'.


def print_section(title: str) -> None:
    """
    Print a title between two separator lines.

    Ex:
        ====================...
        REQUESTS  -  500 rows x 9 columns
        ====================...
    """
    print("\n" + "=" * 100)     # "=" * 100 is a string of 100 "=" characters.
    print(title)
    print("=" * 100)


def preview_table(name: str, df: pd.DataFrame, rows: int, save_dir: Path | None,) -> None:
    """
    Print a summary of one table, then its first and last `rows` rows.

    Parameters:
        name:     table name, ex: "requests".
        df:       the table itself.
        rows:     how many rows to show at the start and at the end.
        save_dir: folder for the CSV export, or None for no export.
    """
    # len(df) is the number of rows and df.shape[1] the number of columns. {:,} adds thousands separators (66,000).
    print_section(f"{name.upper()}  -  {len(df):,} rows x {df.shape[1]} columns")

    # Data type of each column. Ex: window_start -> datetime64[ns, UTC]
    print("\n-- dtypes --")
    print(df.dtypes.to_string())

    # Basic statistics of numeric columns: count, mean, std, min, quartiles, max.
    numeric = df.select_dtypes("number")            # The numeric columns only.
    if not numeric.empty:                           # A table with no numeric column has nothing to describe.
        print("\n-- describe (numeric columns) --")
        print(numeric.describe().T.to_string())     # .T transposes the result: one row per column, which is easier to read.

    # First and last rows. In a table shorter than 2*rows the head and tail would overlap, so it is printed once.
    # Ex: rows = 200 and a table of 250 rows. Head and tail would share 150 rows, so all 250 are printed once.
    if len(df) <= 2 * rows:
        print(f"\n-- all {len(df)} rows --")
        print(df.to_string())
    else:
        print(f"\n-- first {rows} rows --")
        print(df.head(rows).to_string())

        print(f"\n-- last {rows} rows --")
        print(df.tail(rows).to_string())

    # CSV export, when asked for. A CSV is easier to scroll through in Excel or VS Code.
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)                                 # Create the folder if it is missing.
        df.head(rows).to_csv(save_dir / f"{name}_head.csv", index=False)            # Ex: output/preview/requests_head.csv
        df.tail(rows).to_csv(save_dir / f"{name}_tail.csv", index=False)            # Ex: output/preview/requests_tail.csv
        print(f"\nSaved: {save_dir / f'{name}_head.csv'} and {save_dir / f'{name}_tail.csv'}")


def main() -> None:
    # ---- 1. Read the command-line options ----
    parser = argparse.ArgumentParser(description="Preview the first/last rows of the generated dataset.")

    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Folder with the 3 Parquet files.")
    parser.add_argument("--rows", type=int, default=200, help="Number of rows to show at the head and at the tail.")
    # With choices=[...], argparse rejects any other value. *TABLES.keys() expands to "satellites", "requests", "opportunities".
    parser.add_argument("--file", choices=["all", *TABLES.keys()], default="all", help="Which table to preview.")
    parser.add_argument("--save-csv", action="store_true", help="Also save head/tail as CSV in output/preview/.")

    args = parser.parse_args()      # Ex: args.rows = 200, args.file = "all". A "-" in an option name becomes "_".

    configure_pandas()

    # Tables to show: all of them, or the one passed with --file.
    names = list(TABLES.keys()) if args.file == "all" else [args.file]

    # CSV files go to output/preview/ when --save-csv is given. None means no export.
    save_dir = Path("output/preview") if args.save_csv else None

    # ---- 2. Show each table ----
    for name in names:
        path = args.data_dir / TABLES[name]         # Ex: data/processed/requests.parquet

        # A missing file is reported, and the loop moves on to the next table.
        if not path.exists():
            print(f"\n[skip] {path} not found. Run `python -m src.generate_dataset` first.")
            continue

        preview_table(name=name, df=pd.read_parquet(path), rows=args.rows, save_dir=save_dir,)


# main() runs when the file is executed directly (python -m src.preview_dataset). It does not run on import.
if __name__ == "__main__":
    main()

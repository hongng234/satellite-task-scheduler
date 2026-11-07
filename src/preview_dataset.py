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


## Key point: this is only a viewer. It reads the dataset and prints it; it changes nothing.
## Parquet is a binary format, so the files cannot be opened in a text editor. This script shows them.


# Table name -> Parquet file name inside --data-dir.
# Ex: "requests" -> data/processed/requests.parquet
TABLES = {
    "satellites": "satellites.parquet",
    "requests": "requests.parquet",
    "opportunities": "opportunities.parquet",
}


def configure_pandas() -> None:
    """Show every column and avoid line wrapping / '...' truncation when printing DataFrames."""
    pd.set_option("display.max_columns", None)      # show all columns
    pd.set_option("display.width", 250)             # wide lines instead of wrapping
    pd.set_option("display.max_colwidth", 40)       # cut very long cell text
    pd.set_option("display.max_rows", None)         # print all requested rows (no '...')


def print_section(title: str) -> None:
    """
    Print a visible separator line with a title.

    Ex:
        ====================...
        REQUESTS  -  500 rows x 9 columns
        ====================...
    """
    print("\n" + "=" * 100)     # "=" * 100 -> a line of 100 "=" characters.
    print(title)
    print("=" * 100)


def preview_table(name: str, df: pd.DataFrame, rows: int, save_dir: Path | None,) -> None:
    """
    Print summary info + head(rows) + tail(rows) for one table.

    Parameters:
        name:     table name, ex: "requests".
        df:       the table itself.
        rows:     how many rows to show at the start and at the end.
        save_dir: folder for the CSV export, or None for no export.
    """
    # len(df) = number of rows, df.shape[1] = number of columns. {:,} adds thousands separators (66,000).
    print_section(f"{name.upper()}  -  {len(df):,} rows x {df.shape[1]} columns")

    # Data type of each column (ex: window_start -> datetime64[ns, UTC]).
    print("\n-- dtypes --")
    print(df.dtypes.to_string())

    # Basic statistics of numeric columns: count, mean, std, min, quartiles, max.
    numeric = df.select_dtypes("number")            # Keep only the numeric columns.
    if not numeric.empty:                           # Skip if the table has no numeric column.
        print("\n-- describe (numeric columns) --")
        print(numeric.describe().T.to_string())     # .T (transpose): one row per column, easier to read.

    # First and last rows. If the table is shorter than 2*rows, head and tail overlap -> print it once.
    # Ex: rows = 200, table of 250 rows -> head and tail would share 150 rows -> print all 250 once.
    if len(df) <= 2 * rows:
        print(f"\n-- all {len(df)} rows --")
        print(df.to_string())
    else:
        print(f"\n-- first {rows} rows --")
        print(df.head(rows).to_string())

        print(f"\n-- last {rows} rows --")
        print(df.tail(rows).to_string())

    # Optional CSV export (easier to scroll in Excel / VS Code).
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)                                 # Create the folder if missing.
        df.head(rows).to_csv(save_dir / f"{name}_head.csv", index=False)            # Ex: output/preview/requests_head.csv
        df.tail(rows).to_csv(save_dir / f"{name}_tail.csv", index=False)            # Ex: output/preview/requests_tail.csv
        print(f"\nSaved: {save_dir / f'{name}_head.csv'} and {save_dir / f'{name}_tail.csv'}")


def main() -> None:
    # ---- 1. Read the command-line options ----
    parser = argparse.ArgumentParser(description="Preview the first/last rows of the generated dataset.")

    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Folder with the 3 Parquet files.")
    parser.add_argument("--rows", type=int, default=200, help="Number of rows to show at the head and at the tail.")
    # choices=[...] -> argparse rejects any other value. *TABLES.keys() expands to "satellites", "requests", "opportunities".
    parser.add_argument("--file", choices=["all", *TABLES.keys()], default="all", help="Which table to preview.")
    parser.add_argument("--save-csv", action="store_true", help="Also save head/tail as CSV in output/preview/.")

    args = parser.parse_args()      # Ex: args.rows = 200, args.file = "all". ("-" in option names becomes "_".)

    configure_pandas()

    # Tables to show: all of them, or only the one passed with --file.
    names = list(TABLES.keys()) if args.file == "all" else [args.file]

    # Where to save the CSV files: output/preview/ if --save-csv is given, otherwise None (no export).
    save_dir = Path("output/preview") if args.save_csv else None

    # ---- 2. Show each table ----
    for name in names:
        path = args.data_dir / TABLES[name]         # Ex: data/processed/requests.parquet

        # Missing file -> print a hint and move on to the next table (no crash).
        if not path.exists():
            print(f"\n[skip] {path} not found. Run `python -m src.generate_dataset` first.")
            continue

        preview_table(name=name, df=pd.read_parquet(path), rows=args.rows, save_dir=save_dir,)


# Runs main() only when the file is executed directly (python -m src.preview_dataset), not when imported.
if __name__ == "__main__":
    main()

"""
Preview the generated dataset: first and last N rows of each Parquet file.

Usage:
    python -m src.preview_dataset                          # 200 head + 200 tail rows of every file
    python -m src.preview_dataset --rows 50                # fewer rows
    python -m src.preview_dataset --file requests          # only one table
    python -m src.preview_dataset --save-csv               # also write output/preview/<name>_head.csv / _tail.csv
    python -m src.preview_dataset --data-dir data/benchmark/small
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# Table name -> Parquet file name inside --data-dir.
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
    """Print a visible separator line with a title."""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def preview_table(name: str, df: pd.DataFrame, rows: int, save_dir: Path | None,) -> None:
    """Print summary info + head(rows) + tail(rows) for one table."""
    print_section(f"{name.upper()}  -  {len(df):,} rows x {df.shape[1]} columns")

    # Data type of each column (ex: window_start -> datetime64[ns, UTC]).
    print("\n-- dtypes --")
    print(df.dtypes.to_string())

    # Basic statistics of numeric columns: count, mean, std, min, quartiles, max.
    numeric = df.select_dtypes("number")
    if not numeric.empty:
        print("\n-- describe (numeric columns) --")
        print(numeric.describe().T.to_string())

    # First and last rows. If the table is shorter than 2*rows, head and tail overlap -> print it once.
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
        save_dir.mkdir(parents=True, exist_ok=True)
        df.head(rows).to_csv(save_dir / f"{name}_head.csv", index=False)
        df.tail(rows).to_csv(save_dir / f"{name}_tail.csv", index=False)
        print(f"\nSaved: {save_dir / f'{name}_head.csv'} and {save_dir / f'{name}_tail.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview the first/last rows of the generated dataset.")

    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Folder with the 3 Parquet files.")
    parser.add_argument("--rows", type=int, default=200, help="Number of rows to show at the head and at the tail.")
    parser.add_argument("--file", choices=["all", *TABLES.keys()], default="all", help="Which table to preview.")
    parser.add_argument("--save-csv", action="store_true", help="Also save head/tail as CSV in output/preview/.")

    args = parser.parse_args()

    configure_pandas()

    # Tables to show: all of them, or only the one passed with --file.
    names = list(TABLES.keys()) if args.file == "all" else [args.file]

    save_dir = Path("output/preview") if args.save_csv else None

    for name in names:
        path = args.data_dir / TABLES[name]

        if not path.exists():
            print(f"\n[skip] {path} not found. Run `python -m src.generate_dataset` first.")
            continue

        preview_table(name=name, df=pd.read_parquet(path), rows=args.rows, save_dir=save_dir,)


if __name__ == "__main__":
    main()

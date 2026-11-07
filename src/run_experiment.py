from __future__ import annotations     # Lets type hints use names defined later in the file.

import argparse                         # Reads command-line options (--time-limit, --no-hint, ...).
import json                             # Writes the full results as a JSON file.
from pathlib import Path                # File paths that work on Windows and Linux.

import pandas as pd                     # Builds the comparison table and the schedule tables.

from src.models import ScheduledObservation, ScheduleResult                # Shared output format of every scheduler.
from src.scheduler_cp_sat import SatelliteTaskingScheduler                 # Exact optimizer (CP-SAT).
from src.scheduler_greedy import GreedyScheduler, ImprovedGreedyScheduler  # The two greedy heuristics.


## Key point: this script runs the 3 algorithms on the SAME dataset and writes their results side by side.
##
## Usage:
##   python -m src.run_experiment                    # default: data/processed -> output/
##   python -m src.run_experiment --time-limit 60    # give CP-SAT more time
##   python -m src.run_experiment --no-hint          # CP-SAT without the greedy warm start
##
## Output files (in --output-dir, default output/):
##   scheduler_comparison.csv          one row per algorithm (objective, runtime, served requests, ...)
##   cp_sat_schedule.parquet           the chosen captures of each algorithm
##   greedy_schedule.parquet
##   greedy_improved_schedule.parquet
##   experiment_summary.json           everything above, including every capture, as JSON


# Columns of a schedule table. Used so an empty schedule still produces a readable Parquet file.
# model_fields lists the fields of the Pydantic model, in declaration order.
# Ex:
#   ["request_id", "satellite_id", "opportunity_id", "start", "end", "value", "priority"]
SCHEDULE_COLUMNS = list(ScheduledObservation.model_fields.keys())


def result_to_row(result: ScheduleResult, method_name: str,) -> dict:
    """
    One summary row (dict) per algorithm for the comparison table.

    Ex:
        {"method": "cp_sat", "status": "OPTIMAL", "objective_value": 2941.52,
         "runtime_seconds": 0.042, "scheduled_requests": 46, "solver_status": "OPTIMAL"}
    """
    return {
        "method": method_name,                                  # "cp_sat", "greedy" or "greedy_improved".
        "status": result.status,                                # OPTIMAL / FEASIBLE / INFEASIBLE / UNKNOWN.
        "objective_value": round(result.objective_value, 2),    # Total value served, 2 decimals.
        "runtime_seconds": round(result.runtime_seconds, 3),    # Time taken by solve(), 3 decimals.
        "scheduled_requests": len(result.scheduled),     # each request is served at most once -> rows = served requests
        "solver_status": result.solver_status,                  # Raw solver status, or "GREEDY_HEURISTIC".
    }


def scheduled_to_dataframe(result: ScheduleResult) -> pd.DataFrame:
    """Convert the list of ScheduledObservation into a DataFrame (one row per capture)."""
    rows = [observation.model_dump(mode="json") for observation in result.scheduled]   # mode="json": datetimes -> ISO strings

    # columns=... keeps the expected columns even when the schedule is empty.
    return pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)


def add_improvement_column(comparison: pd.DataFrame, baseline_method: str = "greedy",) -> pd.DataFrame:
    """
    Add 'value_improvement_vs_greedy' = relative gain over the baseline greedy.

    Formula:
        improvement = objective / objective_baseline - 1
    Ex:
        CP-SAT = 36,770, greedy = 36,509  -> improvement = 0.0072 (= +0.72 %)
    If the baseline objective is 0, the ratio is undefined -> NaN.
    """
    # Objective of the baseline row: filter the row where method == "greedy", take its objective_value.
    # .iloc[0] -> the first (and only) matching value.
    baseline = comparison.loc[comparison["method"] == baseline_method, "objective_value"].iloc[0]

    if baseline > 0:
        # Computed for every row at once (pandas column arithmetic). The greedy row itself gets 0.
        comparison["value_improvement_vs_greedy"] = comparison["objective_value"] / baseline - 1
    else:
        # Division by 0 is not meaningful -> NaN ("not a number") for every row.
        comparison["value_improvement_vs_greedy"] = float("nan")

    return comparison


def run_all(satellites_path: Path, requests_path: Path, opportunities_path: Path, time_limit: float, workers: int, slew_buffer_seconds: float, use_greedy_hint: bool = True,) -> dict[str, ScheduleResult]:
    """
    Run the 3 algorithms on the same dataset.
    Also used by benchmark.py for every scenario.

    Returns:
        {"cp_sat": result, "greedy": result, "greedy_improved": result}
    """
    # The 3 file paths, bundled once. **paths unpacks them as keyword arguments:
    #   GreedyScheduler(**paths) == GreedyScheduler(satellites_path=..., requests_path=..., opportunities_path=...)
    paths = dict(satellites_path=satellites_path, requests_path=requests_path, opportunities_path=opportunities_path,)

    results: dict[str, ScheduleResult] = {}     # method name -> result, filled below.

    # 1. Exact optimizer (may take up to time_limit seconds).
    print("Running CP-SAT scheduler...")
    results["cp_sat"] = SatelliteTaskingScheduler(**paths, time_limit_seconds=time_limit, num_workers=workers, slew_buffer_seconds=slew_buffer_seconds, use_greedy_hint=use_greedy_hint,).solve()

    # 2. Baseline greedy: start at window_start or skip.
    print("Running greedy scheduler (baseline)...")
    results[GreedyScheduler.METHOD_NAME] = GreedyScheduler(**paths, slew_buffer_seconds=slew_buffer_seconds,).solve()

    # 3. Improved greedy: earliest free start inside the window.
    print("Running improved greedy scheduler...")
    results[ImprovedGreedyScheduler.METHOD_NAME] = ImprovedGreedyScheduler(**paths, slew_buffer_seconds=slew_buffer_seconds,).solve()

    return results


def main() -> None:
    # ---- 1. Read the command-line options ----
    parser = argparse.ArgumentParser(description="Run satellite tasking scheduling experiment")

    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))      # Folder with the 3 Parquet files.
    parser.add_argument("--output-dir", type=Path, default=Path("output"))            # Where results are written.
    parser.add_argument("--time-limit", type=float, default=30.0)                      # CP-SAT time limit (seconds).
    parser.add_argument("--workers", type=int, default=8)                              # CP-SAT threads.
    parser.add_argument("--slew-buffer-seconds", type=float, default=30.0)             # Min idle time between 2 captures.
    parser.add_argument("--no-hint", action="store_true", help="Run CP-SAT without the greedy warm start.")   # Flag: present -> True.

    args = parser.parse_args()      # Ex: args.time_limit = 30.0, args.no_hint = False. ("-" in option names becomes "_".)

    args.output_dir.mkdir(parents=True, exist_ok=True)      # Create output/ if missing (no error if it already exists).

    # ---- 2. Locate the input files ----
    satellites_path = args.data_dir / "satellites.parquet"         # The / operator joins path parts.
    requests_path = args.data_dir / "requests.parquet"
    opportunities_path = args.data_dir / "opportunities.parquet"

    # Fail early with a clear message if the dataset was not generated.
    for path in [satellites_path, requests_path, opportunities_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}. Run `python -m src.generate_dataset` first.")

    # ---- 3. Run the 3 algorithms ----
    results = run_all(
        satellites_path=satellites_path,
        requests_path=requests_path,
        opportunities_path=opportunities_path,
        time_limit=args.time_limit,
        workers=args.workers,
        slew_buffer_seconds=args.slew_buffer_seconds,
        use_greedy_hint=not args.no_hint,       # --no-hint given -> use_greedy_hint = False.
    )

    # ---- Comparison table: one row per algorithm ----
    comparison = pd.DataFrame([result_to_row(result, method) for method, result in results.items()])
    comparison = add_improvement_column(comparison)

    comparison.to_csv(args.output_dir / "scheduler_comparison.csv", index=False,)     # index=False: do not write the 0, 1, 2 row numbers.

    # ---- One schedule file per algorithm: output/<method>_schedule.parquet ----
    schedule_files = []         # Paths written, printed at the end.
    for method, result in results.items():
        path = args.output_dir / f"{method}_schedule.parquet"      # Ex: output/cp_sat_schedule.parquet
        scheduled_to_dataframe(result).to_parquet(path, index=False,)
        schedule_files.append(path)

    # ---- Full results (all observations) as JSON ----
    # `with open(...) as f` closes the file automatically at the end of the block.
    # default=str converts any value json cannot write natively (ex: datetime) to text.
    with open(args.output_dir / "experiment_summary.json", "w", encoding="utf-8",) as f:
        json.dump({method: result.model_dump(mode="json") for method, result in results.items()}, f, indent=2, default=str,)

    # ---- 4. Print a summary on screen ----
    print("\n=== Comparison ===")
    print(comparison.to_string(index=False))

    print("\nOutput files:")
    print(f"- {args.output_dir / 'scheduler_comparison.csv'}")
    for path in schedule_files:
        print(f"- {path}")
    print(f"- {args.output_dir / 'experiment_summary.json'}")


# Runs main() only when the file is executed directly (python -m src.run_experiment),
# not when it is imported by another module (benchmark.py imports run_all from here).
if __name__ == "__main__":
    main()

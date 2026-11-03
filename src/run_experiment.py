from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.models import ScheduledObservation, ScheduleResult
from src.scheduler_cp_sat import SatelliteTaskingScheduler
from src.scheduler_greedy import GreedyScheduler, ImprovedGreedyScheduler


# Columns of a schedule table. Used so an empty schedule still produces a readable Parquet file.
SCHEDULE_COLUMNS = list(ScheduledObservation.model_fields.keys())


def result_to_row(result: ScheduleResult, method_name: str,) -> dict:
    """One summary row (dict) per algorithm for the comparison table."""
    return {
        "method": method_name,
        "status": result.status,
        "objective_value": round(result.objective_value, 2),
        "runtime_seconds": round(result.runtime_seconds, 3),
        "scheduled_requests": len(result.scheduled),     # each request is served at most once -> rows = served requests
        "solver_status": result.solver_status,
    }


def scheduled_to_dataframe(result: ScheduleResult) -> pd.DataFrame:
    """Convert the list of ScheduledObservation into a DataFrame (one row per capture)."""
    rows = [observation.model_dump(mode="json") for observation in result.scheduled]   # mode="json": datetimes -> ISO strings

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
    baseline = comparison.loc[comparison["method"] == baseline_method, "objective_value"].iloc[0]

    if baseline > 0:
        comparison["value_improvement_vs_greedy"] = comparison["objective_value"] / baseline - 1
    else:
        comparison["value_improvement_vs_greedy"] = float("nan")

    return comparison


def run_all(satellites_path: Path, requests_path: Path, opportunities_path: Path, time_limit: float, workers: int, slew_buffer_seconds: float, use_greedy_hint: bool = True,) -> dict[str, ScheduleResult]:
    """
    Run the 3 algorithms on the same dataset.

    Returns:
        {"cp_sat": result, "greedy": result, "greedy_improved": result}
    """
    paths = dict(satellites_path=satellites_path, requests_path=requests_path, opportunities_path=opportunities_path,)

    results: dict[str, ScheduleResult] = {}

    print("Running CP-SAT scheduler...")
    results["cp_sat"] = SatelliteTaskingScheduler(**paths, time_limit_seconds=time_limit, num_workers=workers, slew_buffer_seconds=slew_buffer_seconds, use_greedy_hint=use_greedy_hint,).solve()

    print("Running greedy scheduler (baseline)...")
    results[GreedyScheduler.METHOD_NAME] = GreedyScheduler(**paths, slew_buffer_seconds=slew_buffer_seconds,).solve()

    print("Running improved greedy scheduler...")
    results[ImprovedGreedyScheduler.METHOD_NAME] = ImprovedGreedyScheduler(**paths, slew_buffer_seconds=slew_buffer_seconds,).solve()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run satellite tasking scheduling experiment")

    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))      # Folder with the 3 Parquet files.
    parser.add_argument("--output-dir", type=Path, default=Path("output"))            # Where results are written.
    parser.add_argument("--time-limit", type=float, default=30.0)                      # CP-SAT time limit (seconds).
    parser.add_argument("--workers", type=int, default=8)                              # CP-SAT threads.
    parser.add_argument("--slew-buffer-seconds", type=float, default=30.0)             # Min idle time between 2 captures.
    parser.add_argument("--no-hint", action="store_true", help="Run CP-SAT without the greedy warm start.")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    satellites_path = args.data_dir / "satellites.parquet"
    requests_path = args.data_dir / "requests.parquet"
    opportunities_path = args.data_dir / "opportunities.parquet"

    # Fail early with a clear message if the dataset was not generated.
    for path in [satellites_path, requests_path, opportunities_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}. Run `python -m src.generate_dataset` first.")

    results = run_all(
        satellites_path=satellites_path,
        requests_path=requests_path,
        opportunities_path=opportunities_path,
        time_limit=args.time_limit,
        workers=args.workers,
        slew_buffer_seconds=args.slew_buffer_seconds,
        use_greedy_hint=not args.no_hint,
    )

    # ---- Comparison table: one row per algorithm ----
    comparison = pd.DataFrame([result_to_row(result, method) for method, result in results.items()])
    comparison = add_improvement_column(comparison)

    comparison.to_csv(args.output_dir / "scheduler_comparison.csv", index=False,)

    # ---- One schedule file per algorithm: output/<method>_schedule.parquet ----
    schedule_files = []
    for method, result in results.items():
        path = args.output_dir / f"{method}_schedule.parquet"
        scheduled_to_dataframe(result).to_parquet(path, index=False,)
        schedule_files.append(path)

    # ---- Full results (all observations) as JSON ----
    with open(args.output_dir / "experiment_summary.json", "w", encoding="utf-8",) as f:
        json.dump({method: result.model_dump(mode="json") for method, result in results.items()}, f, indent=2, default=str,)

    print("\n=== Comparison ===")
    print(comparison.to_string(index=False))

    print("\nOutput files:")
    print(f"- {args.output_dir / 'scheduler_comparison.csv'}")
    for path in schedule_files:
        print(f"- {path}")
    print(f"- {args.output_dir / 'experiment_summary.json'}")


if __name__ == "__main__":
    main()

"""
Benchmark: compare CP-SAT, greedy and improved greedy on datasets of different sizes.

For each scenario:
    1. generate a dataset (same generator as src/generate_dataset.py) into data/benchmark/<scenario>/
    2. run the 3 algorithms on it
    3. check every schedule against all constraints (a fast but wrong schedule is worthless)
    4. record objective, number of served requests, runtime, status, gain vs. greedy

Usage:
    python -m src.benchmark                                  # default scenarios: tiny, small, medium, large
    python -m src.benchmark --scenarios tiny small           # only some scenarios
    python -m src.benchmark --scenarios xlarge --time-limit 120
    python -m src.benchmark --no-hint                        # CP-SAT without greedy warm start (to see the difference)
    python -m src.benchmark --generate-only                  # only write data/benchmark/<scenario>/*.parquet, no solving

Results:
    printed table + output/benchmark_results.csv
"""
from __future__ import annotations     # Lets type hints use names defined later in the file.

import argparse                         # Reads command-line options (--scenarios, --time-limit, ...).
import math                             # math.nan: "not a number", used when a ratio is undefined.
import time                             # perf_counter(): precise clock to measure generation time.
import traceback                        # Prints the full error message when a scenario crashes.
from dataclasses import dataclass       # Short way to declare a class that only stores values.
from pathlib import Path                # File paths that work on Windows and Linux.

import pandas as pd                     # Result tables and the constraint checks.

from src.generate_dataset import DEFAULT_START_TIME, generate_all, parse_start_time, save_dataset  # Same generator as `make dataset`.
from src.models import ScheduleResult                                                           # Output format of every scheduler.
from src.run_experiment import run_all                                                          # Runs the 3 algorithms on one dataset.


# One benchmark scenario = the parameters of one generated dataset.
# @dataclass(frozen=True): __init__ is generated from the fields below, and the object cannot be modified afterwards.
# Ex:
#   Scenario("tiny", satellites=3, requests=50)   # horizon_hours=48, max_windows_per_pair=3, seed=42 by default
@dataclass(frozen=True)
class Scenario:
    """Size of one generated dataset."""
    name: str                           # Scenario name, also the folder name. Ex: "tiny" -> data/benchmark/tiny/
    satellites: int                     # Number of satellites.
    requests: int                       # Number of customer requests.
    horizon_hours: int = 48             # Length of the planning horizon, in hours.
    max_windows_per_pair: int = 3       # Max visibility windows per (request, satellite) pair.
    seed: int = 42                      # Random seed -> same dataset on every run.


# Opportunities ≈ requests x satellites x 0.55 (visibility chance) x ~2.5 (windows per pair).
# Ex: medium -> 500 x 12 x 0.55 x 2.5 ≈ 8,000 opportunities.
SCENARIOS = {
    "tiny":   Scenario("tiny",   satellites=3,  requests=50),       # ~250 opportunities: CP-SAT proves the optimum in < 0.1 s.
    "small":  Scenario("small",  satellites=6,  requests=200),      # ~2,000 opportunities.
    "medium": Scenario("medium", satellites=12, requests=500),      # ~10,000 opportunities (same size as `make dataset`).
    "large":  Scenario("large",  satellites=20, requests=2000),     # ~66,000 opportunities: CP-SAT needs the warm start.
    "xlarge": Scenario("xlarge", satellites=30, requests=5000),   # slow to build: use a bigger --time-limit
}

# Scenarios run when --scenarios is not given (xlarge is left out because it is slow).
DEFAULT_SCENARIOS = ["tiny", "small", "medium", "large"]


def check_schedule(result: ScheduleResult, satellites: pd.DataFrame, requests: pd.DataFrame, opportunities: pd.DataFrame, slew_buffer_seconds: float,) -> list[str]:
    """
    Verify a schedule against every constraint. Returns a list of violations (empty = valid).

    It is INDEPENDENT of the schedulers: it re-checks the output from the raw tables,
    so a bug inside a scheduler cannot hide itself.

    Checks:
        - each request served at most once
        - start >= window_start and end <= window_end
        - on each satellite: next_start - previous_end >= slew_buffer
        - sum(energy) <= battery_capacity, sum(data) <= memory_capacity
        - captures per orbit block <= max_observations_per_orbit

    Ex:
        check_schedule(result, satellites, requests, opportunities, 30.0)
        -> []                                   (valid)
        -> ["SAT-03: battery exceeded"]         (invalid)
    """
    # An empty schedule breaks no constraint.
    if not result.scheduled:
        return []

    violations = []     # Messages of the broken constraints.
    eps = 1e-6      # tolerance for float rounding

    # The schedule as a table: one row per capture. Start/end parsed as UTC datetimes to compare with the windows.
    schedule = pd.DataFrame([observation.model_dump() for observation in result.scheduled])
    schedule["start"] = pd.to_datetime(schedule["start"], utc=True)
    schedule["end"] = pd.to_datetime(schedule["end"], utc=True)

    # Tables indexed by id, for lookups like sat.loc["SAT-01", "battery_capacity_wh"].
    sat = satellites.set_index("satellite_id")
    req = requests.set_index("request_id")
    opp = opportunities.set_index("opportunity_id")
    base_time = opportunities.loc[opportunities["feasible"], "window_start"].min()   # same time 0 as the schedulers

    # ---- Check 1: one capture per request ----
    # duplicated() -> True for every request_id already seen above. any() -> at least one.
    if schedule["request_id"].duplicated().any():
        violations.append("request served more than once")

    # ---- Check 2: each capture inside its window ----
    # Window of each scheduled capture, aligned row by row with `schedule` (reset_index -> same 0..n-1 index).
    windows = opp.loc[schedule["opportunity_id"]].reset_index(drop=True)
    if (schedule["start"] < windows["window_start"]).any():
        violations.append("capture starts before its window")
    if (schedule["end"] > windows["window_end"]).any():
        violations.append("capture ends after its window")

    # ---- Checks 3-5: per satellite ----
    # groupby: loop over each satellite with only its own captures (`group`).
    for satellite_id, group in schedule.groupby("satellite_id"):
        group = group.sort_values("start")      # Chronological order.

        # Check 3: slew buffer between consecutive captures on this satellite.
        # Formula:
        #   gap_k = start_{k+1} - end_k   must be >= slew_buffer
        # Ex:
        #   capture A ends 08:12:10, capture B starts 08:12:50 -> gap = 40 s >= 30 s -> OK
        next_starts = group["start"].iloc[1:].reset_index(drop=True)        # starts of captures 2..n
        previous_ends = group["end"].iloc[:-1].reset_index(drop=True)       # ends of captures 1..n-1
        gaps = (next_starts - previous_ends).dt.total_seconds()
        if len(gaps) and gaps.min() < slew_buffer_seconds - eps:            # len(gaps) = 0 when the satellite has only 1 capture.
            violations.append(f"{satellite_id}: gap {gaps.min():.1f}s < slew buffer")

        # Check 4: battery and memory budgets.
        # Formula:
        #   sum(energy_i) <= battery_capacity,   sum(data_i) <= memory_capacity
        served = req.loc[group["request_id"]]       # Rows of the requests served by this satellite.
        if served["energy_wh"].sum() > sat.loc[satellite_id, "battery_capacity_wh"] + eps:
            violations.append(f"{satellite_id}: battery exceeded")
        if served["data_volume_gb"].sum() > sat.loc[satellite_id, "memory_capacity_gb"] + eps:
            violations.append(f"{satellite_id}: memory exceeded")

        # Check 5: captures per orbit block.
        # block_id = floor((window_start - base_time) / orbit_seconds)
        orbit_seconds = sat.loc[satellite_id, "orbit_period_minutes"] * 60
        offsets = (opp.loc[group["opportunity_id"], "window_start"] - base_time).dt.total_seconds()
        per_block = (offsets // orbit_seconds).value_counts()       # block_id -> number of captures in that block.
        if (per_block > sat.loc[satellite_id, "max_observations_per_orbit"]).any():
            violations.append(f"{satellite_id}: too many captures in one orbit")

    return violations


def generate_scenario(scenario: Scenario, data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    """
    Generate the dataset of one scenario and save it to <data_root>/<scenario name>/.

    Returns:
        (satellites, requests, opportunities, data_dir)
    Ex:
        generate_scenario(SCENARIOS["tiny"], Path("data/benchmark"))
        -> writes data/benchmark/tiny/{satellites,requests,opportunities}.parquet
    """
    data_dir = data_root / scenario.name        # Ex: data/benchmark/tiny

    # Same generator and fixed start time as `python -m src.generate_dataset` -> reproducible per seed.
    satellites, requests, opportunities = generate_all(
        n_satellites=scenario.satellites,
        n_requests=scenario.requests,
        horizon_hours=scenario.horizon_hours,
        orbit_period_minutes=96.0,
        max_windows_per_pair=scenario.max_windows_per_pair,
        seed=scenario.seed,
        start_time=parse_start_time(DEFAULT_START_TIME),
    )
    save_dataset(satellites=satellites, requests=requests, opportunities=opportunities, output_dir=data_dir,)

    return satellites, requests, opportunities, data_dir


def run_scenario(scenario: Scenario, args: argparse.Namespace) -> list[dict]:
    """
    Generate one dataset, run the 3 algorithms, return one result row per algorithm.

    `args` = the parsed command-line options (time limit, workers, slew buffer, --no-hint, data root).
    """

    # ---- 1. Generate and save the dataset ----
    t0 = time.perf_counter()        # Start the clock.
    satellites, requests, opportunities, data_dir = generate_scenario(scenario, args.data_root)
    generate_seconds = time.perf_counter() - t0     # Seconds spent generating the data.

    print(f"\n### {scenario.name}: {scenario.satellites} satellites, {scenario.requests} requests, {len(opportunities):,} opportunities (generated in {generate_seconds:.1f}s)")

    # ---- 2. Run the 3 algorithms ----
    results = run_all(
        satellites_path=data_dir / "satellites.parquet",
        requests_path=data_dir / "requests.parquet",
        opportunities_path=data_dir / "opportunities.parquet",
        time_limit=args.time_limit,
        workers=args.workers,
        slew_buffer_seconds=args.slew_buffer_seconds,
        use_greedy_hint=not args.no_hint,       # --no-hint given -> use_greedy_hint = False.
    )

    # ---- 3. Check and summarize ----
    greedy_objective = results["greedy"].objective_value        # Reference value for the "gain vs greedy" column.

    rows = []       # One dict per algorithm, becomes one row of benchmark_results.csv.
    for method, result in results.items():
        violations = check_schedule(result, satellites, requests, opportunities, args.slew_buffer_seconds)

        rows.append({
            "scenario": scenario.name,
            "satellites": scenario.satellites,
            "requests": scenario.requests,
            "opportunities": len(opportunities),
            "method": method,                                           # "cp_sat", "greedy" or "greedy_improved".
            "status": result.status,
            "objective_value": round(result.objective_value, 2),        # Total value served.
            "scheduled_requests": len(result.scheduled),                # Number of requests served.
            # Share of all requests that got served. Formula: served / total
            "served_ratio": round(len(result.scheduled) / scenario.requests, 3),
            "runtime_seconds": round(result.runtime_seconds, 2),
            # Formula: objective / objective_greedy - 1   (NaN if greedy = 0)
            "gain_vs_greedy_pct": round(100 * (result.objective_value / greedy_objective - 1), 2) if greedy_objective > 0 else math.nan,
            "valid": not violations,                                    # True when the violation list is empty.
            "violations": "; ".join(violations),                       # All messages in one text cell, separated by "; ".
            "error": "",                                                # Filled only when the scenario crashes (see main()).
        })

    return rows


def main() -> None:
    # ---- Read the command-line options ----
    parser = argparse.ArgumentParser(description="Compare CP-SAT, greedy and improved greedy on several dataset sizes.")

    # nargs="+": one or more values. Ex: --scenarios tiny small
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=DEFAULT_SCENARIOS, help="Scenarios to run.")
    parser.add_argument("--time-limit", type=float, default=30.0, help="CP-SAT time limit per scenario (seconds).")
    parser.add_argument("--workers", type=int, default=8, help="CP-SAT threads.")
    parser.add_argument("--slew-buffer-seconds", type=float, default=30.0)        # Min idle time between 2 captures.
    parser.add_argument("--no-hint", action="store_true", help="Run CP-SAT without the greedy warm start.")
    parser.add_argument("--data-root", type=Path, default=Path("data/benchmark"), help="Where the generated datasets are written.")
    parser.add_argument("--output", type=Path, default=Path("output/benchmark_results.csv"))   # Result CSV.
    parser.add_argument("--generate-only", action="store_true", help="Only generate and save the datasets, do not run the algorithms.")

    args = parser.parse_args()      # "-" in option names becomes "_": --time-limit -> args.time_limit.

    # ---- Mode 1: only generate the datasets (no solving) ----
    if args.generate_only:
        for name in args.scenarios:
            t0 = time.perf_counter()
            satellites, requests, opportunities, data_dir = generate_scenario(SCENARIOS[name], args.data_root)

            # Aligned columns: {name:<7} = left-aligned in 7 characters, {x:>5} = right-aligned in 5.
            print(f"{name:<7} {len(satellites):>3} satellites  {len(requests):>5} requests  {len(opportunities):>7,} opportunities  "
                  f"-> {data_dir}  ({time.perf_counter() - t0:.1f}s)")

        print("\nPreview one with: python -m src.preview_dataset --data-dir data/benchmark/<scenario>")
        return      # Stop here: do not run the algorithms.

    # ---- Mode 2: full benchmark (generate + solve + check) ----

    all_rows = []       # Result rows of every scenario.

    for name in args.scenarios:
        try:
            all_rows.extend(run_scenario(SCENARIOS[name], args))       # extend: add the 3 rows of this scenario.
        except Exception as error:      # One failing scenario must not stop the whole benchmark.
            traceback.print_exc()       # Show the full error, then record it as one row and continue.
            all_rows.append({"scenario": name, "method": "-", "error": f"{type(error).__name__}: {error}"})

    results = pd.DataFrame(all_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)      # Create output/ if missing.
    results.to_csv(args.output, index=False)

    # Short table on screen (the CSV keeps every column).
    columns = ["scenario", "opportunities", "method", "status", "objective_value", "scheduled_requests", "runtime_seconds", "gain_vs_greedy_pct", "valid", "error"]
    pd.set_option("display.width", 250)        # Wide lines instead of wrapping.
    print("\n=== Benchmark results ===")
    print(results.reindex(columns=columns).to_string(index=False))     # reindex: keep only these columns, in this order.
    print(f"\nSaved: {args.output}")


# Runs main() only when the file is executed directly (python -m src.benchmark), not when imported by the tests.
if __name__ == "__main__":
    main()

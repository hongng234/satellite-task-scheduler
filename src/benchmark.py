"""
Benchmark: compare CP-SAT, greedy and improved greedy on datasets of different sizes.

For each scenario:
    1. generate a dataset into data/benchmark/<scenario>/, with the generator from src/generate_dataset.py
    2. run the 3 algorithms on it
    3. check every schedule against all constraints, because a schedule that breaks one cannot be compared
    4. record objective, number of served requests, runtime, status and gain over greedy

Usage:
    python -m src.benchmark                                  # default scenarios: tiny, small, medium, large
    python -m src.benchmark --scenarios tiny small           # only some scenarios
    python -m src.benchmark --scenarios xlarge --time-limit 120
    python -m src.benchmark --no-hint                        # CP-SAT without the greedy warm start, to see what the hint changes
    python -m src.benchmark --generate-only                  # only write data/benchmark/<scenario>/*.parquet, no solving

Results:
    printed table + output/benchmark_results.csv
"""
from __future__ import annotations     # Lets type hints use names defined later in the file.

import argparse                         # Reads command-line options (--scenarios, --time-limit, ...).
import math                             # math.nan ("not a number"), for a ratio that is undefined.
import time                             # perf_counter(): precise clock to measure generation time.
import traceback                        # Prints the full error message when a scenario crashes.
from dataclasses import dataclass       # Short way to declare a class that only stores values.
from pathlib import Path                # File paths that work on Windows and Linux.

import pandas as pd                     # Result tables and the constraint checks.

from src.generate_dataset import DEFAULT_START_TIME, generate_all, parse_start_time, save_dataset  # The generator behind `make dataset`.
from src.models import ScheduleResult                                                           # Output format of every scheduler.
from src.run_experiment import run_all                                                          # Runs the 3 algorithms on one dataset.


# A benchmark scenario is the set of parameters of one generated dataset.
# @dataclass(frozen=True) generates __init__ from the fields below and makes the object read-only.
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
    seed: int = 42                      # Random seed. The same seed gives the same dataset on every run.


# Opportunities are about requests x satellites x 0.55 (visibility chance) x 3 (windows per pair).
# Ex: medium -> 500 x 12 x 0.55 x 3 = 9,900, and the generated dataset has 9,964.
SCENARIOS = {
    "tiny":   Scenario("tiny",   satellites=3,  requests=50),       # ~250 opportunities. CP-SAT proves the optimum in under 0.1 s.
    "small":  Scenario("small",  satellites=6,  requests=200),      # ~2,000 opportunities.
    "medium": Scenario("medium", satellites=12, requests=500),      # ~10,000 opportunities, the size `make dataset` produces.
    "large":  Scenario("large",  satellites=20, requests=2000),     # ~66,000 opportunities. CP-SAT needs the warm start here.
    "xlarge": Scenario("xlarge", satellites=30, requests=5000),     # Slow to build. Use a bigger --time-limit.
}

# Scenarios that run when --scenarios is not given. xlarge is left out because it is slow.
DEFAULT_SCENARIOS = ["tiny", "small", "medium", "large"]


def check_schedule(result: ScheduleResult, satellites: pd.DataFrame, requests: pd.DataFrame, opportunities: pd.DataFrame, slew_buffer_seconds: float,) -> list[str]:
    """
    Check a schedule against the constraints. Returns a list of violations (empty = valid).

    It shares no code with the schedulers. It works from the raw tables and the schedule,
    so a bug inside a scheduler shows up here as a violation.

    Checks:
        - every capture uses an opportunity that exists, is feasible, and belongs to the
          request and satellite the capture names
        - value, priority and objective_value match the requests table
        - each request served at most once
        - the window lies inside the request's [earliest_start, latest_end]
        - start >= window_start, end <= window_end, and end - start >= duration
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

    violations = []     # One message per broken constraint.
    eps = 1e-6          # Tolerance for float rounding.

    # The schedule as a table, one row per capture. start and end become UTC datetimes so they can be compared with the windows.
    schedule = pd.DataFrame([observation.model_dump() for observation in result.scheduled])
    schedule["start"] = pd.to_datetime(schedule["start"], utc=True)
    schedule["end"] = pd.to_datetime(schedule["end"], utc=True)

    # Tables indexed by id, for lookups like sat.loc["SAT-01", "battery_capacity_wh"].
    sat = satellites.set_index("satellite_id")
    req = requests.set_index("request_id")
    opp = opportunities.set_index("opportunity_id")
    base_time = opportunities.loc[opportunities["feasible"], "window_start"].min()   # Second 0, computed the way the schedulers compute it.

    # ---- Check 0: each capture points to a real opportunity, and the ids agree ----
    # The checks after this one look rows up by these ids. If an id is wrong they would compare
    # the wrong rows or crash, so the function stops here and reports the problem.
    unknown = sorted(set(schedule["opportunity_id"]) - set(opp.index))
    if unknown:
        return [f"unknown opportunity ids: {unknown}"]

    # The opportunity row of each scheduled capture. reset_index gives it the same 0..n-1 index as `schedule`, so the rows line up.
    windows = opp.loc[schedule["opportunity_id"]].reset_index(drop=True)

    # An opportunity is for one request and one satellite. A capture that names another one is mislabelled.
    # Ex: capture says request R2, its opportunity O1 is for R1 -> mismatch
    if (schedule["request_id"] != windows["request_id"]).any():
        violations.append("capture request_id does not match its opportunity")
    if (schedule["satellite_id"] != windows["satellite_id"]).any():
        violations.append("capture satellite_id does not match its opportunity")
    if violations:
        return violations

    # The schedulers drop rows with feasible = False, so none of them may appear in a schedule.
    if not windows["feasible"].all():
        violations.append("capture uses an opportunity with feasible = False")

    # ---- Check 1: value, priority and total value are the real ones ----
    # The request row of each capture, aligned the same way as `windows`.
    served_requests = req.loc[schedule["request_id"]].reset_index(drop=True)
    if ((schedule["value"] - served_requests["value"]).abs() > eps).any():
        violations.append("capture value does not match its request")
    if (schedule["priority"].astype(int) != served_requests["priority"].astype(int)).any():
        violations.append("capture priority does not match its request")

    # objective_value must be the sum of the values of the captures.
    # The tolerance grows with the total, because summing many floats in another order changes the last digits.
    if abs(result.objective_value - schedule["value"].sum()) > eps * max(1.0, abs(result.objective_value)):
        violations.append("objective_value is not the sum of the capture values")

    # ---- Check 2: one capture per request ----
    # duplicated() marks every request_id that already appeared in an earlier row. any() is True if at least one is marked.
    if schedule["request_id"].duplicated().any():
        violations.append("request served more than once")

    # ---- Check 3: each capture inside its window, and the window inside its request ----
    if ((windows["window_start"] < served_requests["earliest_start"]) | (windows["window_end"] > served_requests["latest_end"])).any():
        violations.append("window lies outside its request's time range")
    if (schedule["start"] < windows["window_start"]).any():
        violations.append("capture starts before its window")
    if (schedule["end"] > windows["window_end"]).any():
        violations.append("capture ends after its window")

    # A capture must last at least the duration the request needs.
    # CP-SAT rounds durations up to whole seconds, so a capture may be a little longer, never shorter.
    # Ex: duration 10 s, capture [08:12:00, 08:12:02] -> 2 s -> too short
    lengths = (schedule["end"] - schedule["start"]).dt.total_seconds()
    if (lengths < windows["duration_seconds"] - eps).any():
        violations.append("capture is shorter than its duration")

    # ---- Checks 4-6: per satellite ----
    # groupby gives one satellite at a time, with only its own captures in `group`.
    for satellite_id, group in schedule.groupby("satellite_id"):
        group = group.sort_values("start")      # In time order.

        # Check 4: slew buffer between consecutive captures on this satellite.
        # Formula:
        #   gap_k = start_{k+1} - end_k   must be >= slew_buffer
        # Ex:
        #   capture A ends 08:12:10, capture B starts 08:12:50 -> gap = 40 s >= 30 s -> OK
        next_starts = group["start"].iloc[1:].reset_index(drop=True)        # Starts of captures 2..n.
        previous_ends = group["end"].iloc[:-1].reset_index(drop=True)       # Ends of captures 1..n-1.
        gaps = (next_starts - previous_ends).dt.total_seconds()
        if len(gaps) and gaps.min() < slew_buffer_seconds - eps:            # A satellite with one capture has no gap, and len(gaps) is 0.
            violations.append(f"{satellite_id}: gap {gaps.min():.1f}s < slew buffer")

        # Check 5: battery and memory budgets.
        # Formula:
        #   sum(energy_i) <= battery_capacity,   sum(data_i) <= memory_capacity
        served = req.loc[group["request_id"]]       # The requests this satellite serves.
        if served["energy_wh"].sum() > sat.loc[satellite_id, "battery_capacity_wh"] + eps:
            violations.append(f"{satellite_id}: battery exceeded")
        if served["data_volume_gb"].sum() > sat.loc[satellite_id, "memory_capacity_gb"] + eps:
            violations.append(f"{satellite_id}: memory exceeded")

        # Check 6: captures per orbit block.
        # block_id = floor((window_start - base_time) / orbit_seconds)
        orbit_seconds = sat.loc[satellite_id, "orbit_period_minutes"] * 60
        offsets = (opp.loc[group["opportunity_id"], "window_start"] - base_time).dt.total_seconds()
        per_block = (offsets // orbit_seconds).value_counts()       # block_id -> number of captures in that block.
        if (per_block > sat.loc[satellite_id, "max_observations_per_orbit"]).any():
            violations.append(f"{satellite_id}: too many captures in one orbit")

    return violations


def generate_scenario(scenario: Scenario, data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    """
    Generate the dataset of one scenario and save it in <data_root>/<scenario name>/.

    Returns:
        (satellites, requests, opportunities, data_dir)
    Ex:
        generate_scenario(SCENARIOS["tiny"], Path("data/benchmark"))
        -> writes data/benchmark/tiny/{satellites,requests,opportunities}.parquet
    """
    data_dir = data_root / scenario.name        # Ex: data/benchmark/tiny

    # The generator and the fixed start time are the ones `python -m src.generate_dataset` uses,
    # so a seed always gives the same dataset.
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
    Generate one dataset, run the 3 algorithms and return one result row per algorithm.

    `args` holds the parsed command-line options (time limit, workers, slew buffer, --no-hint, data root).
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
        use_greedy_hint=not args.no_hint,       # With --no-hint, use_greedy_hint is False.
    )

    # ---- 3. Check and summarize ----
    greedy_objective = results["greedy"].objective_value        # The "gain vs greedy" column is measured against this.

    rows = []       # One dict per algorithm. Each becomes a row of benchmark_results.csv.
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
            # Share of all requests that were served. Formula: served / total
            "served_ratio": round(len(result.scheduled) / scenario.requests, 3),
            "runtime_seconds": round(result.runtime_seconds, 2),
            # Formula: objective / objective_greedy - 1   (NaN if greedy = 0)
            "gain_vs_greedy_pct": round(100 * (result.objective_value / greedy_objective - 1), 2) if greedy_objective > 0 else math.nan,
            "valid": not violations,                                    # True when the violation list is empty.
            "violations": "; ".join(violations),                       # All messages in one cell, separated by "; ".
            "error": "",                                                # Empty here. main() fills it when a scenario crashes.
        })

    return rows


def main() -> None:
    # ---- Read the command-line options ----
    parser = argparse.ArgumentParser(description="Compare CP-SAT, greedy and improved greedy on several dataset sizes.")

    # nargs="+" accepts one or more values. Ex: --scenarios tiny small
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=DEFAULT_SCENARIOS, help="Scenarios to run.")
    parser.add_argument("--time-limit", type=float, default=30.0, help="CP-SAT time limit per scenario (seconds).")
    parser.add_argument("--workers", type=int, default=8, help="CP-SAT threads.")
    parser.add_argument("--slew-buffer-seconds", type=float, default=30.0)        # Min idle time between 2 captures.
    parser.add_argument("--no-hint", action="store_true", help="Run CP-SAT without the greedy warm start.")
    parser.add_argument("--data-root", type=Path, default=Path("data/benchmark"), help="Where the generated datasets are written.")
    parser.add_argument("--output", type=Path, default=Path("output/benchmark_results.csv"))   # Result CSV.
    parser.add_argument("--generate-only", action="store_true", help="Only generate and save the datasets, do not run the algorithms.")

    args = parser.parse_args()      # A "-" in an option name becomes "_": --time-limit -> args.time_limit.

    # ---- Mode 1: generate the datasets and stop ----
    if args.generate_only:
        for name in args.scenarios:
            t0 = time.perf_counter()
            satellites, requests, opportunities, data_dir = generate_scenario(SCENARIOS[name], args.data_root)

            # Aligned columns: {name:<7} is left-aligned in 7 characters, {x:>5} is right-aligned in 5.
            print(f"{name:<7} {len(satellites):>3} satellites  {len(requests):>5} requests  {len(opportunities):>7,} opportunities  "
                  f"-> {data_dir}  ({time.perf_counter() - t0:.1f}s)")

        print("\nPreview one with: python -m src.preview_dataset --data-dir data/benchmark/<scenario>")
        return      # The algorithms do not run in this mode.

    # ---- Mode 2: full benchmark (generate, solve, check) ----

    all_rows = []       # Result rows of every scenario.

    for name in args.scenarios:
        try:
            all_rows.extend(run_scenario(SCENARIOS[name], args))       # extend adds the 3 rows of this scenario.
        except Exception as error:      # noqa: BLE001  If one scenario fails, the others still run, so any error is caught here.
            traceback.print_exc()       # Print the full error, record it as one row, and go on.
            all_rows.append({"scenario": name, "method": "-", "error": f"{type(error).__name__}: {error}"})

    results = pd.DataFrame(all_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)      # Create output/ if it is missing.
    results.to_csv(args.output, index=False)

    # A shorter table on screen. The CSV has every column.
    columns = ["scenario", "opportunities", "method", "status", "objective_value", "scheduled_requests", "runtime_seconds", "gain_vs_greedy_pct", "valid", "error"]
    pd.set_option("display.width", 250)        # Lines up to 250 characters wide before wrapping.
    print("\n=== Benchmark results ===")
    print(results.reindex(columns=columns).to_string(index=False))     # reindex keeps these columns, in this order.
    print(f"\nSaved: {args.output}")


# main() runs when the file is executed directly (python -m src.benchmark). It does not run when the tests import this file.
if __name__ == "__main__":
    main()

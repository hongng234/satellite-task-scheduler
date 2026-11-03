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
from __future__ import annotations

import argparse
import math
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.generate_dataset import DEFAULT_START_TIME, generate_all, parse_start_time, save_dataset
from src.models import ScheduleResult
from src.run_experiment import run_all


@dataclass(frozen=True)
class Scenario:
    """Size of one generated dataset."""
    name: str
    satellites: int
    requests: int
    horizon_hours: int = 48
    max_windows_per_pair: int = 3
    seed: int = 42


# Opportunities ≈ requests x satellites x 0.55 (visibility chance) x ~2.5 (windows per pair).
# Ex: medium -> 500 x 12 x 0.55 x 2.5 ≈ 8,000 opportunities.
SCENARIOS = {
    "tiny":   Scenario("tiny",   satellites=3,  requests=50),
    "small":  Scenario("small",  satellites=6,  requests=200),
    "medium": Scenario("medium", satellites=12, requests=500),
    "large":  Scenario("large",  satellites=20, requests=2000),
    "xlarge": Scenario("xlarge", satellites=30, requests=5000),   # slow to build: use a bigger --time-limit
}

DEFAULT_SCENARIOS = ["tiny", "small", "medium", "large"]


def check_schedule(result: ScheduleResult, satellites: pd.DataFrame, requests: pd.DataFrame, opportunities: pd.DataFrame, slew_buffer_seconds: float,) -> list[str]:
    """
    Verify a schedule against every constraint. Returns a list of violations (empty = valid).

    Checks:
        - each request served at most once
        - start >= window_start and end <= window_end
        - on each satellite: next_start - previous_end >= slew_buffer
        - sum(energy) <= battery_capacity, sum(data) <= memory_capacity
        - captures per orbit block <= max_observations_per_orbit
    """
    if not result.scheduled:
        return []

    violations = []
    eps = 1e-6      # tolerance for float rounding

    schedule = pd.DataFrame([observation.model_dump() for observation in result.scheduled])
    schedule["start"] = pd.to_datetime(schedule["start"], utc=True)
    schedule["end"] = pd.to_datetime(schedule["end"], utc=True)

    sat = satellites.set_index("satellite_id")
    req = requests.set_index("request_id")
    opp = opportunities.set_index("opportunity_id")
    base_time = opportunities.loc[opportunities["feasible"], "window_start"].min()   # same time 0 as the schedulers

    if schedule["request_id"].duplicated().any():
        violations.append("request served more than once")

    # Window of each scheduled capture, aligned row by row with `schedule` (reset_index -> same 0..n-1 index).
    windows = opp.loc[schedule["opportunity_id"]].reset_index(drop=True)
    if (schedule["start"] < windows["window_start"]).any():
        violations.append("capture starts before its window")
    if (schedule["end"] > windows["window_end"]).any():
        violations.append("capture ends after its window")

    for satellite_id, group in schedule.groupby("satellite_id"):
        group = group.sort_values("start")

        # Gap between consecutive captures on this satellite.
        # Formula:
        #   gap_k = start_{k+1} - end_k   must be >= slew_buffer
        next_starts = group["start"].iloc[1:].reset_index(drop=True)
        previous_ends = group["end"].iloc[:-1].reset_index(drop=True)
        gaps = (next_starts - previous_ends).dt.total_seconds()
        if len(gaps) and gaps.min() < slew_buffer_seconds - eps:
            violations.append(f"{satellite_id}: gap {gaps.min():.1f}s < slew buffer")

        served = req.loc[group["request_id"]]
        if served["energy_wh"].sum() > sat.loc[satellite_id, "battery_capacity_wh"] + eps:
            violations.append(f"{satellite_id}: battery exceeded")
        if served["data_volume_gb"].sum() > sat.loc[satellite_id, "memory_capacity_gb"] + eps:
            violations.append(f"{satellite_id}: memory exceeded")

        # block_id = floor((window_start - base_time) / orbit_seconds)
        orbit_seconds = sat.loc[satellite_id, "orbit_period_minutes"] * 60
        offsets = (opp.loc[group["opportunity_id"], "window_start"] - base_time).dt.total_seconds()
        per_block = (offsets // orbit_seconds).value_counts()
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
    data_dir = data_root / scenario.name

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
    """Generate one dataset, run the 3 algorithms, return one result row per algorithm."""

    # ---- 1. Generate and save the dataset ----
    t0 = time.perf_counter()
    satellites, requests, opportunities, data_dir = generate_scenario(scenario, args.data_root)
    generate_seconds = time.perf_counter() - t0

    print(f"\n### {scenario.name}: {scenario.satellites} satellites, {scenario.requests} requests, {len(opportunities):,} opportunities (generated in {generate_seconds:.1f}s)")

    # ---- 2. Run the 3 algorithms ----
    results = run_all(
        satellites_path=data_dir / "satellites.parquet",
        requests_path=data_dir / "requests.parquet",
        opportunities_path=data_dir / "opportunities.parquet",
        time_limit=args.time_limit,
        workers=args.workers,
        slew_buffer_seconds=args.slew_buffer_seconds,
        use_greedy_hint=not args.no_hint,
    )

    # ---- 3. Check and summarize ----
    greedy_objective = results["greedy"].objective_value

    rows = []
    for method, result in results.items():
        violations = check_schedule(result, satellites, requests, opportunities, args.slew_buffer_seconds)

        rows.append({
            "scenario": scenario.name,
            "satellites": scenario.satellites,
            "requests": scenario.requests,
            "opportunities": len(opportunities),
            "method": method,
            "status": result.status,
            "objective_value": round(result.objective_value, 2),
            "scheduled_requests": len(result.scheduled),
            # Share of all requests that got served. Formula: served / total
            "served_ratio": round(len(result.scheduled) / scenario.requests, 3),
            "runtime_seconds": round(result.runtime_seconds, 2),
            # Formula: objective / objective_greedy - 1   (NaN if greedy = 0)
            "gain_vs_greedy_pct": round(100 * (result.objective_value / greedy_objective - 1), 2) if greedy_objective > 0 else math.nan,
            "valid": not violations,
            "violations": "; ".join(violations),
            "error": "",
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CP-SAT, greedy and improved greedy on several dataset sizes.")

    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=DEFAULT_SCENARIOS, help="Scenarios to run.")
    parser.add_argument("--time-limit", type=float, default=30.0, help="CP-SAT time limit per scenario (seconds).")
    parser.add_argument("--workers", type=int, default=8, help="CP-SAT threads.")
    parser.add_argument("--slew-buffer-seconds", type=float, default=30.0)
    parser.add_argument("--no-hint", action="store_true", help="Run CP-SAT without the greedy warm start.")
    parser.add_argument("--data-root", type=Path, default=Path("data/benchmark"), help="Where the generated datasets are written.")
    parser.add_argument("--output", type=Path, default=Path("output/benchmark_results.csv"))
    parser.add_argument("--generate-only", action="store_true", help="Only generate and save the datasets, do not run the algorithms.")

    args = parser.parse_args()

    # ---- Mode 1: only generate the datasets (no solving) ----
    if args.generate_only:
        for name in args.scenarios:
            t0 = time.perf_counter()
            satellites, requests, opportunities, data_dir = generate_scenario(SCENARIOS[name], args.data_root)

            print(f"{name:<7} {len(satellites):>3} satellites  {len(requests):>5} requests  {len(opportunities):>7,} opportunities  "
                  f"-> {data_dir}  ({time.perf_counter() - t0:.1f}s)")

        print("\nPreview one with: python -m src.preview_dataset --data-dir data/benchmark/<scenario>")
        return

    # ---- Mode 2: full benchmark (generate + solve + check) ----

    all_rows = []

    for name in args.scenarios:
        try:
            all_rows.extend(run_scenario(SCENARIOS[name], args))
        except Exception as error:      # One failing scenario must not stop the whole benchmark.
            traceback.print_exc()
            all_rows.append({"scenario": name, "method": "-", "error": f"{type(error).__name__}: {error}"})

    results = pd.DataFrame(all_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    # Short table on screen (the CSV keeps every column).
    columns = ["scenario", "opportunities", "method", "status", "objective_value", "scheduled_requests", "runtime_seconds", "gain_vs_greedy_pct", "valid", "error"]
    pd.set_option("display.width", 250)
    print("\n=== Benchmark results ===")
    print(results.reindex(columns=columns).to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()

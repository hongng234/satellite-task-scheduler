from __future__ import annotations

import argparse
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd


## Key point: This is a simulated data generator,
## NOT a real orbital model. It generates “observation windows” based on a simplified orbital period so that the scheduler has data to work with for optimization.

# weights = [0.35, 0.35, 0.22, 0.08]
# LOW:      35%
# MEDIUM:   35%
# HIGH:     22%
# CRITICAL:  8%

PRIORITY_WEIGHTS = {
    1: 0.35,
    2: 0.35,
    3: 0.22,
    4: 0.08,
}


# Value multiplier per priority: a CRITICAL request is worth ~8x a LOW one.
PRIORITY_VALUES_MULTIPLIER = {
    1: 1.0,
    2: 2.0,
    3: 4.0,
    4: 8.0,
}

# Fixed default start of the planning horizon.
# A fixed date (instead of "now") makes the dataset reproducible: same seed -> same files.
DEFAULT_START_TIME = "2026-01-01T00:00:00+00:00"

# Columns of the opportunities table. Used to build an empty DataFrame with the right columns
# when no opportunity is generated (otherwise the schedulers would crash on missing columns).
OPPORTUNITY_COLUMNS = ["opportunity_id", "request_id", "satellite_id", "window_start", "window_end", "duration_seconds", "incidence_angle_deg", "off_nadir_deg", "feasible",]


def generate_satellites(n_satellites: int, orbit_period_minutes: float, seed: int,) -> pd.DataFrame:
    rng = random.Random(seed)   # Separate random generator -> reproducible for a given seed.
    rows = []

    for i in range(n_satellites):
        rows.append({
            "satellite_id": f"SAT-{i + 1:02d}",                                         # Ex: SAT-01, SAT-02, ...
            "orbit_period_minutes": round(orbit_period_minutes + rng.uniform(-2, 2), 2),  # Ex: 96 ± 2 min
            "max_observations_per_orbit": rng.randint(6, 10),                            # 6..10 captures per orbit
            "battery_capacity_wh": round(rng.uniform(900, 1400), 1),                     # Energy budget for the whole horizon
            "memory_capacity_gb": round(rng.uniform(48, 96), 1),                         # Storage budget for the whole horizon
            "slew_rate_deg_per_sec": round(rng.uniform(0.8, 1.8), 2),                    # Not used by the schedulers yet (fixed slew buffer instead)
            "initial_phase_deg": round(rng.uniform(0, 360), 2),                          # Position on the orbit at start_time
        })

    return pd.DataFrame(rows)


# Create requests with random priorities, values, and time windows within the specified horizon.
def generate_requests(n_requests: int, horizon_hours: int, start_time: datetime, seed: int,) -> pd.DataFrame:

    # seed + 1 to ensure different random sequences for requests and satellites
    rng = random.Random(seed + 1)

    # horizon is the total planned duration. If horizon_hours = 48, then horizon = 48 hours.
    horizon = timedelta(hours = horizon_hours)

    rows = []

    priorities = list(PRIORITY_WEIGHTS.keys())
    weights = list(PRIORITY_WEIGHTS.values())

    for i in range(n_requests):
        # Randomly select one priority level based on weight.
        # rng.choices() returns a list, for example [3],
        # Access the first element using [0].
        priority = rng.choices(priorities, weights=weights, k=1)[0]

        value_multiplier = PRIORITY_VALUES_MULTIPLIER[priority]     # High-priority requests will have a higher value.

        # Generate a random time interval in seconds, within the first 70% of the horizon.
        # For example, horizon = 48 hours = 172,800 seconds.
        # earliest_offset_seconds can be any value from 0 to 120,960 seconds, or from 0 to 33.6 hours.
        # take only the top 70% coz to leave time remaining at the end for the request,
        # avoiding the request appearing too close to the end of the horizon, which would leave no valid observation window.
        earliest_offset_seconds = rng.uniform(0, horizon.total_seconds() * 0.70,)

        # The earliest time a request can be made.
        # Formula:
        #   earliest_start = start_time + random interval
        #
        # Example:
        #   start_time = 2026-11-01 00:00
        #   earliest_offset_seconds = 7,200 seconds = 2 hours
        #   earliest_start = 2026-11-01 02:00
        earliest_start = start_time + timedelta(seconds=earliest_offset_seconds)

        # The latest time by which the request must be completed.
        # There are two constraints:
        #   1. Must not exceed the end of the horizon:
        #        start_time + horizon
        #   2. Must not be later than earliest_start:
        #        earliest_start + 4 to 24 hours
        #
        # min() selects the earlier of the two constraints.
        # Example:
        #   start_time      = 2026-11-01 00:00
        #   horizon         = 48 hours
        #   earliest_start  = 2026-11-01 20:00
        #   earliest_start + 10 hours = 2026-11-02 06:00
        #   start_time + horizon   = 2026-11-03 00:00
        # Because 2026-11-02 06:00 is earlier than 2026-11-03 00:00,
        # So latest_end must be 2026-11-02 06:00.
        latest_end = min(start_time + horizon, earliest_start + timedelta(hours=rng.uniform(4, 24)),)

        rows.append({
            "request_id": f"REQ-{i + 1:04d}",                               # Ex: REQ-0001, REQ-0002, ..., REQ-0500
            "aoi_id": f"AOI-{rng.randint(1, 150):03d}",                     # Ex: AOI-001, AOI-002, ..., AOI-150. In real system, AOI-001 may be a specific geographic area, like a city or region.
            "priority": priority,
            "value": round(rng.uniform(10, 50) * value_multiplier, 2),      # The “objective” that the scheduler aims to maximize. $$$
            "earliest_start": earliest_start,                               # The earliest time the request can be made. Must be earlier than latest_end.
            "latest_end": latest_end,                                       # Deadline time for the request to be completed. Must be later than earliest_start.
            "duration_seconds": round(rng.uniform(5, 15), 1),               # The time required to take a single shot. EX: 5 to 15 seconds. This is the time the satellite needs to capture the image of the AOI.
            "data_volume_gb": round(rng.uniform(0.5, 3.0), 2),              # Data produced by one capture.
            "energy_wh": round(rng.uniform(8, 25), 1),                      # Energy used by one capture.
        })

    return pd.DataFrame(rows)

    ### Example of a generated request:
    ## start_time = 2026-11-01 00:00 UTC
    ## horizon_hours = 48
    ##
    ## 1 request can generated:
    ## earliest_offset_seconds = 7,200 = 2 hours
    ##      => earliest_start = 2026-11-01 02:00
    ## rng.uniform(4, 24) = 10
    ##      => earliest_start + 10 hours = 2026-11-01 12:00
    ## start_time + horizon (48 hours) = 2026-11-03 00:00
    ## latest_end = min(2026-11-01 12:00, 2026-11-03 00:00)
    ##            = 2026-11-01 12:00
    ##
    ## So this request can be made between 2026-11-01 02:00 and 2026-11-01 12:00, with a duration of 5 to 15 seconds.



# Generate opportunities for each satellite-request pair based on the satellite's orbit and the request's time window.
# orbit_period_minutes = 96 minutes, satellite return to the exact same position in orbit every 96 minutes. So the satellite can capture the same AOI every 96 minutes. 96 mins, 192 mins, 288 mins,...
# This is a simulated data generator. A real satellite does not always see the same point on each orbit, coz the Earth rotates and the orbit is inclined;
def generate_opportunities(satellites: pd.DataFrame, requests: pd.DataFrame, start_time: datetime, max_windows_per_pair: int, seed: int,) -> pd.DataFrame:
    rng = random.Random(seed + 2)

    # Each AOI gets a fixed position along the orbit, as a fraction of one orbit in [0, 1).
    # Ex: aoi_fraction = 0.25 -> the satellite flies over this AOI 1/4 of an orbit after passing its phase-0 point.
    # sorted() makes the order (and so the random draws) deterministic.
    aoi_rng = random.Random(seed + 3)
    aoi_fraction = {aoi_id: aoi_rng.random() for aoi_id in sorted(requests["aoi_id"].unique())}

    rows = []

    for request in requests.itertuples(index=False):
        for satellite in satellites.itertuples(index=False):
            # Not every satellite can capture every AOI

            # rng.random() returns a random number between 0.0 and 1.0.
            # If this number is greater than 0.55, skip this pair.
            # This means:
            #   There is a 55% chance the satellite can observe the request.
            #   There is a 45% chance there is no opportunity. -> skip this satellite-request pair.
            # Simplified way of looking at visibility/footprint.
            if rng.random() > 0.55:
                continue


            period_seconds = satellite.orbit_period_minutes * 60    # Convert 1 orbit period from minutes to seconds. For example, 96 minutes = 5,760 seconds.

            # Time (after start_time) of the satellite's first pass over this AOI.
            #
            # initial_phase_deg is the satellite's position on its orbit at start_time (in degrees);
            # aoi_fraction is where the AOI sits along the orbit.
            #
            # Formula:
            #   pass_offset = ((initial_phase_deg / 360) + aoi_fraction) * period   (mod period)
            #
            # Ex:
            #   initial_phase_deg = 90  -> 0.25 orbit
            #   aoi_fraction      = 0.5
            #   period            = 5,760 s
            #   pass_offset = (0.25 + 0.5) * 5,760 = 4,320 s = 72 min after start_time
            #
            # All passes of this satellite over this AOI:
            #   pass_k = start_time + pass_offset + k * period,   k = 0, 1, 2, ...
            #
            # The anchor is the COMMON start_time (not the request's earliest_start), so one satellite
            # flies over one AOI at the same times for every request of that AOI.
            pass_offset = (((satellite.initial_phase_deg / 360) + aoi_fraction[request.aoi_id]) % 1.0) * period_seconds

            # First pass at or after the request's earliest_start.
            # Formula:
            #   k = ceil((earliest_start - start_time - pass_offset) / period)
            #   first_pass = start_time + pass_offset + k * period
            # Ex:
            #   earliest_start - start_time = 10,000 s, pass_offset = 4,320 s, period = 5,760 s
            #   k = ceil(5,680 / 5,760) = 1  -> first_pass = start_time + 10,080 s
            seconds_since_start = (request.earliest_start - start_time).total_seconds()
            k = math.ceil((seconds_since_start - pass_offset) / period_seconds)
            first_pass = start_time + timedelta(seconds=pass_offset + k * period_seconds)

            # candidate_time is the current time for a potential observation window.
            # Initially, it is set to first_pass.
            # After each iteration, it increases by one orbital period.
            candidate_time = first_pass
            windows_created = 0     # Count the number of windows created for this request-satellite pair.


            # Condition 1:
            #   A potential window must begin before the request deadline.
            #   (latest_end <= start_time + horizon, so this also keeps windows inside the horizon.)
            # Condition 2:
            #   Each request-satellite pair creates at most max_windows_per_pair windows.
            ##  EX: if max_windows_per_pair = 3, this means a satellite may be proposed for a maximum of 3 observations of the same request.
            while candidate_time < request.latest_end and windows_created < max_windows_per_pair:

                # window_start is the start time of the observation window.
                # EX:
                #   candidate_time = 08:24:00
                #   window_start   = 08:24:00
                window_start = candidate_time

                # window_end is the end time of the observation window.
                # EX:
                #   request.duration_seconds = 10
                #   window_end = window_start + 30 seconds
                #
                # This means:
                #   If 10 seconds of data need to be captured, the scheduler has 30 seconds to choose the start time.
                #
                # This provides “flexibility”:
                # the scheduler is not required to capture exactly at 08:24:00, but can start at any time between 08:24:00 and 08:24:20.
                window_end = window_start + timedelta(seconds=request.duration_seconds * 3)

                # Create an opportunity only if the entire window ends before the request's deadline.
                # EX:
                #   request.latest_end = 08:30
                #   window_end         = 08:24:30
                #   -> Valid.
                #  But if:
                #   request.latest_end = 08:24:10
                #   window_end         = 08:24:30
                #   -> Invalid, skip.
                if window_end <= request.latest_end:
                    rows.append({
                        "opportunity_id": f"OPP-{len(rows) + 1:06d}",           # Deterministic id (was uuid4, random on every run). Ex: OPP-000001
                        "request_id": request.request_id,
                        "satellite_id": satellite.satellite_id,
                        "window_start": window_start,
                        "window_end": window_end,
                        "duration_seconds": request.duration_seconds,
                        "incidence_angle_deg": round(rng.uniform(20, 45), 2),   # Informational only (not used by the schedulers).
                        "off_nadir_deg": round(rng.uniform(5, 35), 2),          # Informational only (not used by the schedulers).
                        "feasible": True,
                    })

                    windows_created += 1

                # After evaluating the current time, skip to the satellite's next pass.
                # EX:
                #   candidate_time = 08:24
                #   period_seconds = 96 minutes
                #   new_candidate_time = 10:00
                candidate_time += timedelta(seconds=period_seconds)

    # columns=... keeps the expected columns even when rows is empty.
    return pd.DataFrame(rows, columns=OPPORTUNITY_COLUMNS)


def generate_all(n_satellites: int, n_requests: int, horizon_hours: int, orbit_period_minutes: float, max_windows_per_pair: int, seed: int, start_time: datetime,) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate the 3 tables in memory: (satellites, requests, opportunities).
    Used by main() and by benchmark.py.
    """
    satellites = generate_satellites(n_satellites=n_satellites, orbit_period_minutes=orbit_period_minutes, seed=seed,)
    requests = generate_requests(n_requests=n_requests, horizon_hours=horizon_hours, start_time=start_time, seed=seed,)
    opportunities = generate_opportunities(satellites=satellites, requests=requests, start_time=start_time, max_windows_per_pair=max_windows_per_pair, seed=seed,)

    return satellites, requests, opportunities


def save_dataset(satellites: pd.DataFrame, requests: pd.DataFrame, opportunities: pd.DataFrame, output_dir: Path,) -> None:
    """Write the 3 tables as Parquet files into output_dir (created if missing)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    satellites.to_parquet(output_dir / "satellites.parquet", index=False,)
    requests.to_parquet(output_dir / "requests.parquet", index=False,)
    opportunities.to_parquet(output_dir / "opportunities.parquet", index=False,)


def parse_start_time(text: str) -> datetime:
    """
    Parse an ISO-8601 date-time and make it UTC.
    Ex: "2026-01-01T00:00:00+00:00" or "2026-01-01" (assumed UTC).
    """
    value = datetime.fromisoformat(text)

    if value.tzinfo is None:                    # No timezone given -> assume UTC.
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simulated satellite dataset.")

    parser.add_argument("--satellites", type=int, default=12, help="Number of satellites to generate.")
    parser.add_argument("--requests", type=int, default=500, help="Number of requests to generate.")
    parser.add_argument("--horizon_hours", type=int, default=48, help="Planning horizon in hours.")
    parser.add_argument("--orbit_period_minutes", type=float, default=96.0, help="Average orbit period in minutes.")
    parser.add_argument("--max_windows_per_pair", type=int, default=3, help="Maximum visibility windows per satellite-request pair.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--start_time", type=str, default=DEFAULT_START_TIME, help="Start of the planning horizon (ISO-8601, UTC). Fixed by default for reproducibility.")
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"), help="Output directory for generated parquet files.")

    args = parser.parse_args()

    start_time = parse_start_time(args.start_time)

    satellites, requests, opportunities = generate_all(
        n_satellites=args.satellites,
        n_requests=args.requests,
        horizon_hours=args.horizon_hours,
        orbit_period_minutes=args.orbit_period_minutes,
        max_windows_per_pair=args.max_windows_per_pair,
        seed=args.seed,
        start_time=start_time,
    )

    save_dataset(satellites=satellites, requests=requests, opportunities=opportunities, output_dir=args.output_dir,)

    print("Dataset generated successfully")
    print(f"Start time: {start_time.isoformat()}")
    print(f"Satellites: {len(satellites)}")
    print(f"Requests: {len(requests)}")
    print(f"Opportunities: {len(opportunities)}")
    print(f"Output directory: {args.output_dir.resolve()}") # Print the absolute path of the output directory.

    print("\nFiles:")
    print(f"- {args.output_dir / 'satellites.parquet'}")
    print(f"- {args.output_dir / 'requests.parquet'}")
    print(f"- {args.output_dir / 'opportunities.parquet'}")


if __name__ == "__main__":
    main()

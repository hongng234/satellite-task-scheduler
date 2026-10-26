from __future__ import annotations

import argparse
import random
import uuid
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


PRIORITY_VALUES_MULTIPLIER = {
    1: 1.0,
    2: 2.0,
    3: 4.0,
    4: 8.0,
}


def generate_satellites(n_satellites: int, orbit_period_minutes: float, seed: int,) -> pd.DataFrame: 
    rng = random.Random(seed)
    rows = []

    for i in range(n_satellites):
        rows.append({
            "satellite_id": f"SAT-{i + 1:02d}",
            "orbit_period_minutes": round(orbit_period_minutes + rng.uniform(-2, 2), 2),
            "max_observations_per_orbit": rng.randint(6, 10),
            "battery_capacity_wh": round(rng.uniform(900, 1400), 1),
            "memory_capacity_gb": round(rng.uniform(48, 96), 1),
            "slew_rate_deg_per_sec": round(rng.uniform(0.8, 1.8), 2),
            "initial_phase_deg": round(rng.uniform(0, 360), 2),
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
            "data_volume_gb": round(rng.uniform(0.5, 3.0), 2),
            "energy_wh": round(rng.uniform(8, 25), 1),
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
def generate_opportunities(satellites: pd.DataFrame, requests: pd.DataFrame, horizon_hours: int, max_windows_per_pair: int, seed: int,) -> pd.DataFrame:
    rng = random.Random(seed + 2)
    horizon = timedelta(hours=horizon_hours)

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

            # initial_phase_deg is the satellite's starting position
            # in orbit, measured in degrees.
            # If the satellite starts at 90/360 of the orbit -> it is at 25% of the orbital cycle.
            # Example:
            #   initial_phase_deg = 90
            #   period_seconds = 5,760
            #   phase_offset = (90 / 360) * 5,760
            #                = 0.25 * 5,760
            #                = 1,440 seconds
            #                = 24 minutes
            phase_offset = (satellite.initial_phase_deg / 360) * period_seconds

            # The time of the “first” pass when the satellite can approach the AOI.
            # Example:
            #   request.earliest_start = 08:00
            #   phase_offset = 24 minutes
            #   first_pass = 08:24
            # Meaning: assuming from the time the request is submitted,
            # the satellite needs an additional 24 minutes to reach its first observation position.
            first_pass = request.earliest_start + timedelta(seconds=phase_offset % period_seconds)

            # candidate_time is the current time for a potential observation window.
            # Initially, it is set to first_pass.
            # After each iteration, it increases by one orbital period.
            candidate_time = first_pass
            windows_created = 0     # Count the number of windows created for this request-satellite pair.


            # Condition 1:
            #   A potential window must begin before the request deadline.
            # Condition 2:
            #   A potential window must not exceed earliest_start + horizon.
            ##  This is a safety limit to prevent the creation of windows that fall outside the planning horizon.
            # Condition 3:
            #   Each request-satellite pair creates at most max_windows_per_pair windows.
            ##  EX: if max_windows_per_pair = 3, this means a satellite may be proposed for a maximum of 3 observations of the same request.
            while (candidate_time < request.latest_end and candidate_time < request.earliest_start + horizon and windows_created < max_windows_per_pair):

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
                        "opportunity_id": str(uuid.uuid4()),
                        "request_id": request.request_id,
                        "satellite_id": satellite.satellite_id,
                        "window_start": window_start,
                        "window_end": window_end,
                        "duration_seconds": request.duration_seconds,
                        "incidence_angle_deg": round(rng.uniform(20, 45), 2),
                        "off_nadir_deg": round(rng.uniform(5, 35), 2),
                        "feasible": True,
                    })

                    windows_created += 1

                # After evaluating the current time, skip to the satellite's next pass.
                # EX:
                #   candidate_time = 08:24
                #   period_seconds = 96 minutes
                #   new_candidate_time = 10:00
                candidate_time += timedelta(seconds=period_seconds)
    
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simulated satellite dataset.")

    parser.add_argument("--satellites", type=int, default=12, help="Number of satellites to generate.")
    parser.add_argument("--requests", type=int, default=500, help="Number of requests to generate.")
    parser.add_argument("--horizon_hours", type=int, default=48, help="Planning horizon in hours.")
    parser.add_argument("--orbit_period_minutes", type=float, default=96.0, help="Average orbit period in minutes.")
    parser.add_argument("--max_windows_per_pair", type=int, default=3, help="Maximum visibility windows per satellite-request pair.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output_dir", type=str, default=Path("data/processed"), help="Output directory for generated parquet files.")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(timezone.utc).replace(microsecond=0, second=0, minute=0)  # Round down to the nearest hour.

    satellites = generate_satellites(n_satellites=args.satellites, orbit_period_minutes=args.orbit_period_minutes, seed=args.seed,)
    requests = generate_requests(n_requests=args.requests, horizon_hours=args.horizon_hours, start_time=start_time, seed=args.seed,)
    opportunities = generate_opportunities(satellites=satellites, requests=requests, horizon_hours=args.horizon_hours, max_windows_per_pair=args.max_windows_per_pair, seed=args.seed,)

    satellites.to_parquet(args.output_dir / "satellites.parquet", index=False,)
    requests.to_parquet(args.output_dir / "requests.parquet", index=False,)
    opportunities.to_parquet(args.output_dir / "opportunities.parquet", index=False,)

    print("Dataset generated successfully")
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

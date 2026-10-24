from __future__ import annotations

import argparse
import random
import uuid
from datetime import datetime, timedelta, timezone 
from pathlib import Path
import pandas as pd


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


def generate_requests(n_requests: int, horizon_hours: int, start_time: datetime, seed: int,) -> pd.DataFrame:
    rng = random.Random(seed + 1)
    horizon = timedelta(hours = horizon_hours)

    rows = []

    priority = list(PRIORITY_WEIGHTS.keys())
    weights = list(PRIORITY_WEIGHTS.values())

    for i in range(n_requests):
        priority = rng.choices(priorities, weights=weights, k=1)[0]

        value_multiplier = PRIORITY_VALUES_MULTIPLIER[priority]

        earliest_offset_seconds = rng.uniform(0, horizon.total_seconds() * 0.70,)

        earliest_start = start_time + timedelta(seconds=earliest_offset_seconds)

        latest_end = min(start_time + horizon, earliest_start + timedelta(hours=rng.uniform(4, 24)),)

        rows.append({
            "request_id": f"REQ-{i + 1:04d}",
            "aoi_id": f"AOI-{rng.randint(1, 150):03d}",
            "priority": priority,
            "value": round(rng.uniform(10, 50) * value_multiplier, 2),
            "earliest_start": earliest_start,
            "latest_end": latest_end,
            "duration_seconds": round(rng.uniform(5, 15), 1),
            "data_volume_gb": round(rng.uniform(0.5, 3.0), 2),
            "energy_wh": round(rng.uniform(8, 25), 1),
        })
        
    return pd.DataFrame(rows)

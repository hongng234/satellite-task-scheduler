"""
Shared pytest fixtures and helpers.

How pytest uses this file
    - pytest loads `conftest.py` by itself before it runs the tests in this folder.
    - A function decorated with `@pytest.fixture` here is available to every test. A test asks for it
      by naming it as a parameter. pytest sees the parameter name, runs the fixture
      and passes in what it returns.

      Ex:
          def test_something(small_dataset):     # pytest calls small_dataset() and passes the result
              ...

Two kinds of test data:
    1. `small_dataset`: a small random dataset from the real generator (~250 opportunities).
    2. `write_dataset(...)`: hand-made datasets of 1 to 3 opportunities. The expected answer can be
       worked out by hand, so each test can check one behaviour.
"""
from __future__ import annotations     # Lets type hints use names defined later in the file.

from dataclasses import dataclass       # Short way to declare a class that only stores values.
from pathlib import Path                # File paths that work on Windows and Linux.

import pandas as pd                     # Builds the hand-made tables.
import pytest                           # Test framework: @pytest.fixture is defined here.

from src.generate_dataset import DEFAULT_START_TIME, generate_all, parse_start_time, save_dataset  # The real data generator.

# Time 0 used by every hand-made dataset. All windows are written as "seconds after T0".
T0 = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")


def at(seconds: float) -> pd.Timestamp:
    """
    The timestamp `seconds` after T0. With it, a hand-made window reads as "from second 0 to second 30".

    Formula:
        at(s) = T0 + s seconds
    Ex:
        at(90) -> 2026-01-01 00:01:30 UTC
    """
    return T0 + pd.Timedelta(seconds=seconds)


# @dataclass generates __init__ from the annotated fields below,
# so Dataset(a, b, c, d, e, f) stores the 6 values and does nothing else.
@dataclass
class Dataset:
    """Paths of the 3 Parquet files, plus the DataFrames themselves for assertions."""
    satellites_path: Path               # <folder>/satellites.parquet
    requests_path: Path                 # <folder>/requests.parquet
    opportunities_path: Path            # <folder>/opportunities.parquet
    satellites: pd.DataFrame            # The data that is in the files, kept in memory so tests can compare against it.
    requests: pd.DataFrame
    opportunities: pd.DataFrame

    # A @property is read like an attribute: ds.paths, without parentheses.
    @property
    def paths(self) -> dict:
        """
        Keyword arguments accepted by every scheduler constructor.

        Ex:
            GreedyScheduler(**ds.paths)
            # ** unpacks the dict, same as:
            GreedyScheduler(satellites_path=..., requests_path=..., opportunities_path=...)
        """
        return dict(satellites_path=self.satellites_path, requests_path=self.requests_path, opportunities_path=self.opportunities_path,)


def make_satellite(satellite_id: str = "SAT-01", **overrides) -> dict:
    """
    One satellite row with capacities large enough that they never limit a hand-made test.
    A test then shrinks the one capacity it wants to check.

    **overrides lets a test change one field and keep the defaults for the rest.
    Ex:
        make_satellite(battery_capacity_wh=15.0)   # all defaults, except a very small battery
    """
    row = {
        "satellite_id": satellite_id,
        "orbit_period_minutes": 96.0,           # One orbit = 5,760 s.
        "max_observations_per_orbit": 10,       # A hand-made test has 1 to 3 captures, so this never limits it.
        "battery_capacity_wh": 1000.0,          # = 100 captures of 10 Wh.
        "memory_capacity_gb": 100.0,            # = 100 captures of 1 GB.
        "slew_rate_deg_per_sec": 1.0,           # Not used by the schedulers.
        "initial_phase_deg": 0.0,               # Not used by the schedulers.
    }
    row.update(overrides)       # Values passed by the test replace the defaults.
    return row


def make_request(request_id: str, **overrides) -> dict:
    """
    One request row that is open for the whole first day: [T0, T0 + 1 day].
    Default: priority 2, value 100, 10 s capture, 1 GB, 10 Wh.
    Ex:
        make_request("R1", value=50.0, priority=4)
    """
    row = {
        "request_id": request_id,
        "aoi_id": "AOI-001",                        # Area to photograph. The schedulers do not read it.
        "priority": 2,                              # MEDIUM.
        "value": 100.0,                             # Reward if served.
        "earliest_start": T0,                       # Request open from T0...
        "latest_end": T0 + pd.Timedelta(days=1),    # ...until T0 + 1 day, so every hand-made window fits inside.
        "duration_seconds": 10.0,                   # 10 s capture.
        "data_volume_gb": 1.0,                      # 1 GB per capture.
        "energy_wh": 10.0,                          # 10 Wh per capture.
    }
    row.update(overrides)       # Values passed by the test replace the defaults.
    return row


def make_opportunity(opportunity_id: str, request_id: str, start_s: float, end_s: float, satellite_id: str = "SAT-01", duration: float = 10.0,) -> dict:
    """
    One opportunity row with window [T0 + start_s, T0 + end_s].

    Ex:
        make_opportunity("O1", "R1", 0, 30)
        -> satellite SAT-01 can capture request R1 between second 0 and second 30 (10 s capture).
    """
    return {
        "opportunity_id": opportunity_id,
        "request_id": request_id,
        "satellite_id": satellite_id,
        "window_start": at(start_s),             # T0 + start_s seconds.
        "window_end": at(end_s),                 # T0 + end_s seconds.
        "duration_seconds": duration,            # Capture length. It should match the request's duration.
        "incidence_angle_deg": 30.0,             # The schedulers do not read this column.
        "off_nadir_deg": 10.0,                   # Nor this one.
        "feasible": True,                        # The opportunity can be used.
    }


@pytest.fixture
def write_dataset(tmp_path):
    """
    A factory fixture: what it returns is a function, and that function writes hand-made rows to Parquet.

    `tmp_path` is a fixture built into pytest. It gives each test its own empty temporary folder
    and removes it later, so the tests never write to the real data/ folder.

    Usage in a test:
        def test_x(write_dataset):
            ds = write_dataset([make_satellite()], [make_request("R1")], [make_opportunity("O1", "R1", 0, 30)])
            GreedyScheduler(**ds.paths).solve()
    """
    # The inner function is what the test receives and calls.
    def _write(satellites: list[dict], requests: list[dict], opportunities: list[dict]) -> Dataset:
        # Each list of dicts becomes a DataFrame, one dict per row.
        sat, req, opp = pd.DataFrame(satellites), pd.DataFrame(requests), pd.DataFrame(opportunities)

        # save_dataset is the writer the real generator uses, so the files have the format the schedulers expect.
        save_dataset(satellites=sat, requests=req, opportunities=opp, output_dir=tmp_path)

        return Dataset(tmp_path / "satellites.parquet", tmp_path / "requests.parquet", tmp_path / "opportunities.parquet", sat, req, opp)

    return _write


# With scope="session" the fixture runs once for the whole test run, and every test that asks for it
# gets the same result. The default, scope="function", would run it again for each test.
# Sharing is safe here because the tests only read this dataset.
@pytest.fixture(scope="session")
def small_dataset(tmp_path_factory) -> Dataset:
    """
    A small dataset from the real generator: 3 satellites, 50 requests (~250 opportunities).
    It has the size of the "tiny" benchmark scenario.
    """
    # tmp_path_factory does what tmp_path does, and also works in a session-scoped fixture.
    folder = tmp_path_factory.mktemp("small_dataset")

    # The seed is fixed, so every test run gets the same dataset and the same results.
    satellites, requests, opportunities = generate_all(
        n_satellites=3, n_requests=50, horizon_hours=48, orbit_period_minutes=96.0,
        max_windows_per_pair=3, seed=42, start_time=parse_start_time(DEFAULT_START_TIME),
    )
    save_dataset(satellites=satellites, requests=requests, opportunities=opportunities, output_dir=folder)

    return Dataset(folder / "satellites.parquet", folder / "requests.parquet", folder / "opportunities.parquet", satellites, requests, opportunities)

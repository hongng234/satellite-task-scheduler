"""
Tests for the simulated dataset generator (src/generate_dataset.py).

How a pytest test works
    - Any function whose name starts with `test_` is a test. pytest finds it and runs it.
    - `assert <expression>` checks that the expression is True.
        * True  -> nothing happens and the test goes on.
        * False -> the test fails and pytest prints the values involved.
      Any condition works, e.g. `assert a == b`, `assert x > 0`, `assert not df.empty`,
      `assert (column >= 0).all()`.
    - A test passes if it reaches the end with no failed assert and no unexpected exception.
"""
from __future__ import annotations     # Lets type hints use names defined later in the file.

from datetime import UTC                # The UTC timezone, to check that parsed dates are in UTC.

import pandas as pd                     # pd.testing.assert_frame_equal and DataFrame operations.

from src.generate_dataset import OPPORTUNITY_COLUMNS, generate_all, generate_opportunities, parse_start_time   # Code under test.

# Fixed start of the horizon for every test in this file.
START = parse_start_time("2026-01-01T00:00:00+00:00")


def _generate(seed: int = 42):
    """
    A small dataset: 4 satellites and 60 requests. Returns (satellites, requests, opportunities).
    pytest does not run this as a test, because its name does not start with test_.
    """
    return generate_all(n_satellites=4, n_requests=60, horizon_hours=48, orbit_period_minutes=96.0, max_windows_per_pair=3, seed=seed, start_time=START,)


def test_same_seed_gives_identical_dataset():
    # Reproducibility: the same seed and start_time must give the same 3 tables, ids included.
    #
    # zip pairs the tables of the two runs:
    #   (satellites_run1, satellites_run2), (requests_run1, requests_run2), (opportunities_run1, opportunities_run2)
    for first, second in zip(_generate(seed=7), _generate(seed=7)):
        # This pandas helper raises an AssertionError, which fails the test, if any value, column or dtype differs.
        pd.testing.assert_frame_equal(first, second)


def test_different_seed_gives_different_dataset():
    # Checks that the seed is used at all: another seed must give other data.
    # `_` is the usual name for a value that is not needed. Here that is satellites and opportunities.
    _, requests_a, _ = _generate(seed=1)
    _, requests_b, _ = _generate(seed=2)

    # .equals() is True when the two tables are identical, and here they must differ.
    assert not requests_a.equals(requests_b)


def test_opportunity_ids_are_sequential_and_unique():
    # Ids must be OPP-000001, OPP-000002, ... A random id such as uuid4 would change on every run.
    _, _, opportunities = _generate()

    # The expected list has one id per row, numbered from 1.
    # f"{i:06d}" pads the integer with zeros to 6 digits. Ex: 7 -> "000007"
    expected = [f"OPP-{i:06d}" for i in range(1, len(opportunities) + 1)]

    # Compare the whole column with the expected list.
    assert opportunities["opportunity_id"].tolist() == expected


def test_windows_are_inside_request_time_range():
    # Every window must satisfy:
    #   earliest_start <= window_start < window_end <= latest_end
    _, requests, opportunities = _generate()

    # Join each opportunity with its request, like a SQL JOIN on request_id.
    # The window times and the request times are then on the same row.
    merged = opportunities.merge(requests, on="request_id", suffixes=("", "_req"))

    # (column_a >= column_b) gives a True/False column, one value per row.
    # .all() is True only if every row is True.
    assert (merged["window_start"] >= merged["earliest_start"]).all()
    assert (merged["window_end"] > merged["window_start"]).all()
    assert (merged["window_end"] <= merged["latest_end"]).all()


def test_window_is_long_enough_for_the_capture():
    # The generator makes each window 3 x the capture duration, so a capture always fits.
    #
    # Formula:
    #   window_length = window_end - window_start  >=  duration
    _, _, opportunities = _generate()

    # Subtracting two datetime columns gives durations. .dt.total_seconds() turns them into seconds.
    length = (opportunities["window_end"] - opportunities["window_start"]).dt.total_seconds()
    assert (length >= opportunities["duration_seconds"]).all()


def test_passes_of_one_satellite_over_one_aoi_are_one_period_apart():
    # Two consecutive windows of one (request, satellite) pair must be one orbit period apart.
    #
    # Formula:
    #   window_start_{k+1} - window_start_k = orbit_period_minutes * 60
    satellites, _, opportunities = _generate()

    # A Series: satellite_id -> orbit period in seconds.
    period = satellites.set_index("satellite_id")["orbit_period_minutes"] * 60

    # groupby gives one (request_id, satellite_id) pair at a time, with its rows.
    for (_, satellite_id), group in opportunities.groupby(["request_id", "satellite_id"]):
        # Sort the starts. diff() is then the difference with the previous row.
        # The first row has no previous row, so it is NaN and dropna() removes it.
        gaps = group["window_start"].sort_values().diff().dropna().dt.total_seconds()

        # Every gap must equal the period, with 0.001 s allowed for float rounding.
        assert ((gaps - period[satellite_id]).abs() < 1e-3).all()


def test_no_opportunity_still_has_all_columns():
    # Edge case: with no satellites there are no opportunities, but the columns must still exist.
    # The schedulers read opportunities["feasible"], for example, and would crash on a missing column.
    _, requests, _ = _generate()

    # The satellites table has 0 rows, so the generator has nothing to loop over.
    empty = generate_opportunities(satellites=pd.DataFrame(columns=["satellite_id"]), requests=requests, start_time=START, max_windows_per_pair=3, seed=1)

    assert empty.empty                                  # 0 rows.
    assert list(empty.columns) == OPPORTUNITY_COLUMNS   # All expected columns, in order.


def test_parse_start_time_defaults_to_utc():
    # A date written without a timezone must be read as UTC.
    value = parse_start_time("2026-01-01")

    assert value.tzinfo == UTC              # The timezone is set, and it is UTC.
    assert value.hour == 0                  # Midnight. A local timezone would have shifted the hour.

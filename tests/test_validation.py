"""
Tests for the independent constraint checker (src/benchmark.py::check_schedule)
and end-to-end validity of all 3 schedulers.

WHY TEST THE CHECKER ITSELF?
    check_schedule() returns a list of violations; an empty list means "valid".
    A broken checker that always returns [] would make every schedule look valid.
    So we also feed it deliberately WRONG schedules and assert that it complains.
"""
from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from src.benchmark import SCENARIOS, check_schedule, generate_scenario
from src.models import Opportunity, Priority, ScheduledObservation, ScheduleResult
from src.scheduler_cp_sat import SatelliteTaskingScheduler
from src.scheduler_greedy import GreedyScheduler, ImprovedGreedyScheduler
from tests.conftest import at

# Same slew buffer as the schedulers' default.
SLEW_BUFFER = 30.0


# Runs 3 times, once per algorithm. Each value is a small function (lambda) that builds a scheduler
# from a dataset. ids=[...] gives readable names in the pytest output, e.g.
#   test_every_scheduler_produces_a_valid_schedule[cp_sat]
@pytest.mark.parametrize("make_scheduler", [
    lambda ds: SatelliteTaskingScheduler(**ds.paths, time_limit_seconds=10, num_workers=4),
    lambda ds: GreedyScheduler(**ds.paths),
    lambda ds: ImprovedGreedyScheduler(**ds.paths),
], ids=["cp_sat", "greedy", "greedy_improved"])
def test_every_scheduler_produces_a_valid_schedule(small_dataset, make_scheduler):
    # End-to-end: build the scheduler, solve, and re-check the schedule against all constraints.
    result = make_scheduler(small_dataset).solve()
    ds = small_dataset

    # A non-empty list is truthy -> passes. The text after the comma is shown if the assert fails.
    assert result.scheduled, "expected a non-empty schedule"

    # No violation at all: the checker must return an empty list.
    assert check_schedule(result, ds.satellites, ds.requests, ds.opportunities, SLEW_BUFFER) == []


def _observation(opportunity_row, start, duration):
    """Helper: build one ScheduledObservation for an opportunity row, with a chosen start time and duration."""
    return ScheduledObservation(
        request_id=opportunity_row["request_id"],
        satellite_id=opportunity_row["satellite_id"],
        opportunity_id=opportunity_row["opportunity_id"],
        start=start,
        end=start + duration,
        value=1.0,
        priority=Priority.LOW,
    )


def test_checker_detects_duplicate_request(small_dataset):
    # Build a WRONG schedule on purpose: the same capture listed twice
    # -> the same request is served twice.
    row = small_dataset.opportunities.iloc[0]                       # first opportunity of the dataset
    duration = row["window_end"] - row["window_start"]              # window length (= 3 x capture duration)
    observation = _observation(row, row["window_start"], duration / 3)

    bad = ScheduleResult(status="FEASIBLE", objective_value=2.0, runtime_seconds=0.0, scheduled=[observation, observation], solver_status="TEST")
    violations = check_schedule(bad, small_dataset.satellites, small_dataset.requests, small_dataset.opportunities, SLEW_BUFFER)

    # any(...) -> True if at least one violation message contains "more than once".
    assert any("more than once" in v for v in violations)


def test_checker_detects_capture_outside_window(small_dataset):
    # WRONG schedule on purpose: the capture starts 1 hour BEFORE its window.
    row = small_dataset.opportunities.iloc[0]
    one_hour = at(3600) - at(0)         # a Timedelta of 3600 s
    ten_seconds = at(10) - at(0)        # a Timedelta of 10 s
    observation = _observation(row, row["window_start"] - one_hour, ten_seconds)

    bad = ScheduleResult(status="FEASIBLE", objective_value=1.0, runtime_seconds=0.0, scheduled=[observation], solver_status="TEST")
    violations = check_schedule(bad, small_dataset.satellites, small_dataset.requests, small_dataset.opportunities, SLEW_BUFFER)

    assert any("before its window" in v for v in violations)


def test_generate_scenario_writes_the_three_files(tmp_path):
    # `make datasets` / `python -m src.benchmark --generate-only` relies on generate_scenario().
    # Generate the "tiny" scenario into a temporary folder (tmp_path) instead of data/benchmark/.
    satellites, requests, opportunities, data_dir = generate_scenario(SCENARIOS["tiny"], tmp_path)

    # Files go to <data_root>/<scenario name>/.
    assert data_dir == tmp_path / "tiny"

    # Each Parquet file exists and has as many rows as the DataFrame returned in memory.
    for name, frame in [("satellites", satellites), ("requests", requests), ("opportunities", opportunities)]:
        path = data_dir / f"{name}.parquet"
        assert path.exists()
        assert len(pd.read_parquet(path)) == len(frame)

    # The scenario sizes are respected (tiny = 3 satellites, 50 requests).
    assert len(satellites) == 3
    assert len(requests) == 50


def test_opportunity_model_rejects_inverted_window():
    # Pydantic validator in models.py: window_end must be after window_start.
    # Here end (50 s) < start (100 s), so creating the object must raise ValidationError.
    # If no error is raised, pytest.raises makes the test FAIL.
    with pytest.raises(ValidationError):
        Opportunity(
            opportunity_id="O1", request_id="R1", satellite_id="SAT-01",
            window_start=at(100), window_end=at(50), duration_seconds=10,
            incidence_angle_deg=30, off_nadir_deg=10,
        )

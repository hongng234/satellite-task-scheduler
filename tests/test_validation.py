"""
Tests for the independent constraint checker (src/benchmark.py::check_schedule)
and end-to-end validity of all 3 schedulers.

Why the checker has tests of its own
    check_schedule() returns a list of violations, and an empty list means "valid".
    A broken checker that always returned [] would make every schedule look valid.
    So some tests give it schedules that are wrong on purpose and assert that it reports them.
"""
from __future__ import annotations     # Lets type hints use names defined later in the file.

import pandas as pd                     # Reads the Parquet files written by generate_scenario.
import pytest                           # Test framework: parametrize, raises.
from pydantic import ValidationError    # Error raised by a Pydantic model when a field is invalid.

from src.benchmark import SCENARIOS, check_schedule, generate_scenario                 # Checker and scenario generator under test.
from src.models import Opportunity, Priority, ScheduledObservation, ScheduleResult     # For building schedules that are wrong on purpose.
from src.scheduler_cp_sat import SatelliteTaskingScheduler                             # The 3 schedulers checked end-to-end.
from src.scheduler_greedy import GreedyScheduler, ImprovedGreedyScheduler
from tests.conftest import at, make_opportunity, make_request, make_satellite          # at(s) = T0 + s seconds, and hand-made rows.

# The schedulers' default slew buffer.
SLEW_BUFFER = 30.0


# Runs 3 times, once per algorithm. Each value is a small function (a lambda) that builds a scheduler
# from a dataset. ids=[...] gives the runs readable names in the pytest output, e.g.
#   test_every_scheduler_produces_a_valid_schedule[cp_sat]
@pytest.mark.parametrize("make_scheduler", [
    lambda ds: SatelliteTaskingScheduler(**ds.paths, time_limit_seconds=10, num_workers=4),
    lambda ds: GreedyScheduler(**ds.paths),
    lambda ds: ImprovedGreedyScheduler(**ds.paths),
], ids=["cp_sat", "greedy", "greedy_improved"])
def test_every_scheduler_produces_a_valid_schedule(small_dataset, make_scheduler):
    # End to end: build the scheduler, solve, and check the schedule against all constraints.
    result = make_scheduler(small_dataset).solve()
    ds = small_dataset

    # A non-empty list counts as True, so this passes when there is at least one capture.
    # pytest shows the text after the comma if the assert fails.
    assert result.scheduled, "expected a non-empty schedule"

    # The checker must find nothing wrong, so it returns an empty list.
    assert check_schedule(result, ds.satellites, ds.requests, ds.opportunities, SLEW_BUFFER) == []


def _observation(opportunity_row, start, duration):
    """Build one ScheduledObservation for an opportunity row, with the start time and duration the test chooses."""
    return ScheduledObservation(
        request_id=opportunity_row["request_id"],           # Ids copied from the opportunity row.
        satellite_id=opportunity_row["satellite_id"],
        opportunity_id=opportunity_row["opportunity_id"],
        start=start,                                        # Chosen by the test, sometimes wrong on purpose.
        end=start + duration,                               # Formula: end = start + duration
        value=1.0,                                          # The checker does not look at value or priority.
        priority=Priority.LOW,
    )


def test_checker_detects_duplicate_request(small_dataset):
    # A schedule that is wrong on purpose: the same capture is listed twice,
    # so the same request is served twice.
    row = small_dataset.opportunities.iloc[0]                       # First opportunity of the dataset.
    duration = row["window_end"] - row["window_start"]              # Window length, which is 3 x the capture duration.
    observation = _observation(row, row["window_start"], duration / 3)

    bad = ScheduleResult(status="FEASIBLE", objective_value=2.0, runtime_seconds=0.0, scheduled=[observation, observation], solver_status="TEST")
    violations = check_schedule(bad, small_dataset.satellites, small_dataset.requests, small_dataset.opportunities, SLEW_BUFFER)

    # any(...) is True if at least one violation message contains "more than once".
    assert any("more than once" in v for v in violations)


def test_checker_detects_capture_outside_window(small_dataset):
    # A schedule that is wrong on purpose: the capture starts 1 hour before its window opens.
    row = small_dataset.opportunities.iloc[0]
    one_hour = at(3600) - at(0)         # A Timedelta of 3600 s.
    ten_seconds = at(10) - at(0)        # A Timedelta of 10 s.
    observation = _observation(row, row["window_start"] - one_hour, ten_seconds)

    bad = ScheduleResult(status="FEASIBLE", objective_value=1.0, runtime_seconds=0.0, scheduled=[observation], solver_status="TEST")
    violations = check_schedule(bad, small_dataset.satellites, small_dataset.requests, small_dataset.opportunities, SLEW_BUFFER)

    assert any("before its window" in v for v in violations)


def test_generate_scenario_writes_the_three_files(tmp_path):
    # `make datasets` and `python -m src.benchmark --generate-only` both call generate_scenario().
    # The "tiny" scenario is written to a temporary folder (tmp_path), so data/benchmark/ stays untouched.
    satellites, requests, opportunities, data_dir = generate_scenario(SCENARIOS["tiny"], tmp_path)

    # The files go to <data_root>/<scenario name>/.
    assert data_dir == tmp_path / "tiny"

    # Each Parquet file exists and has as many rows as the DataFrame that was returned.
    for name, frame in [("satellites", satellites), ("requests", requests), ("opportunities", opportunities)]:
        path = data_dir / f"{name}.parquet"
        assert path.exists()
        assert len(pd.read_parquet(path)) == len(frame)

    # The dataset has the size the scenario asks for: tiny is 3 satellites and 50 requests.
    assert len(satellites) == 3
    assert len(requests) == 50


def test_opportunity_model_rejects_inverted_window():
    # The Pydantic validator in models.py requires window_end to be after window_start.
    # Here the end (50 s) is before the start (100 s), so creating the object must raise ValidationError.
    # If nothing is raised, pytest.raises fails the test.
    with pytest.raises(ValidationError):
        Opportunity(
            opportunity_id="O1", request_id="R1", satellite_id="SAT-01",
            window_start=at(100), window_end=at(50), duration_seconds=10,
            incidence_angle_deg=30, off_nadir_deg=10,
        )


# ---------------------------------------------------------------------------
# Checker on hand-made data: one injected mistake per case
#
# One satellite, two requests worth 100 each (make_request defaults), 10 s captures:
#   O1 -> R1, window [0, 30]
#   O2 -> R2, window [100, 130]
# The correct capture for O1 is [0, 10]. Each case changes one thing and names the message it expects.
# ---------------------------------------------------------------------------

def _hand_made_tables(**request_overrides):
    """The 3 tables above as DataFrames. request_overrides changes request R1. Ex: earliest_start=at(1000)"""
    satellites = pd.DataFrame([make_satellite()])
    requests = pd.DataFrame([make_request("R1", **request_overrides), make_request("R2")])
    opportunities = pd.DataFrame([make_opportunity("O1", "R1", 0, 30), make_opportunity("O2", "R2", 100, 130)])
    return satellites, requests, opportunities


def _check(observation_fields: dict, objective: float | None = None, **request_overrides) -> list[str]:
    """Run the checker on a one-capture schedule. observation_fields changes the correct capture of O1."""
    fields = dict(request_id="R1", satellite_id="SAT-01", opportunity_id="O1", start=at(0), end=at(10), value=100.0, priority=Priority.MEDIUM)
    fields.update(observation_fields)
    observation = ScheduledObservation(**fields)
    total = observation.value if objective is None else objective
    result = ScheduleResult(status="FEASIBLE", objective_value=total, runtime_seconds=0.0, scheduled=[observation], solver_status="TEST")
    return check_schedule(result, *_hand_made_tables(**request_overrides), SLEW_BUFFER)


def test_checker_accepts_the_correct_hand_made_capture():
    # Without this, the cases below could pass because the checker rejects everything.
    assert _check({}) == []


# Each tuple: (what the case changes, the text the violation must contain).
@pytest.mark.parametrize("fields, expected", [
    (dict(end=at(0)), "shorter than its duration"),                    # 0 s capture
    (dict(end=at(2)), "shorter than its duration"),                    # 2 s capture for a 10 s request
    (dict(request_id="R2"), "request_id does not match"),              # O1 belongs to R1
    (dict(satellite_id="SAT-02"), "satellite_id does not match"),      # O1 belongs to SAT-01
    (dict(value=9999.0), "value does not match"),                      # R1 is worth 100
    (dict(priority=Priority.CRITICAL), "priority does not match"),     # R1 is MEDIUM
    (dict(opportunity_id="O-MISSING"), "unknown opportunity ids"),     # used to crash with KeyError
], ids=["zero_length", "too_short", "wrong_request", "wrong_satellite", "inflated_value", "wrong_priority", "unknown_opportunity"])
def test_checker_detects_injected_mistake(fields, expected):
    assert any(expected in v for v in _check(fields))


def test_checker_detects_wrong_objective_value():
    # The capture is fine, but the result claims a total of 500 for one capture worth 100.
    assert any("objective_value" in v for v in _check({}, objective=500.0))


def test_checker_detects_window_outside_request_range():
    # R1 now opens at second 1000, but the window of O1 is [0, 30].
    assert any("outside its request" in v for v in _check({}, earliest_start=at(1000)))


def test_checker_detects_infeasible_opportunity():
    # O1 is marked feasible = False, so no scheduler may use it.
    satellites, requests, opportunities = _hand_made_tables()
    opportunities.loc[opportunities["opportunity_id"] == "O1", "feasible"] = False
    observation = ScheduledObservation(request_id="R1", satellite_id="SAT-01", opportunity_id="O1", start=at(0), end=at(10), value=100.0, priority=Priority.MEDIUM)
    result = ScheduleResult(status="FEASIBLE", objective_value=100.0, runtime_seconds=0.0, scheduled=[observation], solver_status="TEST")
    assert any("feasible = False" in v for v in check_schedule(result, satellites, requests, opportunities, SLEW_BUFFER))


# ---------------------------------------------------------------------------
# Input checks (validate_tables), run by every scheduler constructor
# ---------------------------------------------------------------------------

SCHEDULER_CLASSES = [SatelliteTaskingScheduler, GreedyScheduler, ImprovedGreedyScheduler]


@pytest.mark.parametrize("scheduler_class", SCHEDULER_CLASSES)
def test_negative_slew_buffer_is_rejected(write_dataset, scheduler_class):
    # A negative buffer would let two captures overlap. It used to be accepted without a word.
    ds = write_dataset([make_satellite()], [make_request("R1")], [make_opportunity("O1", "R1", 0, 30)])
    with pytest.raises(ValueError, match="slew_buffer_seconds"):
        scheduler_class(**ds.paths, slew_buffer_seconds=-5)


@pytest.mark.parametrize("scheduler_class", SCHEDULER_CLASSES)
def test_zero_duration_request_is_rejected(write_dataset, scheduler_class):
    # Greedy divides value by duration, so a 0 s request used to crash it with ZeroDivisionError.
    ds = write_dataset([make_satellite()], [make_request("R1", duration_seconds=0.0)], [make_opportunity("O1", "R1", 0, 30, duration=0.0)])
    with pytest.raises(ValueError, match="duration_seconds <= 0"):
        scheduler_class(**ds.paths)


@pytest.mark.parametrize("scheduler_class", SCHEDULER_CLASSES)
def test_opportunity_duration_must_match_request(write_dataset, scheduler_class):
    # The request needs 10 s but its opportunity says 20 s. The tables disagree, so they are rejected.
    ds = write_dataset([make_satellite()], [make_request("R1", duration_seconds=10.0)], [make_opportunity("O1", "R1", 0, 30, duration=20.0)])
    with pytest.raises(ValueError, match="differs from their request"):
        scheduler_class(**ds.paths)


@pytest.mark.parametrize("scheduler_class", SCHEDULER_CLASSES)
@pytest.mark.parametrize("bad_id, message", [("request", "unknown requests"), ("satellite", "unknown satellites")])
def test_unknown_ids_are_rejected_with_a_clear_message(write_dataset, scheduler_class, bad_id, message):
    # CP-SAT used to say "unknow requests" and "Oppotunities". The messages are now shared by all schedulers.
    request_id = "R-MISSING" if bad_id == "request" else "R1"
    satellite_id = "SAT-MISSING" if bad_id == "satellite" else "SAT-01"
    ds = write_dataset([make_satellite()], [make_request("R1")], [make_opportunity("O1", request_id, 0, 30, satellite_id=satellite_id)])
    with pytest.raises(ValueError, match=message):
        scheduler_class(**ds.paths)

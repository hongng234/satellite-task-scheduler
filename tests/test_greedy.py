"""
Tests for GreedyScheduler (baseline) and ImprovedGreedyScheduler (src/scheduler_greedy.py).

Two levels of tests:
    1. UNIT tests: call one small method (`_find_start`) directly with hand-made inputs.
       Fast and precise: if one fails, we know exactly which rule is broken.
    2. Behaviour tests: run the full `solve()` on a tiny hand-made dataset and check the result.
"""
from __future__ import annotations     # Lets type hints use names defined later in the file.

from collections import namedtuple      # Tiny read-only record type (see Opp below).

import pytest                           # Test framework: parametrize, raises.

from src.scheduler_greedy import GreedyScheduler, ImprovedGreedyScheduler         # Code under test.
from tests.conftest import at, make_opportunity, make_request, make_satellite     # Helpers for hand-made datasets.

# namedtuple = a tiny read-only class with named fields.
# Opp(window_start=..., window_end=..., duration_seconds=...) behaves like a row of the opportunities
# table for the only 3 fields `_find_start` reads, so we don't need a full DataFrame.
Opp = namedtuple("Opp", ["window_start", "window_end", "duration_seconds"])

# NOTE on `scheduler_class._find_start(None, ...)` below:
#   _find_start is a normal method, so it expects `self` as first argument.
#   It never uses `self`, so we pass None instead of building a whole scheduler (which needs files).


# ---------------------------------------------------------------------------
# Unit tests of _find_start (the only difference between the two greedy variants)
#
# Notation in comments: [a, b] = time interval in seconds after T0.
# A busy interval = an already scheduled capture plus its slew buffer.
# ---------------------------------------------------------------------------

# @pytest.mark.parametrize runs the SAME test several times with different values.
# Here the test runs twice: once with scheduler_class = GreedyScheduler,
# once with scheduler_class = ImprovedGreedyScheduler. pytest reports them as 2 separate tests.
@pytest.mark.parametrize("scheduler_class", [GreedyScheduler, ImprovedGreedyScheduler])
def test_free_satellite_starts_at_window_start(scheduler_class):
    # Window [100, 130], capture 10 s, nothing busy -> both variants start at 100.
    opportunity = Opp(at(100), at(130), 10)
    assert scheduler_class._find_start(None, busy_intervals=[], opportunity=opportunity) == at(100)


def test_baseline_skips_when_window_start_is_busy():
    # Window [100, 130], capture 10 s -> baseline only tries [100, 110].
    # Busy [90, 105] overlaps it -> baseline gives up and returns None ("skip this opportunity").
    opportunity = Opp(at(100), at(130), 10)
    assert GreedyScheduler._find_start(None, busy_intervals=[(at(90), at(105))], opportunity=opportunity) is None


def test_improved_shifts_to_end_of_busy_interval():
    # Same situation, improved variant: the satellite is free from 105,
    # capture [105, 115] still ends before 130 -> start = 105.
    opportunity = Opp(at(100), at(130), 10)
    assert ImprovedGreedyScheduler._find_start(None, busy_intervals=[(at(90), at(105))], opportunity=opportunity) == at(105)


def test_improved_jumps_over_several_busy_intervals():
    # Busy [90, 105] and [108, 112]:
    #   - free gap [105, 108] is only 3 s, too short for a 10 s capture -> skip it
    #   - free from 112: capture [112, 122] fits in the window -> start = 112
    opportunity = Opp(at(100), at(130), 10)
    busy = [(at(90), at(105)), (at(108), at(112))]
    assert ImprovedGreedyScheduler._find_start(None, busy_intervals=busy, opportunity=opportunity) == at(112)


def test_improved_returns_none_when_no_slot_fits():
    # Busy [90, 125]. Latest possible start = window_end - duration = 130 - 10 = 120.
    # Earliest free time 125 > 120 -> no slot -> None.
    opportunity = Opp(at(100), at(130), 10)
    assert ImprovedGreedyScheduler._find_start(None, busy_intervals=[(at(90), at(125))], opportunity=opportunity) is None


# ---------------------------------------------------------------------------
# Behaviour of the full solve() on hand-made datasets
#
# `write_dataset` is a fixture from conftest.py: pytest passes it in automatically
# because the test function has a parameter with that name.
# ---------------------------------------------------------------------------

def test_improved_greedy_serves_more_than_baseline_when_shifting_helps(write_dataset):
    # Scenario (1 satellite, slew buffer 30 s by default):
    #   R1: priority 4 (CRITICAL) -> handled first, window [0, 30]  -> captured at [0, 10]
    #       -> satellite busy until 10 + 30 s buffer = 40
    #   R2: priority 1 (LOW), window [20, 80]
    #       baseline: tries 20 -> busy -> skipped              => 1 request served
    #       improved: shifts to 40 -> [40, 50] fits in window  => 2 requests served
    ds = write_dataset(
        [make_satellite()],
        [make_request("R1", priority=4), make_request("R2", priority=1)],
        [make_opportunity("O1", "R1", 0, 30), make_opportunity("O2", "R2", 20, 80)],
    )

    # len(result.scheduled) = number of captures in the schedule.
    assert len(GreedyScheduler(**ds.paths).solve().scheduled) == 1
    assert len(ImprovedGreedyScheduler(**ds.paths).solve().scheduled) == 2


def test_request_is_served_at_most_once(write_dataset):
    # Constraint "one capture per request": two satellites can both serve R1, only one capture is kept.
    ds = write_dataset(
        [make_satellite("SAT-01"), make_satellite("SAT-02")],
        [make_request("R1")],
        [make_opportunity("O1", "R1", 0, 30, satellite_id="SAT-01"), make_opportunity("O2", "R1", 0, 30, satellite_id="SAT-02")],
    )
    result = GreedyScheduler(**ds.paths).solve()

    # List of served request ids must be exactly ["R1"] (not ["R1", "R1"]).
    assert [o.request_id for o in result.scheduled] == ["R1"]


def test_battery_limit_is_respected(write_dataset):
    # Battery constraint: capacity 15 Wh, each capture costs 10 Wh (default in make_request).
    #   1 capture  -> 10 Wh <= 15 OK
    #   2 captures -> 20 Wh >  15 not allowed
    # The windows are far apart (0 and 1000 s), so time is not the reason; only the battery limits.
    ds = write_dataset(
        [make_satellite(battery_capacity_wh=15.0)],
        [make_request("R1"), make_request("R2")],
        [make_opportunity("O1", "R1", 0, 30), make_opportunity("O2", "R2", 1000, 1030)],
    )
    assert len(GreedyScheduler(**ds.paths).solve().scheduled) == 1


def test_unknown_request_id_raises(write_dataset):
    # Bad input: the opportunity points to "R-MISSING", which is not in the requests table.
    ds = write_dataset([make_satellite()], [make_request("R1")], [make_opportunity("O1", "R-MISSING", 0, 30)])

    # pytest.raises: the test PASSES only if the code inside the `with` block raises ValueError
    # and its message contains "unknown requests". If no error is raised, the test FAILS.
    with pytest.raises(ValueError, match="unknown requests"):
        GreedyScheduler(**ds.paths)


def test_from_dataframes_matches_file_constructor(small_dataset):
    # Two ways to build the scheduler must give the same result:
    #   - from Parquet files (normal use)
    #   - from DataFrames already in memory (used by CP-SAT to compute its hint)
    from_files = ImprovedGreedyScheduler(**small_dataset.paths).solve()
    from_frames = ImprovedGreedyScheduler.from_dataframes(small_dataset.satellites, small_dataset.requests, small_dataset.opportunities).solve()

    assert from_files.objective_value == from_frames.objective_value

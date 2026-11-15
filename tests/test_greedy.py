"""
Tests for GreedyScheduler (baseline) and ImprovedGreedyScheduler (src/scheduler_greedy.py).

Two levels of tests:
    1. Unit tests call one small method (`_find_start`) directly with hand-made inputs.
       When one fails, it points at the rule that is broken.
    2. Behaviour tests run the full `solve()` on a very small hand-made dataset and check the result.
"""
from __future__ import annotations     # Lets type hints use names defined later in the file.

from collections import namedtuple      # A small read-only record type, used for Opp below.

import pytest                           # Test framework: parametrize, raises.

from src.scheduler_greedy import GreedyScheduler, ImprovedGreedyScheduler         # Code under test.
from tests.conftest import at, make_opportunity, make_request, make_satellite     # Helpers for hand-made datasets.

# A namedtuple is a small read-only class with named fields.
# `_find_start` reads 3 fields of an opportunity row. Opp has those 3 fields,
# so the tests can pass an Opp and do without a DataFrame.
Opp = namedtuple("Opp", ["window_start", "window_end", "duration_seconds"])

# About `scheduler_class._find_start(None, ...)` below:
#   _find_start is a normal method, so its first argument is `self`.
#   It never uses `self`, so the tests pass None. Building a real scheduler would need files.


# ---------------------------------------------------------------------------
# Unit tests of _find_start, the one method that differs between the two greedy variants
#
# In the comments, [a, b] is a time interval in seconds after T0.
# A busy interval is a capture that is already scheduled, plus its slew buffer.
# ---------------------------------------------------------------------------

# @pytest.mark.parametrize runs one test several times with different values.
# This test runs twice: once with scheduler_class = GreedyScheduler and
# once with scheduler_class = ImprovedGreedyScheduler. pytest reports them as 2 separate tests.
@pytest.mark.parametrize("scheduler_class", [GreedyScheduler, ImprovedGreedyScheduler])
def test_free_satellite_starts_at_window_start(scheduler_class):
    # Window [100, 130], capture 10 s, nothing busy -> both variants start at 100.
    opportunity = Opp(at(100), at(130), 10)
    assert scheduler_class._find_start(None, busy_intervals=[], opportunity=opportunity) == at(100)


def test_baseline_skips_when_window_start_is_busy():
    # Window [100, 130], capture 10 s. The baseline tries [100, 110] and nothing else.
    # Busy [90, 105] overlaps it, so the baseline returns None and the opportunity is skipped.
    opportunity = Opp(at(100), at(130), 10)
    assert GreedyScheduler._find_start(None, busy_intervals=[(at(90), at(105))], opportunity=opportunity) is None


def test_improved_shifts_to_end_of_busy_interval():
    # Same situation with the improved variant. The satellite is free from 105,
    # and capture [105, 115] still ends before 130, so start = 105.
    opportunity = Opp(at(100), at(130), 10)
    assert ImprovedGreedyScheduler._find_start(None, busy_intervals=[(at(90), at(105))], opportunity=opportunity) == at(105)


def test_improved_jumps_over_several_busy_intervals():
    # Busy [90, 105] and [108, 112]:
    #   - the free gap [105, 108] is 3 s long, too short for a 10 s capture
    #   - from 112 the satellite is free, and capture [112, 122] fits in the window -> start = 112
    opportunity = Opp(at(100), at(130), 10)
    busy = [(at(90), at(105)), (at(108), at(112))]
    assert ImprovedGreedyScheduler._find_start(None, busy_intervals=busy, opportunity=opportunity) == at(112)


def test_improved_returns_none_when_no_slot_fits():
    # Busy [90, 125]. The latest possible start is window_end - duration = 130 - 10 = 120.
    # The satellite is free from 125, which is after 120, so there is no slot and the result is None.
    opportunity = Opp(at(100), at(130), 10)
    assert ImprovedGreedyScheduler._find_start(None, busy_intervals=[(at(90), at(125))], opportunity=opportunity) is None


# ---------------------------------------------------------------------------
# Behaviour of the full solve() on hand-made datasets
#
# `write_dataset` is a fixture from conftest.py. pytest passes it in
# because the test function has a parameter with that name.
# ---------------------------------------------------------------------------

def test_improved_greedy_serves_more_than_baseline_when_shifting_helps(write_dataset):
    # One satellite and the default 30 s slew buffer:
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

    # len(result.scheduled) is the number of captures in the schedule.
    assert len(GreedyScheduler(**ds.paths).solve().scheduled) == 1
    assert len(ImprovedGreedyScheduler(**ds.paths).solve().scheduled) == 2


def test_request_is_served_at_most_once(write_dataset):
    # One capture per request: two satellites can both serve R1, and only one capture is kept.
    ds = write_dataset(
        [make_satellite("SAT-01"), make_satellite("SAT-02")],
        [make_request("R1")],
        [make_opportunity("O1", "R1", 0, 30, satellite_id="SAT-01"), make_opportunity("O2", "R1", 0, 30, satellite_id="SAT-02")],
    )
    result = GreedyScheduler(**ds.paths).solve()

    # The served request ids must be ["R1"]. A scheduler that served it twice would give ["R1", "R1"].
    assert [o.request_id for o in result.scheduled] == ["R1"]


def test_battery_limit_is_respected(write_dataset):
    # Battery: the capacity is 15 Wh and each capture costs 10 Wh (the default in make_request).
    #   1 capture  -> 10 Wh <= 15 OK
    #   2 captures -> 20 Wh >  15 not allowed
    # The windows are 1000 s apart, so the captures cannot collide in time. The battery is the only limit.
    ds = write_dataset(
        [make_satellite(battery_capacity_wh=15.0)],
        [make_request("R1"), make_request("R2")],
        [make_opportunity("O1", "R1", 0, 30), make_opportunity("O2", "R2", 1000, 1030)],
    )
    assert len(GreedyScheduler(**ds.paths).solve().scheduled) == 1


def test_unknown_request_id_raises(write_dataset):
    # Bad input: the opportunity points to "R-MISSING", which is not in the requests table.
    ds = write_dataset([make_satellite()], [make_request("R1")], [make_opportunity("O1", "R-MISSING", 0, 30)])

    # With pytest.raises, the test passes only if the code in the `with` block raises ValueError
    # with a message that contains "unknown requests". If nothing is raised, the test fails.
    with pytest.raises(ValueError, match="unknown requests"):
        GreedyScheduler(**ds.paths)


def test_from_dataframes_matches_file_constructor(small_dataset):
    # The two ways to build the scheduler must give the same result:
    #   - from Parquet files, the normal use
    #   - from DataFrames already in memory, which CP-SAT uses to compute its hint
    from_files = ImprovedGreedyScheduler(**small_dataset.paths).solve()
    from_frames = ImprovedGreedyScheduler.from_dataframes(small_dataset.satellites, small_dataset.requests, small_dataset.opportunities).solve()

    assert from_files.objective_value == from_frames.objective_value

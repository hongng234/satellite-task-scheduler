"""
Tests for the CP-SAT scheduler (src/scheduler_cp_sat.py).

REGRESSION TESTS
    A regression test reproduces a bug that was already fixed, with the smallest possible input.
    If someone re-introduces the bug later, the test fails immediately.
    Bugs covered here:
        - mandatory (non-optional) intervals made the model infeasible,
        - an opportunity with a too-short window caused a KeyError,
        - the slew buffer was ignored.
"""
from __future__ import annotations     # Lets type hints use names defined later in the file.

import pytest                           # Test framework: parametrize.

from src.scheduler_cp_sat import SatelliteTaskingScheduler                 # Code under test.
from src.scheduler_greedy import GreedyScheduler, ImprovedGreedyScheduler  # Used as a reference: CP-SAT must never be worse.
from tests.conftest import make_opportunity, make_request, make_satellite  # Helpers for hand-made datasets.


def solve(ds, **kwargs):
    """
    Helper: run CP-SAT on dataset `ds` with test-friendly settings
    (10 s time limit, 4 threads). Extra keyword arguments override them.
    Ex:
        solve(ds, slew_buffer_seconds=0)
    """
    options = dict(time_limit_seconds=10, num_workers=4)       # Default test settings.
    options.update(kwargs)                                      # Keyword arguments given by the test replace the defaults.
    return SatelliteTaskingScheduler(**ds.paths, **options).solve()


def test_overlapping_windows_on_one_satellite_pick_one(write_dataset):
    # Regression (mandatory intervals).
    # Two requests with the SAME window [0, 15] on one satellite: both captures cannot happen.
    #   Correct model -> OPTIMAL, exactly one selected: R1, because its value is higher (100 > 50).
    #   Old bug       -> every interval was forced into NoOverlap, even unselected ones -> INFEASIBLE.
    ds = write_dataset(
        [make_satellite()],
        [make_request("R1", value=100.0), make_request("R2", value=50.0)],
        [make_opportunity("O1", "R1", 0, 15), make_opportunity("O2", "R2", 0, 15)],
    )
    result = solve(ds)

    assert result.status == "OPTIMAL"                                 # solver proved the best answer
    assert [o.request_id for o in result.scheduled] == ["R1"]        # and that answer is "only R1"


def test_too_short_window_is_skipped_without_crash(write_dataset):
    # Regression (KeyError).
    # O1: window [0, 5] = 5 s, but the capture needs 10 s -> impossible -> removed from the model.
    # The old code still looked up O1's variable later and crashed with KeyError.
    # Expected now: no crash, and only O2 is scheduled.
    ds = write_dataset(
        [make_satellite()],
        [make_request("R1"), make_request("R2")],
        [make_opportunity("O1", "R1", 0, 5), make_opportunity("O2", "R2", 100, 130)],
    )
    result = solve(ds)

    assert result.status == "OPTIMAL"
    assert [o.opportunity_id for o in result.scheduled] == ["O2"]


# Runs twice: (buffer=0, expected=2) and (buffer=30, expected=1).
@pytest.mark.parametrize("buffer, expected", [(0, 2), (30, 1)])
def test_slew_buffer_is_enforced(write_dataset, buffer, expected):
    # Regression (slew buffer ignored).
    # Two 10 s captures with windows exactly 10 s long, so their times are fixed:
    #   O1 = [0, 10], O2 = [15, 25]  -> gap between them = 15 - 10 = 5 s.
    #   buffer = 0  -> gap 5 >= 0  -> both fit      -> 2 captures
    #   buffer = 30 -> gap 5 <  30 -> only one fits -> 1 capture
    ds = write_dataset(
        [make_satellite()],
        [make_request("R1"), make_request("R2")],
        [make_opportunity("O1", "R1", 0, 10), make_opportunity("O2", "R2", 15, 25)],
    )
    assert len(solve(ds, slew_buffer_seconds=buffer).scheduled) == expected


def test_capture_stays_inside_its_window(small_dataset):
    # On the realistic small dataset, every scheduled capture must satisfy:
    #   window_start <= start   and   end <= window_end
    # (checks the ceil/floor rounding of times in the CP-SAT model).
    result = solve(small_dataset)

    # Table indexed by opportunity_id -> windows.loc["OPP-000001"] gives that row.
    windows = small_dataset.opportunities.set_index("opportunity_id")

    for observation in result.scheduled:
        window = windows.loc[observation.opportunity_id]
        assert observation.start >= window["window_start"]
        assert observation.end <= window["window_end"]


def test_small_dataset_is_solved_to_optimality(small_dataset):
    # ~250 opportunities: CP-SAT should PROVE the optimum well within the 10 s limit.
    assert solve(small_dataset).status == "OPTIMAL"


# Runs twice: with the greedy hint and without it.
@pytest.mark.parametrize("use_hint", [True, False])
def test_cp_sat_is_never_worse_than_greedy(small_dataset, use_hint):
    # CP-SAT reaches the optimum on this small dataset, so its total value must be
    # >= the value of both greedy variants.
    cp_sat = solve(small_dataset, use_greedy_hint=use_hint).objective_value
    greedy = GreedyScheduler(**small_dataset.paths).solve().objective_value
    improved = ImprovedGreedyScheduler(**small_dataset.paths).solve().objective_value

    # Formula:
    #   cp_sat >= max(greedy, improved) - 0.000001
    # The tiny tolerance absorbs float rounding when summing values like 123.45.
    assert cp_sat >= max(greedy, improved) - 1e-6


def test_empty_opportunities_returns_empty_schedule(write_dataset):
    # Edge case: no opportunity at all -> empty schedule, no crash.
    # (Without the guard, base_time = min() of an empty column = NaT and the time conversion fails.)
    ds = write_dataset([make_satellite()], [make_request("R1")], [make_opportunity("O1", "R1", 0, 30)])

    # Overwrite the opportunities file with 0 rows (iloc[0:0] keeps the columns but no data).
    ds.opportunities.iloc[0:0].to_parquet(ds.opportunities_path, index=False)

    result = solve(ds)
    assert result.scheduled == []           # empty list of captures
    assert result.objective_value == 0      # total value 0

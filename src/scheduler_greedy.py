from __future__ import annotations

import bisect
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.models import (Priority, ScheduleResult, ScheduledObservation,)


class GreedyScheduler:
    """
    Baseline greedy heuristic.

    Idea:
        1. Sort all opportunities from "best" to "worst" (priority first, then value per second).
        2. Walk through the sorted list once and accept an opportunity if it does not break any
           constraint, given the opportunities already accepted.
        3. Never go back to change an earlier decision.

    Baseline rule for the start time:
        The capture always starts exactly at window_start.
        If the satellite is busy at that moment, the opportunity is skipped
        (even if a later start inside the window would have been free).
    """

    # Name used in reports (run_experiment.py, benchmark.py).
    METHOD_NAME = "greedy"

    def __init__(self, satellites_path: Path, requests_path: Path, opportunities_path: Path, slew_buffer_seconds: float = 30.0,) -> None:
        # Read the 3 Parquet files and hand them to the shared setup.
        self._setup(
            satellites=pd.read_parquet(satellites_path),
            requests=pd.read_parquet(requests_path),
            opportunities=pd.read_parquet(opportunities_path),
            slew_buffer_seconds=slew_buffer_seconds,
        )

    @classmethod
    def from_dataframes(cls, satellites: pd.DataFrame, requests: pd.DataFrame, opportunities: pd.DataFrame, slew_buffer_seconds: float = 30.0,) -> "GreedyScheduler":
        """
        Build the scheduler from DataFrames already in memory (no file reading).
        Used by the CP-SAT scheduler to compute a warm-start hint.
        """
        scheduler = cls.__new__(cls)    # Create the object without calling __init__ (which reads files).
        scheduler._setup(satellites=satellites, requests=requests, opportunities=opportunities, slew_buffer_seconds=slew_buffer_seconds,)
        return scheduler

    def _setup(self, satellites: pd.DataFrame, requests: pd.DataFrame, opportunities: pd.DataFrame, slew_buffer_seconds: float,) -> None:
        self.satellites = satellites
        self.requests = requests

        # Keep only usable opportunities (feasible = True). .copy() avoids modifying the caller's DataFrame.
        self.opportunities = opportunities[opportunities["feasible"]].copy()

        # Dictionaries for O(1) lookup:
        #   satellite_id -> {"battery_capacity_wh": ..., ...}
        #   request_id   -> {"value": ..., "energy_wh": ..., ...}
        self.satellite_lookup = self.satellites.set_index("satellite_id").to_dict("index")
        self.request_lookup = self.requests.set_index("request_id").to_dict("index")

        # Minimum idle time required between two captures on the same satellite (to re-orient).
        self.slew_buffer_seconds = slew_buffer_seconds

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Every opportunity must reference an existing request and satellite (same check as CP-SAT)."""
        # set(A) - set(B): ids used in opportunities but missing from the master tables.
        unknown_requests = set(self.opportunities["request_id"]) - set(self.requests["request_id"])
        unknown_satellites = set(self.opportunities["satellite_id"]) - set(self.satellites["satellite_id"])

        if unknown_requests:
            raise ValueError(f"Opportunities refer to unknown requests: {unknown_requests}")

        if unknown_satellites:
            raise ValueError(f"Opportunities refer to unknown satellites: {unknown_satellites}")

    def solve(self) -> ScheduleResult:
        started = time.perf_counter()   # Start the runtime clock.

        # ---- Step 1: score and sort opportunities -------------------------------------------
        #
        # Sort key (ascending, so "best" comes first):
        #   1. -priority          -> CRITICAL (4) before LOW (1)
        #   2. -value_density     -> higher value per second of capture first
        #   3. window_start       -> earlier windows first (leaves room for later ones)
        #   4. opportunity_id     -> deterministic tie-break
        #
        # Formula:
        #   value_density = value / duration_seconds
        # Ex:
        #   value = 120, duration = 10 s  -> density = 12 per second
        scored = []

        for opportunity in self.opportunities.itertuples(index=False):
            request = self.request_lookup[opportunity.request_id]

            value_density = request["value"] / request["duration_seconds"]

            sort_key = (-int(request["priority"]), -value_density, opportunity.window_start, opportunity.opportunity_id,)

            scored.append((sort_key, opportunity))

        # Sort only by the key. The opportunity namedtuple itself is never compared.
        scored.sort(key=lambda item: item[0])

        # ---- Step 2: state of each satellite while we accept opportunities ------------------

        # busy_intervals[satellite_id] = list of (busy_start, busy_end), kept sorted by busy_start.
        # Each accepted capture blocks [start - buffer, end + buffer], so the next capture must
        # stay at least `buffer` seconds away on both sides.
        busy_intervals: dict[str, list[tuple]] = defaultdict(list)

        used_energy: dict[str, float] = defaultdict(float)          # Wh consumed so far per satellite.
        used_memory: dict[str, float] = defaultdict(float)          # GB stored so far per satellite.
        orbit_counts: dict[str, dict[int, int]] = defaultdict(dict) # orbit_counts[sat][block_id] = captures in that orbit.

        # Time 0 of the problem. Computed once (it was recomputed inside the loop before).
        base_time = self.opportunities["window_start"].min()

        scheduled = []              # Accepted observations.
        served_requests = set()     # request_ids already served (a request is served at most once).

        # ---- Step 3: walk through the sorted list and accept what fits ---------------------
        for _, opportunity in scored:
            request = self.request_lookup[opportunity.request_id]
            satellite = self.satellite_lookup[opportunity.satellite_id]

            # Constraint: a request may be served at most once.
            if opportunity.request_id in served_requests:
                continue

            # Constraint: the window lies inside the request's time range and is long enough.
            if not self._fits_time_window(opportunity=opportunity, request=request):
                continue

            # Constraint: battery.
            # Formula:
            #   used_energy + energy_i <= battery_capacity
            if used_energy[opportunity.satellite_id] + request["energy_wh"] > satellite["battery_capacity_wh"]:
                continue

            # Constraint: memory.
            # Formula:
            #   used_memory + data_volume_i <= memory_capacity
            if used_memory[opportunity.satellite_id] + request["data_volume_gb"] > satellite["memory_capacity_gb"]:
                continue

            # Constraint: max observations per orbit.
            # Formula:
            #   block_id = floor((window_start - base_time) / orbit_seconds)
            #   count[block_id] + 1 <= max_observations_per_orbit
            orbit_block = self._orbit_block(timestamp=opportunity.window_start, base_time=base_time, orbit_period_minutes=satellite["orbit_period_minutes"],)

            current_count = orbit_counts[opportunity.satellite_id].get(orbit_block, 0)

            if current_count >= satellite["max_observations_per_orbit"]:
                continue

            # Constraint: no overlap with captures already on this satellite (including slew buffer).
            # The start-time rule differs between the baseline and the improved greedy.
            start = self._find_start(busy_intervals=busy_intervals[opportunity.satellite_id], opportunity=opportunity,)

            if start is None:       # No free start time -> skip this opportunity.
                continue

            # The capture lasts exactly duration_seconds.
            # Formula:
            #   end = start + duration
            end = start + timedelta(seconds=opportunity.duration_seconds)

            # ---- Accept the opportunity and update the state ----
            scheduled.append(
                ScheduledObservation(
                    request_id=opportunity.request_id,
                    satellite_id=opportunity.satellite_id,
                    opportunity_id=opportunity.opportunity_id,
                    start=start,
                    end=end,
                    value=request["value"],
                    priority=Priority(request["priority"]),
                )
            )

            served_requests.add(opportunity.request_id)

            # Block the satellite for [start - buffer, end + buffer].
            # bisect.insort keeps the list sorted by busy_start (needed by the improved greedy).
            buffer = timedelta(seconds=self.slew_buffer_seconds)
            bisect.insort(busy_intervals[opportunity.satellite_id], (start - buffer, end + buffer))

            used_energy[opportunity.satellite_id] += request["energy_wh"]
            used_memory[opportunity.satellite_id] += request["data_volume_gb"]
            orbit_counts[opportunity.satellite_id][orbit_block] = current_count + 1

        return ScheduleResult(
            status="FEASIBLE",      # A heuristic always returns a valid (not necessarily optimal) schedule.
            objective_value=sum(observation.value for observation in scheduled),    # Total value served.
            runtime_seconds=time.perf_counter() - started,
            scheduled=scheduled,
            solver_status="GREEDY_HEURISTIC",
        )

    def _find_start(self, busy_intervals: list[tuple], opportunity):
        """
        Baseline rule: start exactly at window_start, or skip.

        Returns:
            window_start if [window_start, window_start + duration) does not overlap any busy
            interval, otherwise None.
        """
        start = opportunity.window_start
        end = start + timedelta(seconds=opportunity.duration_seconds)

        for busy_start, busy_end in busy_intervals:
            # Two intervals [a, b) and [c, d) overlap  <=>  a < d and b > c
            if start < busy_end and end > busy_start:
                return None

        return start

    @staticmethod
    def _fits_time_window(opportunity, request,) -> bool:
        """
        Formula:
            earliest_start <= window_start
            window_end     <= latest_end
            duration       <= window_end - window_start
        """
        return (
            opportunity.window_start >= request["earliest_start"]
            and opportunity.window_end <= request["latest_end"]
            and opportunity.duration_seconds <= (opportunity.window_end - opportunity.window_start).total_seconds()
        )

    @staticmethod
    def _orbit_block(timestamp, base_time, orbit_period_minutes: float,) -> int:
        """
        Index of the orbit that contains `timestamp`.

        Formula:
            orbit_seconds = orbit_period_minutes * 60
            block_id = floor((timestamp - base_time) / orbit_seconds)
        Ex:
            orbit = 96 min = 5,760 s, offset = 7,000 s  -> block_id = 1
        """
        orbit_seconds = orbit_period_minutes * 60

        return int((timestamp - base_time).total_seconds() // orbit_seconds)


class ImprovedGreedyScheduler(GreedyScheduler):
    """
    Same greedy order and constraints as GreedyScheduler, but a smarter start-time rule:

    Instead of "window_start or nothing", it takes the EARLIEST free start time inside
        [window_start, window_end - duration]
    that does not overlap any busy interval of the satellite.

    Ex:
        window = 08:12:00 - 08:12:45, duration = 10 s
        busy   = 08:11:50 - 08:12:20  (previous capture + buffer)
        baseline -> 08:12:00 overlaps -> skip
        improved -> start at 08:12:20, end 08:12:30 (still inside the window) -> accept
    """

    METHOD_NAME = "greedy_improved"

    def _find_start(self, busy_intervals: list[tuple], opportunity):
        duration = timedelta(seconds=opportunity.duration_seconds)

        # Latest start that still ends inside the window.
        # Formula:
        #   latest_start = window_end - duration
        latest_start = opportunity.window_end - duration

        candidate = opportunity.window_start    # Try the earliest possible start first.

        # busy_intervals is sorted by busy_start, so one left-to-right pass is enough:
        #   - an interval that ends before the candidate is irrelevant,
        #   - an interval that starts after the candidate capture ends means all later ones do too,
        #   - otherwise the candidate overlaps -> push it to the end of that busy interval.
        for busy_start, busy_end in busy_intervals:
            if busy_end <= candidate:
                continue

            if busy_start >= candidate + duration:
                break

            candidate = busy_end

            if candidate > latest_start:    # Pushed past the window -> no free slot.
                return None

        return candidate if candidate <= latest_start else None

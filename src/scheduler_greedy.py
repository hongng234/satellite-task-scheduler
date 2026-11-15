from __future__ import annotations     # Lets type hints use names defined later in the file.

import bisect                           # insort(): insert into a list while keeping it sorted.
import time                             # perf_counter(): precise clock to measure the runtime.
from collections import defaultdict     # Dict that creates a default value (0, [], {}) for a missing key.
from datetime import timedelta          # A duration. Ex: timedelta(seconds=10)
from pathlib import Path                # File paths that work on Windows and Linux.

import pandas as pd                     # Tables (DataFrames) read from the Parquet files.

from src.models import (Priority, ScheduleResult, ScheduledObservation, validate_tables,)   # Shared output format and input checks.


## Key point: a greedy scheduler is fast, but its schedule is usually not the best one.
## It looks at the opportunities in a fixed order, decides "take it" or "skip it" once for each,
## and never changes an earlier decision.
## The exact CP-SAT scheduler (scheduler_cp_sat.py) is compared against it, and CP-SAT also
## starts its search from the better of the two greedy schedules (warm start).
##
## There are two variants. The only difference is how they choose the start time (_find_start):
##   GreedyScheduler          -> start exactly at window_start, or skip.
##   ImprovedGreedyScheduler  -> earliest free start anywhere inside the window.


class GreedyScheduler:
    """
    Baseline greedy heuristic.

    Idea:
        1. Sort all opportunities from best to worst (priority first, then value per second).
        2. Go through the sorted list once. Accept an opportunity if it breaks no constraint
           together with the ones already accepted.
        3. Never change an earlier decision.

    Baseline rule for the start time:
        The capture always starts exactly at window_start.
        If the satellite is busy at that moment, the opportunity is skipped,
        even when a later start inside the window would have been free.
    """

    # Name used in reports (run_experiment.py, benchmark.py) and in output file names.
    # Ex: output/greedy_schedule.parquet
    METHOD_NAME = "greedy"

    def __init__(self, satellites_path: Path, requests_path: Path, opportunities_path: Path, slew_buffer_seconds: float = 30.0,) -> None:
        # Read the 3 Parquet files and pass them to _setup.
        #   satellites_path     -> data/processed/satellites.parquet
        #   requests_path       -> data/processed/requests.parquet
        #   opportunities_path  -> data/processed/opportunities.parquet
        #   slew_buffer_seconds -> min idle time between 2 captures on one satellite (default 30 s)
        self._setup(
            satellites=pd.read_parquet(satellites_path),
            requests=pd.read_parquet(requests_path),
            opportunities=pd.read_parquet(opportunities_path),
            slew_buffer_seconds=slew_buffer_seconds,
        )

    # A @classmethod is called on the class, as in GreedyScheduler.from_dataframes(...), and receives that class as `cls`.
    # Since it builds the object from `cls`, ImprovedGreedyScheduler.from_dataframes(...) returns an ImprovedGreedyScheduler.
    @classmethod
    def from_dataframes(cls, satellites: pd.DataFrame, requests: pd.DataFrame, opportunities: pd.DataFrame, slew_buffer_seconds: float = 30.0,) -> GreedyScheduler:
        """
        Build the scheduler from DataFrames that are already in memory.
        The CP-SAT scheduler uses this to compute its warm-start hint.
        """
        scheduler = cls.__new__(cls)    # Create the object without calling __init__, which would read files.
        scheduler._setup(satellites=satellites, requests=requests, opportunities=opportunities, slew_buffer_seconds=slew_buffer_seconds,)
        return scheduler

    def _setup(self, satellites: pd.DataFrame, requests: pd.DataFrame, opportunities: pd.DataFrame, slew_buffer_seconds: float,) -> None:
        # Called by both __init__ and from_dataframes. Stores the tables and builds the lookup dictionaries.
        # The leading "_" marks a helper that code outside the class should not call.
        self.satellites = satellites        # One row per satellite (capacities, orbit period, ...).
        self.requests = requests            # One row per request (value, priority, energy, data, ...).

        # Keep only the rows with feasible = True. .copy() leaves the caller's DataFrame untouched.
        self.opportunities = opportunities[opportunities["feasible"]].copy()

        # Dictionaries keyed by id, so a lookup does not have to search the table:
        #   satellite_id -> {"battery_capacity_wh": ..., ...}
        #   request_id   -> {"value": ..., "energy_wh": ..., ...}
        # Ex:
        #   self.request_lookup["REQ-0001"]["value"]  -> 120.5
        self.satellite_lookup = self.satellites.set_index("satellite_id").to_dict("index")
        self.request_lookup = self.requests.set_index("request_id").to_dict("index")

        # Idle time a satellite needs between two captures to turn towards the next target.
        self.slew_buffer_seconds = slew_buffer_seconds

        # Stop here if the three tables do not match each other.
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """
        Raise ValueError if the tables do not fit together. CP-SAT runs the same checks.
        The checks are listed in validate_tables (models.py). Ex: an unknown request id, a negative slew buffer.
        """
        validate_tables(self.satellites, self.requests, self.opportunities, self.slew_buffer_seconds)

    def solve(self) -> ScheduleResult:
        """
        Build the greedy schedule.

        Returns:
            ScheduleResult with status "FEASIBLE" and solver_status "GREEDY_HEURISTIC".
            The status is never "OPTIMAL", because a heuristic cannot prove that its schedule is the best.
        """
        started = time.perf_counter()   # Start the runtime clock.

        # ---- Step 1: score and sort opportunities -------------------------------------------
        #
        # Sort key, in ascending order, so the best opportunity comes first:
        #   1. -priority          -> CRITICAL (4) before LOW (1)
        #   2. -value_density     -> higher value per second of capture first
        #   3. window_start       -> earlier windows first
        #   4. opportunity_id     -> breaks ties, so the order is the same on every run
        #
        # sort() puts the smallest key first. The minus signs make the largest priority and density the smallest keys.
        #
        # Formula:
        #   value_density = value / duration_seconds
        # Ex:
        #   value = 120, duration = 10 s  -> density = 12 per second
        scored = []         # List of (sort_key, opportunity) pairs.

        for opportunity in self.opportunities.itertuples(index=False):     # One row at a time, with fields as attributes (opportunity.request_id).
            request = self.request_lookup[opportunity.request_id]           # The request this opportunity would serve.

            value_density = request["value"] / request["duration_seconds"]

            sort_key = (-int(request["priority"]), -value_density, opportunity.window_start, opportunity.opportunity_id,)

            scored.append((sort_key, opportunity))

        # Sort by the key only, so Python never has to compare two opportunity rows.
        scored.sort(key=lambda item: item[0])

        # ---- Step 2: state of each satellite while we accept opportunities ------------------

        # busy_intervals[satellite_id] = list of (busy_start, busy_end), kept sorted by busy_start.
        # Each accepted capture blocks [start - buffer, end + buffer]. Any other capture then has
        # to stay at least `buffer` seconds away, before or after.
        # Ex (buffer = 30 s):
        #   capture [08:12:00, 08:12:10]  -> busy [08:11:30, 08:12:40]
        busy_intervals: dict[str, list[tuple]] = defaultdict(list)

        used_energy: dict[str, float] = defaultdict(float)          # Wh consumed so far per satellite (starts at 0.0).
        used_memory: dict[str, float] = defaultdict(float)          # GB stored so far per satellite (starts at 0.0).
        orbit_counts: dict[str, dict[int, int]] = defaultdict(dict) # orbit_counts[sat][block_id] = captures already in that orbit.

        # Second 0 of the problem: the earliest window of all. Orbit blocks are counted from here.
        base_time = self.opportunities["window_start"].min()

        scheduled = []              # Accepted observations.
        served_requests = set()     # request_ids that already have a capture.

        # ---- Step 3: go through the sorted list and accept what fits -----------------------
        # Each `continue` below skips an opportunity that would break a constraint.
        for _, opportunity in scored:       # `_` is the sort key, which is no longer needed.
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
            # Ex:
            #   used 1,190 Wh + 18 Wh = 1,208 Wh > capacity 1,200 Wh -> skip
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

            current_count = orbit_counts[opportunity.satellite_id].get(orbit_block, 0)     # .get(..., 0) returns 0 for an orbit with no capture yet.

            if current_count >= satellite["max_observations_per_orbit"]:    # This orbit is already full.
                continue

            # Constraint: no overlap with the captures already on this satellite, slew buffer included.
            # This is the one step where the baseline and the improved greedy differ.
            start = self._find_start(busy_intervals=busy_intervals[opportunity.satellite_id], opportunity=opportunity,)

            if start is None:       # The satellite has no free start time for this capture.
                continue

            # The capture lasts exactly duration_seconds.
            # Formula:
            #   end = start + duration
            end = start + timedelta(seconds=opportunity.duration_seconds)

            # ---- Every check passed: accept the opportunity and update the state ----
            scheduled.append(
                ScheduledObservation(
                    request_id=opportunity.request_id,
                    satellite_id=opportunity.satellite_id,
                    opportunity_id=opportunity.opportunity_id,
                    start=start,
                    end=end,
                    value=request["value"],
                    priority=Priority(request["priority"]),     # int -> Priority enum. Ex: 3 -> Priority.HIGH
                )
            )

            served_requests.add(opportunity.request_id)     # This request cannot be served again.

            # Block the satellite for [start - buffer, end + buffer].
            # bisect.insort keeps the list sorted by busy_start, which the improved greedy relies on.
            buffer = timedelta(seconds=self.slew_buffer_seconds)
            bisect.insort(busy_intervals[opportunity.satellite_id], (start - buffer, end + buffer))

            # Take the energy, memory and orbit slot from the satellite's budget.
            used_energy[opportunity.satellite_id] += request["energy_wh"]
            used_memory[opportunity.satellite_id] += request["data_volume_gb"]
            orbit_counts[opportunity.satellite_id][orbit_block] = current_count + 1

        return ScheduleResult(
            status="FEASIBLE",      # Valid, but with no proof that it is the best schedule.
            objective_value=sum(observation.value for observation in scheduled),    # Total value served.
            runtime_seconds=time.perf_counter() - started,                          # Elapsed time since `started`.
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
        start = opportunity.window_start                                    # The only start time the baseline tries.
        end = start + timedelta(seconds=opportunity.duration_seconds)       # Formula: end = start + duration

        for busy_start, busy_end in busy_intervals:
            # Two intervals [a, b) and [c, d) overlap  <=>  a < d and b > c
            # Ex:
            #   capture [100, 110), busy [90, 105)  -> 100 < 105 and 110 > 90  -> overlap -> None
            #   capture [100, 110), busy [110, 140) -> 110 > 110 is False      -> no overlap
            if start < busy_end and end > busy_start:
                return None

        return start        # Nothing overlaps, so the satellite is free at window_start.

    # A @staticmethod is a plain function kept inside the class. It does not use `self`.
    @staticmethod
    def _fits_time_window(opportunity, request,) -> bool:
        """
        Formula:
            earliest_start <= window_start
            window_end     <= latest_end
            duration       <= window_end - window_start
        Ex:
            request [02:00, 12:00], window [08:12:00, 08:12:30], duration 10 s -> True
        """
        # The 3 conditions are joined with `and`, so all of them must hold.
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
        orbit_seconds = orbit_period_minutes * 60       # Minutes -> seconds.

        # `//` is floor division: it divides and rounds down. int() turns the float result into an integer index.
        return int((timestamp - base_time).total_seconds() // orbit_seconds)


# ImprovedGreedyScheduler(GreedyScheduler) inherits the constructor, the sorting and the constraint
# checks from GreedyScheduler. It replaces only METHOD_NAME and _find_start.
class ImprovedGreedyScheduler(GreedyScheduler):
    """
    Same order and same constraints as GreedyScheduler, with a different start-time rule.

    The baseline tries window_start and nothing else. This variant takes the earliest start in
        [window_start, window_end - duration]
    that does not overlap any busy interval of the satellite.

    Ex:
        window = 08:12:00 - 08:12:45, duration = 10 s
        busy   = 08:11:50 - 08:12:20  (previous capture + buffer)
        baseline -> 08:12:00 overlaps -> skip
        improved -> start at 08:12:20, end 08:12:30 (still inside the window) -> accept
    """

    METHOD_NAME = "greedy_improved"     # Ex: output/greedy_improved_schedule.parquet

    def _find_start(self, busy_intervals: list[tuple], opportunity):
        duration = timedelta(seconds=opportunity.duration_seconds)     # Capture length as a timedelta.

        # Latest start that still ends inside the window.
        # Formula:
        #   latest_start = window_end - duration
        # Ex:
        #   window_end = 08:12:45, duration = 10 s -> latest_start = 08:12:35
        latest_start = opportunity.window_end - duration

        candidate = opportunity.window_start    # Try the earliest possible start first.

        # busy_intervals is sorted by busy_start, so one pass from left to right is enough:
        #   - an interval that ends before the candidate cannot be in the way,
        #   - an interval that starts after the capture would end leaves the slot free, and so do all later ones,
        #   - any other interval overlaps, and the candidate moves to the end of it.
        for busy_start, busy_end in busy_intervals:
            if busy_end <= candidate:                   # This busy interval is over before the candidate starts.
                continue

            if busy_start >= candidate + duration:      # It starts after the capture would end, so the slot is free.
                break

            candidate = busy_end                        # Overlap. Try again right after this busy interval.

            if candidate > latest_start:    # The candidate was pushed past the window, so there is no free slot.
                return None

        # The loop can also end without a busy interval in the way, so check the window once more.
        return candidate if candidate <= latest_start else None

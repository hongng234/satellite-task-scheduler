from __future__ import annotations     # Lets type hints use names defined later in the file.

import math                             # ceil() / floor() to round times and amounts to integers.
import time                             # perf_counter(): precise clock to measure the runtime.
from collections import defaultdict     # Dict that creates a default value ([], 0, ...) for a missing key.
from datetime import timedelta          # A duration. Ex: timedelta(seconds=735)
from pathlib import Path                # File paths that work on Windows and Linux.

import pandas as pd                     # Tables (DataFrames) read from the Parquet files.
from ortools.sat.python import cp_model # Google OR-Tools CP-SAT: an exact solver for problems with integer variables.

from src.models import (Priority, ScheduleResult, ScheduledObservation, validate_tables,)   # Shared output format and input checks.
from src.scheduler_greedy import GreedyScheduler, ImprovedGreedyScheduler  # Used to build the warm-start hint.


## Key point: this is the exact scheduler. The greedy schedulers decide on one opportunity at a time.
## This one writes the whole problem as a mathematical model and lets CP-SAT search for the best schedule.
##
## The model:
##   Decision variable per opportunity i:  x_i ∈ {0, 1}   (1 = take this capture, 0 = skip it)
##   Objective:                           maximize sum(value_i * x_i)
##   Constraints (numbered as in the code below):
##     1. No overlap per satellite, with a slew buffer between two captures.
##     2. At most one opportunity per request.
##     3. Battery:  sum(energy_i * x_i) <= battery_capacity   (per satellite)
##     4. Memory:   sum(data_i * x_i)   <= memory_capacity    (per satellite)
##     5. Per orbit block: sum(x_i) <= max_observations_per_orbit
##   "The capture stays inside its window" has no number here. It is enforced by the range
##   of the start variable, so it needs no constraint of its own.
##
## If CP-SAT finishes before the time limit it returns OPTIMAL (proven best schedule).
## If time runs out first it returns FEASIBLE (the best schedule found so far).
##
## CP-SAT only handles integers, so:
##   - times are converted to whole seconds since base_time,
##   - decimal values are multiplied by a scale factor (value x100, energy x10, data x100).


class SatelliteTaskingScheduler:
    # Scale factors to turn decimal data into integers for CP-SAT.
    VALUE_SCALE = 100       # value has 2 decimals, e.g. 123.45 -> 12345
    ENERGY_SCALE = 10       # energy_wh has 1 decimal, e.g. 18.3 -> 183
    DATA_SCALE = 100        # data_volume_gb has 2 decimals, e.g. 2.57 -> 257

    def __init__(self, satellites_path: Path, requests_path: Path, opportunities_path: Path, slew_buffer_seconds: float=30.0, time_limit_seconds: float=30.0, num_workers: int=8, use_greedy_hint: bool=True,) -> None:
        """
        Initialize the scheduler from 3 Parquet files.

        Parameters:
            satellites_path: file containing satellite information.
            requests_path: file containing image acquisition requests.
            opportunities_path: file containing feasible observation windows.
            slew_buffer_seconds: the buffer time for the satellite to reorient
                                 between two image captures.
            time_limit_seconds: the maximum runtime for CP-SAT.
            num_workers: the number of parallel threads used by CP-SAT.
            use_greedy_hint: warm-start CP-SAT with the best greedy schedule.
                             On large datasets CP-SAT may otherwise hit the time limit with
                             only the empty solution (objective = 0).
        """

        # ---- Read the 3 input tables ----
        self.satellites = pd.read_parquet(satellites_path)  # One row per satellite. Ex: satellite_id, orbit_period_minutes, battery_capacity_wh, ...
        self.requests = pd.read_parquet(requests_path)      # One row per customer request. Ex: request_id, priority, value, earliest_start, latest_end, ...
        self.opportunities = pd.read_parquet(opportunities_path)    # One row per opportunity: "SAT-03 can capture REQ-001 from 08:12 to 08:15".

        # Keep only the rows with feasible = True. A row with feasible = False cannot be used.
        # Ex: the satellite cannot see the AOI, or the viewing angle is too large.
        self.opportunities = self.opportunities[self.opportunities["feasible"]].copy()

        # Turn the two tables into dictionaries keyed by id.
        #
        #   {
        #       "SAT-01": {"orbit_period_minutes": 96, ...},
        #       "SAT-02": {"orbit_period_minutes": 97, ...},
        #   }
        #
        #   {
        #       "REQ-0001": {"value": 120, "energy_wh": 18, ...},
        #       "REQ-0002": {"value": 45, "energy_wh": 10, ...},
        #   }
        # Looking up SAT-01 is then a single dictionary access. Searching the DataFrame
        # on every lookup would be much slower.
        self.satellite_lookup = self.satellites.set_index("satellite_id").to_dict("index")
        self.request_lookup = self.requests.set_index("request_id").to_dict("index")


        # ---- Store the solver settings (described in the docstring) ----
        self.slew_buffer_seconds = slew_buffer_seconds      # Ex: 30.0 s of idle time between 2 captures on one satellite.
        self.time_limit_seconds = time_limit_seconds        # Ex: 30.0 s, then CP-SAT returns the best schedule found so far.
        self.num_workers = num_workers                      # Ex: 8 threads searching in parallel.
        self.use_greedy_hint = use_greedy_hint              # True -> start the search from the greedy schedule.

        # Stop here if the three tables do not match each other.
        self._validate_inputs()


    def _validate_inputs(self) -> None:

        """
        Raise ValueError if the tables do not fit together. The greedy schedulers run the same checks.
        The checks are listed in validate_tables (models.py). Ex: an unknown request id, a negative slew buffer.
        """
        validate_tables(self.satellites, self.requests, self.opportunities, self.slew_buffer_seconds)


    def solve(self) -> ScheduleResult:
        """
        Build the CP-SAT model and solve it.

        Objective:
            Maximize the total value of the requests served.

        Constraints:
            - A request may be served at most once.
            - A satellite may not perform two overlapping observations.
            - A satellite may not use more energy than its battery holds.
            - A satellite may not store more data than its memory holds.
            - A satellite may not take more captures in one orbit than its limit.
        """

        started = time.perf_counter()   # perf_counter() is more accurate than time.time() for measuring a duration.

        model = cp_model.CpModel()  # An empty CP-SAT problem. The variables, constraints and objective are added to it below.

        # base_time is second 0 of the whole problem.
        # CP-SAT needs integers, so every datetime becomes a number of seconds since base_time.
        #
        # Ex:
        #   base_time = 08:00:00
        #   window_start = 08:12:00
        #   window_start_seconds = 720 seconds
        base_time = self.opportunities["window_start"].min()

        # With no usable opportunity there is nothing to schedule, and base_time would be NaT.
        if self.opportunities.empty:
            return ScheduleResult(status="OPTIMAL", objective_value=0.0, runtime_seconds=time.perf_counter() - started, scheduled=[], solver_status="NO_OPPORTUNITIES",)

        # opportunity_vars: the decision variable of each opportunity.
        #
        # Key: opportunity_id
        # Value: binary variable x
        # x = 1: this opportunity is selected.
        # x = 0: it is not.
        # Formula:
        #   x_i ∈ {0, 1}
        opportunity_vars: dict[str, cp_model.IntVar] = {}

        # opportunity_id -> the variable that holds the capture's start time.
        interval_start_vars: dict[str, cp_model.IntVar] = {}

        # Integer (rounded-up) capture duration of each opportunity, as used in the model.
        duration_vars: dict[str, int] = {}

        # Intervals grouped by satellite.
        # Each satellite has its own timeline, and NoOverlap is added once per satellite.
        intervals_by_satellite: dict[str, list[cp_model.IntervalVar]] = defaultdict(list)

        # Terms of the objective, kept as 2 parallel lists (variables and their coefficients).
        #
        # For each opportunity:
        #   contribution = value(request) * x
        #
        # At the end:
        #   objective = sum(contribution)
        #
        # LinearExpr.weighted_sum(vars, coeffs) builds this sum as one flat expression.
        # Python's sum() is far slower once there are tens of thousands of terms.
        objective_vars = []
        objective_coeffs = []

        # opportunity_id -> (earliest_start, latest_start), in integer seconds.
        # The greedy warm-start hint needs these bounds again.
        start_bounds: dict[str, tuple[int, int]] = {}

        # End variable of each interval (= start + duration + slew_buffer). The hint sets it too.
        interval_end_vars: dict[str, cp_model.IntVar] = {}

        # Slew buffer in whole seconds. Rounded up, so the buffer is never shorter than requested.
        slew_buffer = math.ceil(self.slew_buffer_seconds)

        # ---- Create the variables of every opportunity ------------------------------------
        # itertuples gives one row at a time, with columns readable as attributes (opportunity.window_start).
        for opportunity in self.opportunities.itertuples(index=False):
            request = self.request_lookup[opportunity.request_id]   # The request this opportunity would serve.

            # The window must lie inside the request's [earliest_start, latest_end]. The greedy scheduler checks the same thing.
            if opportunity.window_start < request["earliest_start"] or opportunity.window_end > request["latest_end"]:
                continue

            # window_start as seconds since base_time.
            #
            # Formula:
            #   window_start_seconds = ceil((window_start - base_time).total_seconds())
            #
            # ceil, because rounding down would let the capture start before the real window opens.
            #
            # Ex:
            #   base_time = 08:00:00
            #   window_start = 08:12:00
            #   window_start_seconds = 720
            window_start_seconds = math.ceil((opportunity.window_start - base_time).total_seconds())

            # window_end as seconds since base_time.
            #
            # Ex:
            #   base_time = 08:00:00
            #   window_end = 08:12:30
            #   window_end_seconds = 750
            # floor, because rounding up would let the capture end after the real window closes.
            window_end_seconds = math.floor((opportunity.window_end - base_time).total_seconds())

            # Round the duration up to a whole second (8.2 s -> 9 s).
            # Rounding down could make the real capture run past the window.
            duration_seconds = math.ceil(opportunity.duration_seconds)

            # The latest start that still leaves room for the whole capture.
            #
            # Formula:
            #   latest_start = window_end - duration
            #
            # Ex:
            #   window_end = 08:12:30
            #   duration = 10 seconds
            #   latest_start = 08:12:20
            #
            # A capture that starts after 08:12:20 ends after 08:12:30, outside the window.
            latest_start_seconds = (window_end_seconds - duration_seconds)

            # The window is shorter than the capture, so this opportunity can never be used.
            # It gets no variable, and every later loop has to skip it.
            if latest_start_seconds < window_start_seconds:
                continue

            # The binary variable x of this opportunity (1 = selected, 0 = not selected).
            is_selected = model.new_bool_var(f"select_{opportunity.opportunity_id}")

            # The start time of the capture. Its range is what keeps the capture inside the window:
            #   window_start <= interval_start <= latest_start
            #
            # Ex:
            #   window_start = 720 seconds
            #   latest_start = 740 seconds
            #   duration = 10 seconds
            #   window_end = 750 seconds
            #
            # The solver can choose start = 720, 721, ..., 740.
            interval_start = model.new_int_var(window_start_seconds, latest_start_seconds, f"start_{opportunity.opportunity_id}",)

            # The interval on the satellite's timeline is the capture plus the slew buffer after it.
            # The capture itself ([start, start + duration]) stays inside the window.
            # The buffer is the time the satellite needs to turn towards its next target,
            # so the interval may end up to one buffer after window_end.
            #
            #   interval_end = interval_start + duration + slew_buffer
            interval_end = model.new_int_var(window_start_seconds + duration_seconds + slew_buffer, window_end_seconds + slew_buffer, f"end_{opportunity.opportunity_id}",)

            # An optional interval: it exists only when is_selected = 1.
            # When it exists, CP-SAT enforces:
            #   end = start + size
            #
            # It has to be optional. A mandatory interval takes part in NoOverlap even when its
            # opportunity is not selected. Every window of a satellite would then have to be
            # disjoint from the others, and on normal data that makes the model infeasible.
            #
            # Ex (slew_buffer = 30):
            #   start = 730
            #   size = 10 + 30
            #   end = 770
            interval = model.new_optional_interval_var(start=interval_start, size=duration_seconds + slew_buffer, end=interval_end, is_present=is_selected, name=f"interval_{opportunity.opportunity_id}",)

            interval_start_vars[opportunity.opportunity_id] = interval_start    # Its value is read back after solving.

            duration_vars[opportunity.opportunity_id] = duration_seconds        # Integer duration, needed again when the schedule is read back.

            start_bounds[opportunity.opportunity_id] = (window_start_seconds, latest_start_seconds)

            interval_end_vars[opportunity.opportunity_id] = interval_end        # The hint has to cover this variable too.

            opportunity_vars[opportunity.opportunity_id] = is_selected  # The constraints below use this variable.

            intervals_by_satellite[opportunity.satellite_id].append(interval)    # Put the interval on its satellite's timeline.

            # This opportunity's term in the objective.
            # If x = 0:
            #   contribution = value * 0 = 0
            # If x = 1:
            #   contribution = value * 1 = value
            # Formula:
            #   objective_term_i = value_i * x_i
            #
            # value has 2 decimals, so multiplying by 100 gives an exact integer.
            objective_vars.append(is_selected)
            objective_coeffs.append(round(request["value"] * self.VALUE_SCALE))     # round() with no digits returns an int.


        # Constraint 1: a satellite cannot take two captures at the same time.
        #
        # For any two captures j and k on the same satellite:
        #   [start_j, end_j) and [start_k, end_k) must not overlap.
        #
        # NoOverlap enforces this for every pair of intervals in the list it is given.
        # The intervals already include the slew buffer, so the buffer is enforced as well.
        for intervals in intervals_by_satellite.values():
            if intervals:                           # Skip a satellite that has no interval.
                model.add_no_overlap(intervals)


        # The x variables grouped by request: request_id -> [x of each of its opportunities].
        # Ex:
        #   "REQ-0001" -> [x_OPP-000001, x_OPP-000007]
        opportunities_by_request: dict[str, list[cp_model.IntVar]] = defaultdict(list)

        for opportunity in self.opportunities.itertuples(index=False):
            # Dropped earlier, so it has no variable.
            if opportunity.opportunity_id not in opportunity_vars:
                continue
            opportunities_by_request[opportunity.request_id].append(opportunity_vars[opportunity.opportunity_id])

        # Constraint 2: a request is served at most once.
        #
        # A request can have several opportunities:
        #   SAT-01 at 08:12
        #   SAT-04 at 09:30
        # At most one of them may be selected.
        # Formula:
        #   sum(x_i) <= 1
        for variables in opportunities_by_request.values():
            model.add_at_most_one(variables)


        # Constraint 3: battery.
        #
        # The selected captures of a satellite may not use more energy than its battery holds.
        #
        # Formula:
        #   sum(energy_i * x_i) <= battery_capacity
        self._add_resource_constraint(model=model, opportunity_vars=opportunity_vars, resource_column="energy_wh", capacity_column="battery_capacity_wh", scale=self.ENERGY_SCALE,)


        # Constraint 4: memory.
        #
        # The selected captures of a satellite may not produce more data than its memory holds.
        #
        # Formula:
        #   sum(data_volume_i * x_i) <= memory_capacity
        #
        # data_volume_gb has 2 decimals, so scale = 100 makes it an integer.
        # Ex: 2.57 GB * 100 = 257 units.
        self._add_resource_constraint(model=model, opportunity_vars=opportunity_vars, resource_column="data_volume_gb", capacity_column="memory_capacity_gb", scale=self.DATA_SCALE,)

        # Constraint 5: maximum number of captures per orbit.
        self._add_orbit_capacity_constraints(model=model, opportunity_vars=opportunity_vars,)

        # Objective: the total value of the requests served, as large as possible.
        # Formula:
        #   max sum(value_i * x_i)
        model.maximize(cp_model.LinearExpr.weighted_sum(objective_vars, objective_coeffs))

        # Warm start: hand CP-SAT the better of the two greedy schedules as a starting solution (a hint).
        # CP-SAT then only has to improve a schedule that is already good. Started from nothing,
        # it can reach the time limit on large data with objective = 0.
        if self.use_greedy_hint:
            self._add_greedy_hint(model=model, opportunity_vars=opportunity_vars, interval_start_vars=interval_start_vars, interval_end_vars=interval_end_vars, duration_vars=duration_vars, start_bounds=start_bounds, slew_buffer=slew_buffer,)

        # ---- Solve ------------------------------------------------------------------------
        solver = cp_model.CpSolver()        # The engine that searches for the best value of every variable.

        # Time limit. If it runs out before the solver has proved optimality,
        # the solver still returns the best schedule it found.
        solver.parameters.max_time_in_seconds = self.time_limit_seconds

        solver.parameters.num_workers = self.num_workers                # Number of parallel threads.
        solver.parameters.log_search_progress = False                   # Set to True to see the solver's search log when debugging.

        status = solver.solve(model)         # Run the solver. The status is OPTIMAL, FEASIBLE, INFEASIBLE or UNKNOWN.

        runtime_seconds = time.perf_counter() - started                 # Total time, model building included.

        # solver.value() only has something to return when a solution was found.
        # With UNKNOWN or INFEASIBLE there is none, and reading a variable raises an error.
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            scheduled = self._extract_schedule(solver=solver, opportunity_vars=opportunity_vars, interval_start_vars=interval_start_vars, duration_vars=duration_vars, base_time=base_time,)
        else:
            # No solution, so the schedule is empty.
            scheduled = []

        # OR-Tools returns the status as a number. Turn it into a readable label.
        if status == cp_model.OPTIMAL:
            status_label = "OPTIMAL"        # The best possible schedule, and the solver proved it.
        elif status == cp_model.FEASIBLE:
            status_label = "FEASIBLE"       # A valid schedule, but time ran out before the solver could prove it is the best.
        elif status == cp_model.INFEASIBLE:
            status_label = "INFEASIBLE"     # The constraints contradict each other, so no valid schedule exists.
        else:
            status_label = "UNKNOWN"        # Time ran out before any solution was found.

        return ScheduleResult(
            status=status_label,
            # Summed from the original float values of the scheduled captures.
            objective_value=sum(observation.value for observation in scheduled),
            runtime_seconds=runtime_seconds,
            scheduled=scheduled,
            solver_status=solver.status_name(status),
            )


    def _add_resource_constraint(self, model: cp_model.CpModel, opportunity_vars: dict[str, cp_model.IntVar], resource_column: str, capacity_column: str, scale: int,) -> None:

        """
        Add one resource constraint per satellite.

        Used twice:
            - energy_wh against battery_capacity_wh
            - data_volume_gb against memory_capacity_gb

        Formula:
            sum(resource_i * x_i) <= capacity

        Parameters:
            resource_column: request column with the amount one capture uses.
            capacity_column: satellite column with the amount available.
            scale: multiplier that turns the decimal amounts into integers.
        """

        # Opportunities grouped by satellite.
        opportunities_by_satellite: dict[str, list] = defaultdict(list)

        for opportunity in self.opportunities.itertuples(index=False):
            opportunities_by_satellite[opportunity.satellite_id].append(opportunity)

        # One constraint per satellite.
        for satellite_id, opportunities in opportunities_by_satellite.items():
            satellite = self.satellite_lookup[satellite_id]     # Row of this satellite.

            terms = []          # The x_i variables of this satellite.
            coefficients = []   # The scaled amount each one uses, in the same order.

            for opportunity in opportunities:
                if opportunity.opportunity_id not in opportunity_vars:     # Dropped earlier, so it has no variable.
                    continue

                request = self.request_lookup[opportunity.request_id]   # The request holds the energy and data amounts.

                # Add the term:
                #   resource_i * x_i
                # If x_i = 0, the contribution is 0.
                # If x_i = 1, the contribution is resource_i.
                #
                # The amount is rounded up, so the real usage can never exceed the capacity.
                # The small epsilon absorbs float noise: 18.3 * 10 = 183.00000000000003 would otherwise become 184.
                terms.append(opportunity_vars[opportunity.opportunity_id])
                coefficients.append(math.ceil(request[resource_column] * scale - 1e-9))

            if not terms:       # No variable left for this satellite, so there is nothing to constrain.
                continue

            # The satellite's capacity, scaled the same way as the terms.
            # It is rounded down, for the same reason the amounts are rounded up.
            capacity = math.floor(satellite[capacity_column] * scale + 1e-9)

            # Constraint:
            #   sum(resource_i * x_i) <= capacity
            model.add(cp_model.LinearExpr.weighted_sum(terms, coefficients) <= capacity)


    def _add_orbit_capacity_constraints(self, model: cp_model.CpModel, opportunity_vars: dict[str, cp_model.IntVar],) -> None:

        """
        Limit the number of captures per orbit.

        Formula:
            For each satellite s and orbit block b:
                sum(x_i) <= max_observations_per_orbit_s

        The orbit block is calculated as:
            block_id = floor((window_start - base_time) / orbit_period)
        """

        base_time = self.opportunities["window_start"].min()    # Second 0, the same one solve() uses.

        # Opportunities grouped by satellite.
        opportunities_by_satellite: dict[str, list] = defaultdict(list)

        for opportunity in self.opportunities.itertuples(index=False):
            opportunities_by_satellite[opportunity.satellite_id].append(opportunity)

        for satellite_id, opportunities in opportunities_by_satellite.items():
            satellite = self.satellite_lookup[satellite_id]     # Each satellite has its own orbit period and per-orbit limit.

            blocks: dict[int, list[cp_model.IntVar]] = defaultdict(list)       # block_id -> x variables of that orbit block.

            for opportunity in opportunities:
                if opportunity.opportunity_id not in opportunity_vars:     # Dropped earlier, so it has no variable.
                    continue

                # The orbit block this opportunity falls in.
                # Formula:
                #   orbit_seconds = orbit_period_minutes * 60
                #   block_id = floor((window_start - base_time) / orbit_seconds)
                # Ex:
                #   orbit_seconds = 5,760 seconds = 96 minutes
                #   window_start - base_time = 7,000 seconds
                #   block_id = floor(7000 / 5760) = 1
                #
                # Blocks are numbered from 0, so block 1 is the satellite's second orbit.
                #
                # The greedy schedulers' function is reused so all schedulers and the checker put an
                # opportunity in the same block. An earlier version rounded the offset down to whole
                # seconds first; with an orbit period like 5,782.2 s that moved some opportunities
                # into the previous block.
                block_id = GreedyScheduler._orbit_block(timestamp=opportunity.window_start, base_time=base_time, orbit_period_minutes=satellite["orbit_period_minutes"],)

                blocks[block_id].append(opportunity_vars[opportunity.opportunity_id])     # Add this opportunity's x to its block.

            # In each orbit block, the number of selected captures stays within the satellite's limit.
            for variables in blocks.values():
                model.add(cp_model.LinearExpr.sum(variables) <= int(satellite["max_observations_per_orbit"]))


    def _add_greedy_hint(self, model: cp_model.CpModel, opportunity_vars: dict[str, cp_model.IntVar], interval_start_vars: dict[str, cp_model.IntVar], interval_end_vars: dict[str, cp_model.IntVar], duration_vars: dict[str, int], start_bounds: dict[str, tuple[int, int]], slew_buffer: int,) -> None:
        """
        Warm-start CP-SAT with the better of the two greedy schedules.

        Steps:
            1. Run GreedyScheduler and ImprovedGreedyScheduler on the same data and keep the better one.
            2. Place its captures again on this model's integer grid (durations and window starts
               rounded up), one satellite at a time. Greedy works with float times, so its
               schedule can break the CP-SAT constraints by a fraction of a second.
            3. Hint every variable of every opportunity: x, start and end.

        Placing the captures of one satellite, sorted by their greedy start time:
            start_k = max(earliest_start_k, previous_end + slew_buffer)
            keep it only if start_k <= latest_start_k
        Ex:
            previous capture: start = 700, duration = 10   -> previous_end = 710
            buffer = 30, next window earliest = 720         -> start = max(720, 740) = 740
        """
        # Run both greedy variants and keep the better one. The improved greedy usually wins,
        # but the baseline came out ahead on the large benchmark scenario.
        # The list comprehension builds and solves one scheduler per class: [greedy_result, improved_result].
        greedy_results = [
            scheduler_class.from_dataframes(
                satellites=self.satellites,
                requests=self.requests,
                opportunities=self.opportunities,
                slew_buffer_seconds=self.slew_buffer_seconds,
            ).solve()
            for scheduler_class in (GreedyScheduler, ImprovedGreedyScheduler)
        ]
        greedy_result = max(greedy_results, key=lambda result: result.objective_value)     # The one with the higher total value.

        # The greedy captures grouped by satellite.
        observations_by_satellite: dict[str, list] = defaultdict(list)
        for observation in greedy_result.scheduled:
            if observation.opportunity_id in opportunity_vars:      # Leave out opportunities that have no variable in this model.
                observations_by_satellite[observation.satellite_id].append(observation)

        selected_starts: dict[str, int] = {}    # opportunity_id -> integer start used in the hint.

        for observations in observations_by_satellite.values():
            observations.sort(key=lambda observation: observation.start)    # In time order on this satellite.

            previous_end = None     # End, in model seconds, of the last capture kept on this satellite.

            for observation in observations:
                earliest, latest = start_bounds[observation.opportunity_id]     # Range of start times the model allows, in seconds.

                # The earliest start that is inside the window and leaves the slew buffer after the previous capture.
                start = earliest if previous_end is None else max(earliest, previous_end + slew_buffer)

                if start > latest:      # After rounding it no longer fits, so the hint leaves it out.
                    continue

                selected_starts[observation.opportunity_id] = start                 # This capture goes into the hint.
                previous_end = start + duration_vars[observation.opportunity_id]    # Formula: previous_end = start + duration

        # Hint every variable. CP-SAT uses a complete hint directly as its first solution.
        # It has to fill in an incomplete one itself, and on the 66k-opportunity model that failed
        # and the hint was ignored.
        #   selected     -> x = 1, start = the start computed above
        #   not selected -> x = 0, start = earliest start (any value in its range would do)
        #   end = start + duration + slew_buffer   (what the interval requires)
        for opportunity_id, is_selected in opportunity_vars.items():
            selected = opportunity_id in selected_starts        # True if this opportunity is in the hint.
            start = selected_starts[opportunity_id] if selected else start_bounds[opportunity_id][0]

            # model.add_hint(variable, value) asks the solver to try this value first.
            # It is only a suggestion, and the solver may still choose another value.
            model.add_hint(is_selected, 1 if selected else 0)
            model.add_hint(interval_start_vars[opportunity_id], start)
            model.add_hint(interval_end_vars[opportunity_id], start + duration_vars[opportunity_id] + slew_buffer)


    def _extract_schedule(self, solver: cp_model.CpSolver, opportunity_vars: dict[str, cp_model.IntVar], interval_start_vars: dict[str, cp_model.IntVar], duration_vars: dict[str, int], base_time: pd.Timestamp,) -> list[ScheduledObservation]:
        """
        Read the x variables after solving and build the list of selected captures.
        """

        scheduled = []      # The selected captures, as ScheduledObservation objects.

        for opportunity in self.opportunities.itertuples(index=False):
            variable = opportunity_vars.get(opportunity.opportunity_id)     # The x variable, or None.

            # None means the opportunity was dropped while the model was built.
            if variable is None:
                continue

            # solver.value(variable) is the value the solver gave x.
            # Only opportunities with x = 1 are part of the schedule.
            if solver.value(variable) != 1:
                continue

            request = self.request_lookup[opportunity.request_id]   # Needed for the value and the priority.

            # The start CP-SAT chose, in seconds since base_time.
            # Ex:
            #   base_time = 08:00:00
            #   solver.value(interval_start) = 735
            #   start = 08:00:00 + 735 seconds
            #         = 08:12:15
            start_seconds = solver.value(interval_start_vars[opportunity.opportunity_id])

            # Back to a datetime.
            start = base_time + timedelta(seconds=start_seconds)

            # The model's rounded-up duration is used here as well, which guarantees end <= window_end.
            # The slew buffer is idle time after the capture and is left out of the observation.
            end = start + timedelta(seconds=duration_vars[opportunity.opportunity_id])

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

        return scheduled


    # ---- Worked example: one opportunity from input to output ----
    #
    # base_time = 08:00:00
    # window_start = 08:12:00
    # window_end = 08:12:30
    # duration = 10 seconds
    # value = 120
    # energy = 18 Wh
    # memory = 2.5 GB

    # In model seconds:
    # start can be 720, 721, ..., 740
    # end = start + 10
    # end is in the range [730, 750]

    # Say the solver selects this opportunity:
    # x = 1
    # start = 735

    # so
    # end = 735 + 10 = 745

    # The capture starts at 08:12:15
    # and ends at 08:12:25

    # Both times are inside the 08:12:00 to 08:12:30 window.

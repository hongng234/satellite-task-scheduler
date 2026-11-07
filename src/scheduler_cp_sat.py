from __future__ import annotations     # Lets type hints use names defined later in the file.

import math                             # ceil() / floor() to round times and amounts to integers.
import time                             # perf_counter(): precise clock to measure the runtime.
from collections import defaultdict     # Dict that creates a default value ([], 0, ...) for a missing key.
from datetime import timedelta          # A duration. Ex: timedelta(seconds=735)
from pathlib import Path                # File paths that work on Windows and Linux.

import pandas as pd                     # Tables (DataFrames) read from the Parquet files.
from ortools.sat.python import cp_model # Google OR-Tools CP-SAT: an exact solver for problems with integer variables.

from src.models import (Priority, ScheduleResult, ScheduledObservation,)   # Shared output format of every scheduler.
from src.scheduler_greedy import GreedyScheduler, ImprovedGreedyScheduler  # Used to build the warm-start hint.


## Key point: this is the EXACT scheduler. Instead of deciding opportunities one by one (greedy),
## it writes the whole problem as a mathematical model and lets CP-SAT search for the best schedule.
##
## The model:
##   Decision variable per opportunity i:  x_i ∈ {0, 1}   (1 = take this capture, 0 = skip it)
##   Objective:                           maximize sum(value_i * x_i)
##   Constraints:
##     1. No overlap per satellite, with a slew buffer between two captures.
##     2. At most one opportunity per request.
##     3. Battery:  sum(energy_i * x_i) <= battery_capacity   (per satellite)
##     4. Memory:   sum(data_i * x_i)   <= memory_capacity    (per satellite)
##     5. Per orbit block: sum(x_i) <= max_observations_per_orbit
##
## If CP-SAT finishes before the time limit it returns OPTIMAL (proven best schedule);
## otherwise FEASIBLE (the best schedule found so far).
##
## CP-SAT only works with INTEGERS, so:
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
        self.satellites = pd.read_parquet(satellites_path)  # Each line represents a satellite; Ex: satellite_id, orbit_period_minutes, battery_capacity_wh, ...
        self.requests = pd.read_parquet(requests_path)      # Each line represents a customer request; Ex: request_id, priority, value, earliest_start, latest_end, ...
        self.opportunities = pd.read_parquet(opportunities_path)    # Each line represents a opportunity: “SAT-03 can capture REQ-001 from 08:12 to 08:15”.

        # Keep only opportunities where feasible = True. "feasible" = False means this opportunity is unusable,
        # Ex: the satellite cannot see the AOI or the observation angle is invalid.
        self.opportunities = self.opportunities[self.opportunities["feasible"]].copy()

        # Create a dictionary for quickly looking up satellite and request information.
        #
        # Before:
        #   satellite_id -> row in DataFrame
        #
        # After:
        #   {
        #       “SAT-01”: {“orbit_period_minutes”: 96, ...},
        #       “SAT-02”: {“orbit_period_minutes”: 97, ...},
        #   }
        #
        #   {
        #       "REQ-0001": {"value": 120, "energy_wh": 18, ...},
        #       "REQ-0002": {"value": 45, "energy_wh": 10, ...},
        #   }
        # This way, when you need information about SAT-01,
        # scheduler can retrieve satellite and request directly without iterating through the entire DataFrame.
        self.satellite_lookup = self.satellites.set_index("satellite_id").to_dict("index")
        self.request_lookup = self.requests.set_index("request_id").to_dict("index")


        # ---- Store the solver settings (explained in the docstring above) ----
        self.slew_buffer_seconds = slew_buffer_seconds      # Ex: 30.0 s of idle time between 2 captures on one satellite.
        self.time_limit_seconds = time_limit_seconds        # Ex: 30.0 s, then CP-SAT returns the best schedule found so far.
        self.num_workers = num_workers                      # Ex: 8 threads searching in parallel.
        self.use_greedy_hint = use_greedy_hint              # True -> start the search from the greedy schedule.

        # Check whether the input data is consistent.
        self._validate_inputs()


    def _validate_inputs(self) -> None:

        """
        Check that every opportunity references
        an existing request and satellite.
        """

        # Find the request_ids that appear in opportunities but are not in the requests table.
        #
        # set(A) - set(B) is the set difference:
        # the elements that are in A but not in B.
        unknown_requests = set(self.opportunities["request_id"]) - set(self.requests["request_id"])
        unknown_satellites = set(self.opportunities["satellite_id"]) - set(self.satellites["satellite_id"])   # Same idea for satellite ids.

        # A non-empty set is "True" in Python -> at least one bad id -> stop with a clear error.
        if unknown_requests:
            raise ValueError(f"Opportunities refer to unknow requests: {unknown_requests}")
        
        if unknown_satellites:
            raise ValueError(f"Oppotunities refer to unknown satellites: {unknown_satellites}")


    def solve(self) -> ScheduleResult:
        """
        A CP-SAT model and find the optimal schedule.

        Objective:
            Maximize the total value of the requests served.

        Constraints:
            - A request may be served at most once.
            - A satellite may not perform two overlapping observations.
            - Not exceed battery capacity.
            - Not exceed memory capacity.
            - Not exceed the maximum number of observations per orbit.
        """

        started = time.perf_counter()   # perf_counter() is more accurate than time.time() for measuring duration.

        model = cp_model.CpModel()  # Create an empty CP-SAT problem. Then variables will be added, constraints, and objective to this model.

        # base_time is the “0-second reference point” for the entire problem.
        # CP-SAT works well with integers, so converting the datetime to the number of seconds since base_time.
        #
        # Ex:
        #   base_time = 08:00:00
        #   window_start = 08:12:00
        #   window_start_seconds = 720 seconds
        base_time = self.opportunities["window_start"].min()

        # No usable opportunity -> nothing to schedule (and base_time would be NaT).
        if self.opportunities.empty:
            return ScheduleResult(status="OPTIMAL", objective_value=0.0, runtime_seconds=time.perf_counter() - started, scheduled=[], solver_status="NO_OPPORTUNITIES",)

        # opportunity_vars: decision variables for each opportunity.
        #
        # Key: opportunity_id
        # Value: binary variable x
        # x = 1: select this opportunity.
        # x = 0: not select.
        # Formula:
        #   x_o ∈ {0, 1}
        opportunity_vars: dict[str, cp_model.IntVar] = {}

        # Store the start time of each opportunity.
        # Value: the interval_start variable for that opportunity
        interval_start_vars: dict[str, cp_model.IntVar] = {}

        # Integer (rounded-up) capture duration of each opportunity, as used in the model.
        duration_vars: dict[str, int] = {}

        # Group intervals by satellite.
        #
        # Purpose: Each satellite has its own timeline. Then, can set NoOverlap for each satellite.
        intervals_by_satellite: dict[str, list[cp_model.IntervalVar]] = defaultdict(list)

        # Components of the objective, kept as 2 parallel lists (variables and their coefficients).
        #
        # For each opportunity:
        #   contribution = value(request) * x
        #
        # Finally:
        #   objective = sum(contribution)
        #
        # LinearExpr.weighted_sum(vars, coeffs) builds this sum in one flat expression.
        # It is much faster than Python's sum() on tens of thousands of terms.
        objective_vars = []
        objective_coeffs = []

        # Integer time bounds of each opportunity in the model: opportunity_id -> (earliest_start, latest_start).
        # Reused to build the greedy warm-start hint.
        start_bounds: dict[str, tuple[int, int]] = {}

        # End variable of each interval (= start + duration + slew_buffer), also hinted.
        interval_end_vars: dict[str, cp_model.IntVar] = {}

        # Slew buffer converted to whole seconds (rounded up so the buffer is never shorter than requested).
        slew_buffer = int(math.ceil(self.slew_buffer_seconds))

        # ---- Create the variables of every opportunity ------------------------------------
        # itertuples: one row at a time, columns readable as attributes (opportunity.window_start).
        for opportunity in self.opportunities.itertuples(index=False):
            request = self.request_lookup[opportunity.request_id]   # The request this opportunity would serve.

            # The window must lie inside the request's [earliest_start, latest_end] (same check as the greedy scheduler).
            if opportunity.window_start < request["earliest_start"] or opportunity.window_end > request["latest_end"]:
                continue

            # Convert window_start from datetime to seconds calculated from base_time.
            #
            # Formula:
            #   window_start_seconds = ceil((window_start - base_time).total_seconds())
            #
            # ceil (not int/floor) so the capture never starts before the real window start.
            #
            # Ex:
            #   base_time = 08:00:00
            #   window_start = 08:12:00
            #   window_start_seconds = 720
            window_start_seconds = int(math.ceil((opportunity.window_start - base_time).total_seconds()))

            # Convert window_end to the number of seconds since base_time.
            #
            # Example:
            #   base_time = 08:00:00
            #   window_end = 08:12:30
            #   window_end_seconds = 750
            # floor so the capture never ends after the real window end.
            window_end_seconds = int(math.floor((opportunity.window_end - base_time).total_seconds()))

            # Round the duration UP to a whole second (8.2 s -> 9 s). CP-SAT works with integers,
            # and rounding down could make the real capture overrun the window.
            duration_seconds = int(math.ceil(opportunity.duration_seconds))

            # The latest possible start time that still allows enough time for the capture.
            #
            # Formula:
            #   latest_start = window_end - duration
            #
            # Ex:
            #   window_end = 08:12:30
            #   duration = 10 seconds
            #   latest_start = 08:12:20
            #
            # If the start time is after 08:12:20, the capture will end after 08:12:30, which means it will fall outside the observation window.
            latest_start_seconds = (window_end_seconds - duration_seconds)

            # If the window is too short, and there isn't enough time to make the request, then skip this opportunity.
            if latest_start_seconds < window_start_seconds:
                continue

            # Create a binary variable x for this opportunity.
            # x = 1: the opportunity is selected.
            # x = 0: the opportunity is not selected.
            is_selected = model.new_bool_var(f"select_{opportunity.opportunity_id}")

            # Create a variable for the actual start time of the capture.
            # It must fall within the following range:
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

            # The interval on the satellite timeline = capture + slew buffer afterwards.
            # The capture itself ([start, start + duration]) stays inside the window;
            # the buffer is time the satellite needs to re-orient before the next capture,
            # so the interval end may exceed window_end by at most the buffer.
            #
            #   interval_end = interval_start + duration + slew_buffer
            interval_end = model.new_int_var(window_start_seconds + duration_seconds + slew_buffer, window_end_seconds + slew_buffer, f"end_{opportunity.opportunity_id}",)

            # Create an OPTIONAL interval variable, present only when is_selected = 1.
            # CP-SAT imposes (only if present):
            #   end = start + size
            #
            # IMPORTANT: it must be optional. A mandatory interval would take part in NoOverlap
            # even when the opportunity is not selected, forcing ALL windows of a satellite to be
            # disjoint (usually infeasible).
            #
            # Example (slew_buffer = 30):
            #   start = 730
            #   size = 10 + 30
            #   end = 770
            interval = model.new_optional_interval_var(start=interval_start, size=duration_seconds + slew_buffer, end=interval_end, is_present=is_selected, name=f"interval_{opportunity.opportunity_id}",)

            interval_start_vars[opportunity.opportunity_id] = interval_start    # The actual values can be read after the solution is calculated.

            duration_vars[opportunity.opportunity_id] = duration_seconds        # Integer duration used in the model (reused when extracting the schedule).

            start_bounds[opportunity.opportunity_id] = (window_start_seconds, latest_start_seconds)

            interval_end_vars[opportunity.opportunity_id] = interval_end        # Needed for a complete hint.

            opportunity_vars[opportunity.opportunity_id] = is_selected  # Store the decision for use in later constraints.

            intervals_by_satellite[opportunity.satellite_id].append(interval)    # Add the interval to the corresponding satellite's timeline.

            # Add a contribution to the objective.
            # If x = 0:
            #   contribution = value * 0 = 0
            # If x = 1:
            #   contribution = value * 1 = value
            # Formula:
            #   objective_term_i = value_i * x_i
            #
            # value has 2 decimals, so it is scaled by 100 to keep it exact as an integer.
            objective_vars.append(is_selected)
            objective_coeffs.append(int(round(request["value"] * self.VALUE_SCALE)))


        # A satellite cannot perform overlapping observations.
        # Constraint 1:
        # A satellite cannot perform two temporally overlapping observations.
        #
        # For any pair of tasks j and k on the same satellite:
        #   [start_j, end_j) and [start_k, end_k) must not overlap.
        #
        # CP-SAT's NoOverlap does this automatically
        # for all intervals in the same list.
        for satellite_id, intervals in intervals_by_satellite.items():
            if intervals:                           # Only satellites that have at least one interval.
                model.add_no_overlap(intervals)


        # Group the x variables by request: request_id -> [x of each of its opportunities].
        # Ex:
        #   "REQ-0001" -> [x_OPP-000001, x_OPP-000007]
        opportunities_by_request: dict[str, list[cp_model.IntVar]] = defaultdict(list)

        for opportunity in self.opportunities.itertuples(index=False):
            # Skip opportunities that were filtered out above (no variable in the model).
            if opportunity.opportunity_id not in opportunity_vars:
                continue
            opportunities_by_request[opportunity.request_id].append(opportunity_vars[opportunity.opportunity_id])

        # Each request can be served at most once.
        # Constraint 2:
        # A request may be served at most once.
        #
        # A request may have multiple opportunities:
        #   SAT-01 at 08:12
        #   SAT-04 at 09:30
        # However, maximum one opportunity may be selected.
        # Formula:
        #   sum(x_i) <= 1
        for variables in opportunities_by_request.values():
            model.add_at_most_one(variables)


        # Constraint 3: Battery constraint.
        #
        # The total energy of the selected tasks must not exceed the battery capacity.
        #
        # Formula:
        #   sum(energy_i * x_i) <= battery_capacity
        self._add_resource_constraint(model=model, opportunity_vars=opportunity_vars, resource_column="energy_wh", capacity_column="battery_capacity_wh", scale=self.ENERGY_SCALE,)


        # Constraint 4: Memory constraint.
        #
        # The total data volume of the selected tasks must not exceed the memory capacity.
        #
        # Formula:
        #   sum(data_volume_i * x_i) <= memory_capacity
        #
        # scale = 100 to avoid decimal places (data_volume_gb has 2 decimals).
        # Example: 2.57 GB * 100 = 257 integer units.
        self._add_resource_constraint(model=model, opportunity_vars=opportunity_vars, resource_column="data_volume_gb", capacity_column="memory_capacity_gb", scale=self.DATA_SCALE,)

        # Constraint 5: Max observations per orbit.
        self._add_orbit_capacity_constraints(model=model, opportunity_vars=opportunity_vars,)

        # Objective:
        #   Maximize the total value of the requests served.
        # Formula:
        #   max sum(value_i * x_i)
        model.maximize(cp_model.LinearExpr.weighted_sum(objective_vars, objective_coeffs))

        # Warm start: give CP-SAT the improved greedy schedule as a starting solution (hint).
        # CP-SAT then begins from a good solution and only has to improve it,
        # instead of starting from scratch (which on large data can end at objective = 0).
        if self.use_greedy_hint:
            self._add_greedy_hint(model=model, opportunity_vars=opportunity_vars, interval_start_vars=interval_start_vars, interval_end_vars=interval_end_vars, duration_vars=duration_vars, start_bounds=start_bounds, slew_buffer=slew_buffer,)

        # ---- Solve ------------------------------------------------------------------------
        solver = cp_model.CpSolver()        # The engine that searches for the best values of all variables.

        # Time limit for the solver.
        # If the time limit expires before the solver can prove that the solution is optimal, the solver may still return the best solution it has found.
        solver.parameters.max_time_in_seconds = self.time_limit_seconds

        solver.parameters.num_workers = self.num_workers                # Number of parallel threads.
        solver.parameters.log_search_progress = False                   # Turn off the solver's detailed log.

        status = solver.solve(model)         # Run the solver. The status can be OPTIMAL, FEASIBLE, INFEASIBLE, or UNKNOWN.

        runtime_seconds = time.perf_counter() - started                 # Measure the total elapsed time.

        # Only read the solution when the solver has actually found a solution.
        # If the status is UNKNOWN or INFEASIBLE, solver.value() does not have a valid value to read.
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            scheduled = self._extract_schedule(solver=solver, opportunity_vars=opportunity_vars, interval_start_vars=interval_start_vars, duration_vars=duration_vars, base_time=base_time,)
        else:
            # No solution: returns an empty calendar instead of reading the variable and causing an overflow.
            scheduled = []

        # Change the numerical status of OR-Tools to an easy-to-read label.
        if status == cp_model.OPTIMAL:
            status_label = "OPTIMAL"        # Best possible schedule, and the solver proved it.
        elif status == cp_model.FEASIBLE:
            status_label = "FEASIBLE"       # Valid schedule, but the time limit ran out before proving it is the best.
        elif status == cp_model.INFEASIBLE:
            status_label = "INFEASIBLE"     # The constraints contradict each other: no valid schedule exists.
        else:
            status_label = "UNKNOWN"        # Time limit reached before any solution was found.

        return ScheduleResult(
            status=status_label,
            # Sum of the values of the scheduled observations.
            objective_value=sum(observation.value for observation in scheduled),
            runtime_seconds=runtime_seconds,
            scheduled=scheduled,
            solver_status=solver.status_name(status),
            )


    def _add_resource_constraint(self, model: cp_model.CpModel, opportunity_vars: dict[str, cp_model.IntVar], resource_column: str, capacity_column: str, scale: int,) -> None:

        """
        Add resource constraints for each satellite.

        Applicable to:
            - energy_wh <= battery_capacity_wh
            - data_volume_gb <= memory_capacity_gb

        General formula:
            sum(resource_i * x_i) <= capacity

        Parameters:
            resource_column: the resource consumption column in the request.
            capacity_column: the resource capacity column in the satellite.
            scale: a multiplier to convert decimal numbers to integers.
        """

        # Group opportunities by satellite.
        opportunities_by_satellite: dict[str, list] = defaultdict(list)

        for opportunity in self.opportunities.itertuples(index=False):
            opportunities_by_satellite[opportunity.satellite_id].append(opportunity)

        # For each satellite, add a separate resource constraint.
        for satellite_id, opportunities in opportunities_by_satellite.items():
            satellite = self.satellite_lookup[satellite_id]     # Look up the satellite.

            terms = []          # Variables x_i of this satellite.
            coefficients = []   # Matching scaled resource amounts resource_i.

            for opportunity in opportunities:
                if opportunity.opportunity_id not in opportunity_vars:     # Filtered out of the model.
                    continue

                request = self.request_lookup[opportunity.request_id]   # Look up the corresponding request.

                # Add:
                #   resource_i * x_i
                # If x_i = 0, the contribution is 0.
                # If x_i = 1, the contribution is resource_i.
                #
                # Consumption is rounded UP (conservative) so the real usage never exceeds capacity.
                # The small epsilon avoids float noise like 18.3 * 10 = 183.00000000000003 -> 184.
                terms.append(opportunity_vars[opportunity.opportunity_id])
                coefficients.append(int(math.ceil(request[resource_column] * scale - 1e-9)))

            if not terms:       # This satellite has no variable left in the model -> no constraint needed.
                continue

            # Maximum capacity of the satellite, also multiplied by the scale factor to match the units of the terms.
            # Capacity is rounded DOWN (conservative).
            capacity = int(math.floor(satellite[capacity_column] * scale + 1e-9))

            # Constraints:
            #   sum(resource_i * x_i) <= capacity
            model.add(cp_model.LinearExpr.weighted_sum(terms, coefficients) <= capacity)


    def _add_orbit_capacity_constraints(self, model: cp_model.CpModel, opportunity_vars: dict[str, cp_model.IntVar],) -> None:

        """
        Limits the number of observations per orbit.

        Formula:
            For each satellite s and orbit block b:
                sum(x_i) <= max_observations_per_orbit_s

        The orbit block is calculated as:
            block_id = floor((window_start - base_time) / orbit_period)
        """

        base_time = self.opportunities["window_start"].min()    # Timeline 0.

        # Group opportunities by satellite.
        opportunities_by_satellite: dict[str, list] = defaultdict(list)     

        for opportunity in self.opportunities.itertuples(index=False):
            opportunities_by_satellite[opportunity.satellite_id].append(opportunity)

        for satellite_id, opportunities in opportunities_by_satellite.items():
            satellite = self.satellite_lookup[satellite_id]     # Each satellite has its own orbit period and per-orbit limit.

            # Convert the orbital period from minutes to seconds.
            # Formula:
            #   orbit_seconds = orbit_period_minutes * 60
            orbit_seconds = satellite["orbit_period_minutes"] * 60

            blocks: dict[int, list[cp_model.IntVar]] = defaultdict(list)       # Group opportunities by orbit block.

            for opportunity in opportunities:
                if opportunity.opportunity_id not in opportunity_vars:     # Filtered out of the model.
                    continue

                # The time of window_start relative to base_time.
                offset_seconds = int((opportunity.window_start - base_time).total_seconds())

                # Determine which orbit block an opportunity belongs to.
                # Formula:
                #   block_id =
                #       floor(offset_seconds / orbit_seconds)
                # Ex:
                #   orbit_seconds = 5,760 seconds = 96 minutes
                #   offset_seconds = 7,000
                #   block_id = floor(7000 / 5760) = 1
                #
                # This means this opportunity belongs to the second orbit if counting from the first block
                block_id = int(offset_seconds // orbit_seconds)

                blocks[block_id].append(opportunity_vars[opportunity.opportunity_id])     # Place the variable x from the “opportunity” block in the correct block.

            # For each orbit block:
            #   The total number of selected tasks does not exceed the limit.
            for variables in blocks.values():
                model.add(cp_model.LinearExpr.sum(variables) <= int(satellite["max_observations_per_orbit"]))


    def _add_greedy_hint(self, model: cp_model.CpModel, opportunity_vars: dict[str, cp_model.IntVar], interval_start_vars: dict[str, cp_model.IntVar], interval_end_vars: dict[str, cp_model.IntVar], duration_vars: dict[str, int], start_bounds: dict[str, tuple[int, int]], slew_buffer: int,) -> None:
        """
        Warm-start CP-SAT with the best greedy schedule.

        Steps:
            1. Run GreedyScheduler and ImprovedGreedyScheduler on the same data (fast, ~1 s) and keep the better one.
            2. Re-place its captures on the integer grid of THIS model
               (rounded-up durations, ceil'd window starts), satellite by satellite,
               so the hint is valid for the CP-SAT constraints.
            3. model.add_hint(x, 1/0) for every opportunity, and add_hint(start, value) for selected ones.

        Re-placement per satellite, captures sorted by greedy start time:
            start_k = max(earliest_start_k, previous_end + slew_buffer)
            keep it only if start_k <= latest_start_k
        Ex:
            previous capture: start = 700, duration = 10   -> previous_end = 710
            buffer = 30, next window earliest = 720         -> start = max(720, 740) = 740
        """
        # Run both greedy variants and keep the better one.
        # (The improved greedy is usually better, but greedy is not monotone: on some data the baseline wins.)
        # List comprehension: build and solve one scheduler per class -> [greedy_result, improved_result].
        greedy_results = [
            scheduler_class.from_dataframes(
                satellites=self.satellites,
                requests=self.requests,
                opportunities=self.opportunities,
                slew_buffer_seconds=self.slew_buffer_seconds,
            ).solve()
            for scheduler_class in (GreedyScheduler, ImprovedGreedyScheduler)
        ]
        greedy_result = max(greedy_results, key=lambda result: result.objective_value)     # Keep the one with the higher total value.

        # Group the greedy observations by satellite and sort them by start time.
        observations_by_satellite: dict[str, list] = defaultdict(list)
        for observation in greedy_result.scheduled:
            if observation.opportunity_id in opportunity_vars:      # Only opportunities that exist in this model.
                observations_by_satellite[observation.satellite_id].append(observation)

        selected_starts: dict[str, int] = {}    # opportunity_id -> integer start used in the hint.

        for observations in observations_by_satellite.values():
            observations.sort(key=lambda observation: observation.start)    # Chronological order on this satellite.

            previous_end = None     # End (in model seconds) of the last capture kept on this satellite.

            for observation in observations:
                earliest, latest = start_bounds[observation.opportunity_id]     # Allowed start range of this capture in the model (seconds).

                # Earliest start that respects both the window and the slew buffer after the previous capture.
                start = earliest if previous_end is None else max(earliest, previous_end + slew_buffer)

                if start > latest:      # Does not fit after rounding -> leave it out of the hint.
                    continue

                selected_starts[observation.opportunity_id] = start                 # Keep this capture in the hint.
                previous_end = start + duration_vars[observation.opportunity_id]    # Formula: previous_end = start + duration

        # Hint EVERY variable (a complete hint is used directly as the first solution;
        # an incomplete one must be "completed" by the solver, which may fail on large models):
        #   selected     -> x = 1, start = re-placed start
        #   not selected -> x = 0, start = earliest start (any value in its domain is fine)
        #   end = start + duration + slew_buffer   (consistent with the interval)
        for opportunity_id, is_selected in opportunity_vars.items():
            selected = opportunity_id in selected_starts        # True if the greedy schedule kept this opportunity.
            start = selected_starts[opportunity_id] if selected else start_bounds[opportunity_id][0]

            # model.add_hint(variable, value): "try this value first" (a suggestion, not a constraint).
            model.add_hint(is_selected, 1 if selected else 0)
            model.add_hint(interval_start_vars[opportunity_id], start)
            model.add_hint(interval_end_vars[opportunity_id], start + duration_vars[opportunity_id] + slew_buffer)


    def _extract_schedule(self, solver: cp_model.CpSolver, opportunity_vars: dict[str, cp_model.IntVar], interval_start_vars: dict[str, cp_model.IntVar], duration_vars: dict[str, int], base_time: pd.Timestamp,) -> list[ScheduledObservation]:
        """
        After the solver finishes running,
        read the x variables and create a list of the selected iterations.
        """

        scheduled = []      # The selected captures, as ScheduledObservation objects.

        for opportunity in self.opportunities.itertuples(index=False):
            variable = opportunity_vars.get(opportunity.opportunity_id)     # Retrieve the corresponding variable x.

            # If the opportunity has no variables, that means it has been removed from the model.
            if variable is None:
                continue

            # solver.value(variable) returns the value after solving.
            # If x = 0: the opportunity is not selected.
            # If x = 1: the opportunity is selected.
            if solver.value(variable) != 1:
                continue

            request = self.request_lookup[opportunity.request_id]   # Look up the corresponding request.

            # Read the start value that CP-SAT selected for this opportunity.
            # interval_start is the number of seconds from base_time.
            # Example:
            #   base_time = 08:00:00
            #   solver.value(interval_start) = 735
            #   start = 08:00:00 + 735 seconds
            #         = 08:12:15
            start_seconds = solver.value(interval_start_vars[opportunity.opportunity_id])

            # Start time chosen by the solver (a decision variable inside the window).
            start = base_time + timedelta(seconds=start_seconds)

            # Use the same (rounded-up) duration as in the model, so end <= window_end is guaranteed.
            # The slew buffer is NOT part of the observation; it is only idle time on the timeline.
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

    # Mean:
    # start can be 720, 721, …, 740
    # end = start + 10
    # end will be in the range [730, 750]

    # if solver pick the opportunity in the request
    # x = 1
    # start = 735

    # so
    # end = 735 + 10 = 745

    # started at 08:12:15
    # ended at 08:12:25

    # Both still fall in the period 08:12:00–08:12:30 window.

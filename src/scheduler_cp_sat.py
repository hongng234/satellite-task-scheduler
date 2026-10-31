from __future__ import annotations

import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

from src.models import (Priority, ScheduleResult, ScheduledObservation,)


class SatelliteTaskingScheduler:
    def __init__(self, satellites_path: Path, requests_path: Path, opportunities_path: Path, slew_buffer_seconds: float=30.0, time_limit_seconds: float=30.0, num_workers: int=8,) -> None:
        self.satellites = pd.read_parquet(satellites_path)
        self.requests = pd.read_parquet(requests_path)
        self.opportunities = pd.read_parquet(opportunities_path)

        self.opportunities = self.opportunities[self.opportunities["feasible"]].copy()

        self.satellite_lookup = self.satellites.set_index("satellite_id").to_dict("index")
        self.request_lookup = self.requests.set_index("request_id").to_dict("index")

        self.slew_buffer_seconds = slew_buffer_seconds
        self.time_limite_seconds = time_limit_seconds
        self.num_workers = num_workers

        self._validate_inputs()


    def _validate_inputs(self) -> None:
        unknow_requests = set(self.opportunities["request_id"]) - set(self.requests["request_id"])
        unknow_satellites = set(self.opportunities["satellite_id"]) - set(self.satellites["satellite_id"])

        if unknow_requests:
            raise ValueError(f"Opportunities refer to unknow requests: {unknow_requests}")
        
        if unknow_satellites:
            raise ValueError(f"Oppotunities refer to unknow satellites: {unknow_satellites}")


    def solve(self) -> ScheduleResult:
        started = time.perf_counter()

        model = cp_model.CpModel()

        base_time = self.opportunities["window_start"].min()

        opportunity_vars: dict[str, cp_model.IntVar] = {}

        intervals_by_satellite: dict[str, list[cp_model.IntervalVar]] = defaultdict(list)

        objective_terms = []

        for opportunity in self.opportunities.itertuples(index=False):
            request = self.request_lookup[opportunity.request_id]

            window_start_seconds = int((opportunity.window_start - base_time).total_seconds())
            window_end_seconds = int((opportunity.window_end - base_time).total_seconds())

            duration_seconds = int(round(opportunity.duration_seconds))

            latest_start_seconds = (window_end_seconds - duration_seconds)

            if latest_start_seconds < window_start_seconds:
                continue

            is_selected = model.new_bool_var(f"select_{opportunity.opportunity_id}")

            interval_start = model.new_int_var(window_start_seconds, latest_start_seconds, f"start_{opportunity.opportunity_id}",)
            interval_end = model.new_int_var(window_start_seconds, window_end_seconds, f"end_{opportunity.opportunity_id}",)

            interval = model.new_interval_var(start=interval_start, size=duration_seconds, end=interval_end, name=f"interval_{opportunity.opportunity_id}",)

            opportunity_vars[opportunity.opportunity_id] = is_selected

            intervals_by_satellite[opportunity.satellite_id].append(interval)

            objective_terms.append(int(round(request["value"])) * is_selected)


        # A satellite cannot perform overlapping observations.
        for satellite_id, intervals in intervals_by_satellite.items():
            if intervals:
                model.add_no_overlap(intervals)


        # Each request can be served at most once.
        opportunities_by_request: dict[str, list[cp_model.IntVar]] = defaultdict(list)

        for opportunity in self.opportunities.itertuples(index=False):
            opportunities_by_request[opportunity.request_id].append(opportunity_vars[opportunity.opportunity_id])

        for variables in opportunities_by_request.values():
            model.add_at_most_one(variables)


        # Battery constraint.
        self._add_resource_constraint(model=model, opportunity_vars=opportunity_vars, resource_column="energy_wh", capacity_column="battery_capacity_wh", scale=1,)

        # Memory constraint.
        self._add_resource_constraint(model=model, opportunity_vars=opportunity_vars, resource_column="data_volume_gb", capacity_column="memory_capacity_gb", scale=10,)

        # Max observations per orbit.
        self._add_orbit_capacity_constraints(model=model, opportunity_vars=opportunity_vars,)

        model.maximize(sum(objective_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limite_seconds
        solver.parameters.num_search_workers = self.num_workers
        solver.parameters.log_search_progress = False

        status = solver.solve(model)

        runtime_seconds = time.perf_counter() - started

        scheduled = self._extract_schedule(solver=solver, opportunity_vars=opportunity_vars, base_time=base_time,)

        if status == cp_model.OPTIMAL:
            status_label = "OPTIMAL"
        elif status == cp_model.FEASIBLE:
            status_label = "FEASIBLE"
        elif status == cp_model.INFEASIBLE:
            status_label = "INFEASIBLE"
        else:
            status_label = "UNKNOWN"

        return ScheduleResult(
            status=status_label,
            objective_value=sum(observation.value for observation in scheduled),
            runtime_seconds=runtime_seconds,
            scheduled=scheduled,
            solver_status=solver.status_name(status),
            )


    def _add_resource_constraint(self, model: cp_model.CpModel, opportunity_vars: dict[str, cp_model.IntVar], resource_column: str, capacity_column: str, scale: int,) -> None:
        opportunities_by_satellite: dict[str, list] = defaultdict(list)

        for opportunity in self.opportunities.itertuples(index=False):
            opportunities_by_satellite[opportunity.satellite_id].append(opportunity)

        for satellite_id, opportunities in opportunities_by_satellite.items():
            satellite = self.satellite_lookup[satellite_id]

            terms = []

            for opportunity in opportunities:
                request = self.request_lookup[opportunity.request_id]

                terms.append(int(round(request[resource_column] * scale)) * opportunity_vars[opportunity.opportunity_id])

            capacity = int(round(satellite[capacity_column] * scale))

            model.add(sum(terms) <= capacity)


    def _add_orbit_capacity_constraints(self, model: cp_model.CpModel, opportunity_vars: dict[str, cp_model.IntVar],) -> None:
        base_time = self.opportunities["window_start"].min()

        opportunities_by_satellite: dict[str, list] = defaultdict(list)

        for opportunity in self.opportunities.itertuples(index=False):
            opportunities_by_satellite[opportunity.satellite_id].append(opportunity)

        for satellite_id, opportunities in opportunities_by_satellite.items():
            satellite = self.satellite_lookup[satellite_id]

            orbit_seconds = satellite["orbit_period_minutes"] * 60

            blocks: dict[int, list[cp_model.IntVar]] = defaultdict(list)

            for opportunity in opportunities:
                offset_seconds = int((opportunity.window_start - base_time).total_seconds())

                block_id = int(offset_seconds // orbit_seconds)

                blocks[block_id].append(opportunity_vars[opportunity.opportunity_id])

            for variables in blocks.values():
                model.add(sum(variables) <= int(satellite["max_observations_per_orbit"]))


    def _extract_schedule(self, solver: cp_model.CpSolver, opportunity_vars: dict[str, cp_model.IntVar], base_time: pd.Timestamp,) -> list[ScheduledObservation]:
        scheduled = []

        for opportunity in self.opportunities.itertuples(index=False):
            variable = opportunity_vars.get(opportunity.opportunity_id)

            if variable is None:
                continue

            if solver.value(variable) != 1:
                continue

            request = self.request_lookup[opportunity.request_id]

            start = base_time + timedelta(seconds=solver.value(opportunity_vars[opportunity.opportunity_id]))

            # In this simplified model, start is the decision variable, but I use the window start for a stable readable schedule.
            start = opportunity.window_start
            end = start + timedelta(seconds=opportunity.duration_seconds)

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

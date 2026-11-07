from __future__ import annotations     # Lets type hints refer to classes defined later in the file (e.g. list[ScheduledObservation]).

from datetime import datetime           # Date + time type used for every timestamp (window_start, earliest_start, ...).
from enum import IntEnum                # Enum whose members are also integers (Priority.HIGH == 3).

from pydantic import BaseModel, Field, field_validator  # Pydantic: classes that check the type and value of every field when an object is created.


## Key point: this file only DESCRIBES the data. It contains no scheduling logic.
## Every table of the dataset (satellites, requests, opportunities) has one model here,
## and every scheduler returns the same output model (ScheduleResult), so the results are easy to compare.
##
## How a Pydantic model works:
##   class Satellite(BaseModel):
##       battery_capacity_wh: float = Field(gt=0)
##
##   Satellite(battery_capacity_wh=1200)  -> OK
##   Satellite(battery_capacity_wh=-5)    -> raises ValidationError (gt=0 means "greater than 0")
##   Satellite(battery_capacity_wh="abc") -> raises ValidationError (not a number)
##
##   Field(gt=0) -> value must be  > 0
##   Field(ge=0) -> value must be >= 0


# Importance level of a request. Higher number = more important.
# IntEnum, so a priority can be compared and sorted like a number:
#   Priority.CRITICAL > Priority.LOW   -> True
#   Priority(3)                        -> Priority.HIGH
class Priority(IntEnum):
    LOW = 1             # Ordinary request.
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4        # Most urgent request (ex: disaster monitoring). The greedy schedulers handle these first.


# One satellite of the constellation (one row of satellites.parquet).
# Ex:
#   Satellite(satellite_id="SAT-01", orbit_period_minutes=96.4, max_observations_per_orbit=8,
#             battery_capacity_wh=1200.5, memory_capacity_gb=64.0, slew_rate_deg_per_sec=1.2, initial_phase_deg=45.0)
class Satellite(BaseModel):
    satellite_id: str                               # Unique id. Ex: "SAT-01"
    orbit_period_minutes: float = Field(gt=0)       # Time for one full orbit around the Earth, in minutes. Ex: 96 min
    max_observations_per_orbit: int = Field(gt=0)   # Max number of captures the satellite may take during one orbit. Ex: 8
    battery_capacity_wh: float = Field(gt=0)        # Energy budget for the whole horizon, in watt-hours (no recharge in this model). Ex: 1200 Wh
    memory_capacity_gb: float = Field(gt=0)         # On-board storage for the whole horizon, in GB (no downlink to the ground in this model). Ex: 64 GB
    slew_rate_deg_per_sec: float = Field(gt=0)      # How fast the satellite rotates to point at a new target. Ex: 1.5 deg/s -> turning 30 deg takes 20 s.
                                                    # Not used by the schedulers yet: they use a fixed slew buffer (30 s) instead.
    initial_phase_deg: float = 0.0                  # Position of the satellite on its orbit at start_time, in degrees (0..360). Default 0.


# One customer request: "take a picture of this area, between these two times" (one row of requests.parquet).
# Ex:
#   Request(request_id="REQ-0001", aoi_id="AOI-042", priority=Priority.HIGH, value=120.5,
#           earliest_start=2026-01-01 02:00, latest_end=2026-01-01 12:00,
#           duration_seconds=10.0, data_volume_gb=1.5, energy_wh=18.3)
class Request(BaseModel):
    request_id: str                                 # Unique id. Ex: "REQ-0001"
    aoi_id: str                                     # Area Of Interest (the place on Earth to photograph). One AOI can have several requests. Ex: "AOI-042"
    priority: Priority                              # LOW / MEDIUM / HIGH / CRITICAL (see Priority above).
    value: float = Field(gt=0)                      # Reward earned if this request is served. The schedulers maximize the sum of these values.
    earliest_start: datetime                        # The capture may not start before this time.
    latest_end: datetime                            # Deadline: the capture must be finished by this time.
    duration_seconds: float = Field(gt=0)           # Length of one capture, in seconds. Ex: 10.0 s
    data_volume_gb: float = Field(ge=0)             # Memory used by one capture, in GB. Ex: 1.5 GB
    energy_wh: float = Field(gt=0)                  # Battery energy used by one capture, in Wh. Ex: 18.3 Wh

    # Extra rule checked when a Request is created: the time range must have a positive length.
    # Formula:
    #   latest_end > earliest_start
    # Ex:
    #   earliest_start = 02:00, latest_end = 12:00 -> OK
    #   earliest_start = 12:00, latest_end = 02:00 -> ValidationError
    #
    # @field_validator("latest_end") -> Pydantic calls this function with the value of latest_end.
    # @classmethod                   -> the function receives the class (cls), not an object (self).
    @field_validator("latest_end")
    @classmethod
    def validate_time_window(cls, v:datetime, info):
        # info.data holds the fields validated so far. earliest_start is declared ABOVE latest_end,
        # so it is already there. .get() returns None if earliest_start itself was invalid.
        earliest = info.data.get("earliest_start")

        # Only compare when earliest_start is known. "v <= earliest" means the range is empty or reversed.
        if earliest and v <= earliest:
            raise ValueError("latest_end must be after earliest_start")

        return v        # A validator must return the (possibly modified) value; here it is unchanged.


# One possible capture: "satellite S can photograph request R during this time window"
# (one row of opportunities.parquet). The scheduler picks some of these opportunities.
# Ex:
#   Opportunity(opportunity_id="OPP-000001", request_id="REQ-0001", satellite_id="SAT-03",
#               window_start=08:12:00, window_end=08:12:30, duration_seconds=10.0,
#               incidence_angle_deg=32.1, off_nadir_deg=12.4)
#   -> SAT-03 may take the 10 s capture of REQ-0001 at any start between 08:12:00 and 08:12:20.
class Opportunity(BaseModel):
    opportunity_id: str                 # Unique id. Ex: "OPP-000001"
    request_id: str                     # Which request this capture would serve.
    satellite_id: str                   # Which satellite would take it.
    window_start: datetime              # Earliest moment the satellite can see the AOI on this pass.
    window_end: datetime                # Last moment the satellite can see the AOI on this pass.
    duration_seconds: float             # Length of the capture (copied from the request), in seconds.
    incidence_angle_deg: float          # Angle between the line of sight and the ground's vertical at the AOI.
                                        # Too large -> distorted image, shadows, layover. Informational only (not used by the schedulers).
    off_nadir_deg: float                # How far the satellite must tilt away from looking straight down (nadir).
                                        # Too large -> poor image quality or impossible capture. Informational only.
    feasible: bool = True               # False = this opportunity cannot be used. The schedulers drop rows with feasible = False.

    # Same rule as Request: the window must have a positive length.
    # Formula:
    #   window_end > window_start
    # Ex:
    #   window_start = 08:12:00, window_end = 08:12:30 -> OK
    #   window_start = 08:12:30, window_end = 08:12:00 -> ValidationError
    @field_validator("window_end")
    @classmethod
    def validate_window(cls, v: datetime, info):
        start = info.data.get("window_start")     # Fields already validated (declared above window_end).

        # Reject an empty or reversed window.
        if start and v <= start:
            raise ValueError("window_end must be after window_start")

        return v


# One capture chosen by a scheduler (one row of the output schedule).
# Ex:
#   ScheduledObservation(request_id="REQ-0001", satellite_id="SAT-03", opportunity_id="OPP-000001",
#                        start=08:12:05, end=08:12:15, value=120.5, priority=Priority.HIGH)
class ScheduledObservation(BaseModel):
    request_id: str             # Request served by this capture.
    satellite_id: str           # Satellite that takes the capture.
    opportunity_id: str         # Opportunity (window) that was used.
    start: datetime             # Actual capture start, chosen by the scheduler inside the window.
    end: datetime               # Actual capture end. Formula: end = start + duration (slew buffer not included).
    value: float                # Value of the served request (copied from the request).
    priority: Priority          # Priority of the served request (copied from the request).


# Output of every scheduler (CP-SAT, greedy, improved greedy) -> same format, easy to compare.
# Ex:
#   ScheduleResult(status="OPTIMAL", objective_value=2941.52, runtime_seconds=0.04,
#                  scheduled=[...46 ScheduledObservation...], solver_status="OPTIMAL")
class ScheduleResult(BaseModel):
    status: str                             # Our label: OPTIMAL / FEASIBLE / INFEASIBLE / UNKNOWN.
                                            #   OPTIMAL  = proven best schedule; FEASIBLE = valid but maybe not the best.
    objective_value: float                  # Score of the schedule. Formula: objective = sum(value) of the scheduled observations.
    runtime_seconds: float                  # Wall-clock time of solve(), in seconds.
    scheduled: list[ScheduledObservation]   # The chosen captures (one per served request).
    solver_status: str                      # Raw status from the solver, or "GREEDY_HEURISTIC" for the greedy schedulers,
                                            # or "NO_OPPORTUNITIES" when there was nothing to schedule.

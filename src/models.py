from __future__ import annotations

from datetime import datetime
from enum import IntEnum

from pydantic import BaseModel, Field, field_validator



class Priority(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Satellite(BaseModel):
    satellite_id: str
    orbit_period_minutes: float = Field(gt=0)       # time it takes for the satellite to complete one Earth orbit in minutes
    max_observations_per_orbit: int = Field(gt=0)   # maximum times the satellite can capture in one orbit 
    battery_capacity_wh: float = Field(gt=0)        # maximum energy storage capacity of the satellite, regenerate by solar panels
    memory_capacity_gb: float = Field(gt=0)         # maximum data storage capacity of the satellite in GB not transferred to ground station
    slew_rate_deg_per_sec: float = Field(gt=0)      # rotation rate of the satellite, antenna adjustment, or look angle ex: 1.5 deg/sec, slew time = angle / slew_rate
    initial_phase_deg: float = 0.0                  # initial position of the satellite in its orbit, measured in degrees


class Request(BaseModel):
    request_id: str
    aoi_id: str                                     # area of interest id, 1 aoi can have multiple requests
    priority: Priority
    value: float = Field(gt=0)                      # value of the request
    earliest_start: datetime                        # earliest time of the customer request
    latest_end: datetime                            # deadline of the customer request
    duration_seconds: float = Field(gt=0)           # duration of 1 capture in seconds
    data_volume_gb: float = Field(ge=0)             # data volume in GB
    energy_wh: float = Field(gt=0)                  # energy consumption of the 1 capture in Wh

    @field_validator("latest_end")
    @classmethod
    def validate_time_window(cls, v:datetime, info):
        earliest = info.data.get("earliest_start")

        if earliest and v <= earliest:
            raise ValueError("latest_end must be after earliest_start")

        return v


class Opportunity(BaseModel):
    opportunity_id: str
    request_id: str
    satellite_id: str
    window_start: datetime  # start time of the opportunity window of the AOI
    window_end: datetime    # end time of the opportunity window of the AOI
    duration_seconds: float # time capture of the request of the opportunity in seconds
    incidence_angle_deg: float # angle between the satellite and the AOI normal ground ( too big or too small may cause image distortion, shadowing, layover,...)
    off_nadir_deg: float #  The angle of deviation between the satellite’s line of sight and the vertical direction, If the off-nadir angle is too large, the satellite may not be able to capture an image, or the image quality may be poor.
    feasible: bool = True   # opportunity can be used or not


class ScheduledObservation(BaseModel):
    request_id: str
    satellite_id: str
    opportunity_id: str
    start: datetime
    end: datetime
    value: float
    priority: Priority


class Schedule(BaseModel):
    status: str
    objective_value: float
    runtime_seconds: float
    scheduled: list[ScheduledObservation]
    solver_status: str
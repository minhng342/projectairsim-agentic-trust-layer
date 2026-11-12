"""Small fakes shared by the observation-channel tests (no simulator)."""
import math
from datetime import datetime, timezone

from atal.core.telemetry import (
    GroundState,
    Quaternion,
    TelemetrySnapshot,
    ValidationStatus,
    Vector3,
)


class ListRecorder:
    """Stands in for EvaluationRecorder: keeps events in memory."""

    def __init__(self):
        self.events = []

    def record(self, kind, **fields):
        self.events.append({"kind": kind, **fields})
        return len(self.events) - 1

    def of(self, kind):
        return [e for e in self.events if e["kind"] == kind]


def snapshot(vid="Drone1", n=1.0, e=2.0, alt=6.0, t_ns=10_000_000_000,
             status=ValidationStatus.VALID, errors=()):
    return TelemetrySnapshot(
        vehicle_id=vid, sim_time_ns=t_ns, received_at_utc=datetime.now(timezone.utc),
        position_ned_m=Vector3(x=n, y=e, z=-alt),
        orientation_quaternion=Quaternion(w=1, x=0, y=0, z=0),
        velocity_ned_mps=Vector3(x=0, y=0, z=0), acceleration_ned_mps2=Vector3(x=0, y=0, z=0),
        angular_velocity_rad_s=Vector3(x=0, y=0, z=0), latitude_deg=47.6, longitude_deg=-122.1,
        altitude_msl_m=120 + alt, altitude_local_m=alt, heading_deg=0.0, ground_speed_mps=0.0,
        vertical_speed_mps=0.0, ground_state=GroundState.AIRBORNE, telemetry_age_ms=1.0,
        validation_status=status, validation_errors=list(errors))


class SnapAdapter:
    """get_snapshot() returns a fresh snapshot per call; sim time advances 0.1 s each call."""

    def __init__(self, ids=("Drone1", "Drone2")):
        self.vehicle_ids = list(ids)
        self.state = {v: {"n": 1.0 + i, "e": 2.0, "alt": 6.0} for i, v in enumerate(ids)}
        self.t_ns = 10_000_000_000
        self.status = ValidationStatus.VALID
        self.raises = False
        self.calls = 0

    def get_snapshot(self, vid):
        self.calls += 1
        if self.raises:
            raise RuntimeError("request timed out")
        self.t_ns += 100_000_000
        s = self.state[vid]
        errors = () if self.status == ValidationStatus.VALID else ("simulated stale",)
        return snapshot(vid, s["n"], s["e"], s["alt"], self.t_ns, self.status, errors)


def finite(x):
    return isinstance(x, float) and math.isfinite(x)

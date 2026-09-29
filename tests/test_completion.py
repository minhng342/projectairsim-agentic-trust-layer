"""Pure completion predicate tests (no simulator)."""
from datetime import datetime, timezone

import pytest

from executor.completion import (Deadline, DwellTracker, altitude_settled, grounded,
                                 heading_error_deg, heading_settled, position_settled,
                                 track_velocity_ok)
from models.telemetry import (GroundState, Quaternion, TelemetrySnapshot, ValidationStatus,
                              Vector3)


def snap(n=0.0, e=0.0, alt=10.0, vn=0.0, ve=0.0, vs=0.0, heading=90.0, track=None,
         status=ValidationStatus.VALID, ground=GroundState.AIRBORNE, t_ns=1_000_000_000):
    import math
    return TelemetrySnapshot(
        vehicle_id="Drone1", sim_time_ns=t_ns, received_at_utc=datetime.now(timezone.utc),
        position_ned_m=Vector3(x=n, y=e, z=-alt),
        orientation_quaternion=Quaternion(w=1, x=0, y=0, z=0),
        velocity_ned_mps=Vector3(x=vn, y=ve, z=-vs),
        acceleration_ned_mps2=Vector3(x=0, y=0, z=0), angular_velocity_rad_s=Vector3(x=0, y=0, z=0),
        latitude_deg=47.6, longitude_deg=-122.1, altitude_msl_m=120 + alt, altitude_local_m=alt,
        heading_deg=heading, track_deg=track, ground_speed_mps=math.hypot(vn, ve),
        vertical_speed_mps=vs, ground_state=ground, telemetry_age_ms=1.0,
        validation_status=status, validation_errors=[] if status == ValidationStatus.VALID else ["x"])


@pytest.mark.parametrize("target,actual,expected", [
    (90, 88, 2), (88, 90, -2), (0, 359, 1), (359, 0, -1), (180, 0, 180), (0, 180, 180),
    (10, 350, 20), (350, 10, -20), (315, 88.4, -133.4),
])
def test_heading_error_wraps(target, actual, expected):
    assert heading_error_deg(target, actual) == pytest.approx(expected)


def test_heading_settled_checks_error_and_drift():
    assert heading_settled(snap(heading=88.4), 90, 0, 0)[0]            # 1.6 deg: ok
    assert not heading_settled(snap(heading=85), 90, 0, 0)[0]          # 5 deg: no
    assert not heading_settled(snap(heading=90, n=0.6), 90, 0, 0)[0]   # drifted 0.6 m
    assert heading_settled(snap(heading=359), 1, 0, 0)[0]              # wraps through north


def test_altitude_settled_needs_error_and_vertical_speed():
    assert altitude_settled(snap(alt=9.9), 10)[0]
    assert not altitude_settled(snap(alt=9.6), 10)[0]
    assert not altitude_settled(snap(alt=10.0, vs=0.5), 10)[0]         # passing through


def test_position_settled_rejects_the_pass_4_1_early_return():
    # Pass 4.1 live: "over launch point" at (-1.1, 8.0, 10.5) still moving 2.6 m/s
    moving = snap(n=-1.1, e=8.0, alt=10.5, vn=-2.2, ve=-1.4)
    assert not position_settled(moving, -1.0, 8.0, 10.0)[0]
    assert position_settled(snap(n=-1.1, e=8.1, alt=10.2), -1.0, 8.0, 10.0)[0]


def test_track_velocity_ok_requires_heading_unchanged():
    # Pass 4.1 live: track 000 at 3 m/s while the nose stayed ~88 deg
    ok, detail = track_velocity_ok(snap(vn=3.0, heading=88.0, track=0.0), 0, 3.0, start_heading_deg=88.4)
    assert ok, detail
    assert not track_velocity_ok(snap(vn=3.0, heading=0.0, track=0.0), 0, 3.0, 88.4)[0]   # nose turned
    assert not track_velocity_ok(snap(vn=2.0, heading=88, track=0.0), 0, 3.0, 88.4)[0]    # too slow
    assert not track_velocity_ok(snap(ve=3.0, heading=88, track=90.0), 0, 3.0, 88.4)[0]   # wrong way
    assert not track_velocity_ok(snap(heading=88, track=None), 0, 3.0, 88.4)[0]           # not moving


def test_grounded_uses_operational_ground_state():
    assert grounded(snap(ground=GroundState.GROUNDED))[0]
    assert not grounded(snap(ground=GroundState.AIRBORNE))[0]


@pytest.mark.parametrize("check", [
    lambda s: heading_settled(s, 90, 0, 0),
    lambda s: altitude_settled(s, 10),
    lambda s: position_settled(s, 0, 0, 10),
    lambda s: grounded(s),
])
@pytest.mark.parametrize("status", [ValidationStatus.STALE, ValidationStatus.INVALID])
def test_non_valid_telemetry_never_confirms(check, status):
    s = snap(ground=GroundState.GROUNDED, status=status)
    ok, detail = check(s)
    assert not ok and status.value in detail


def test_dwell_tracker_requires_continuous_sim_time():
    d = DwellTracker(0.5)
    assert not d.update(1_000_000_000, True)
    assert not d.update(1_300_000_000, True)
    assert not d.update(1_400_000_000, False)       # interrupted: resets
    assert not d.update(1_500_000_000, True)
    assert not d.update(1_900_000_000, True)
    assert d.update(2_000_000_000, True)            # 0.5 s continuous
    assert d.held_s == pytest.approx(0.5)


def test_dwell_tracker_resets_when_sim_clock_goes_backwards():
    d = DwellTracker(0.5)
    d.update(5_000_000_000, True)
    assert not d.update(1_000_000_000, True)        # clock reset
    assert not d.update(1_400_000_000, True)
    assert d.update(1_500_000_000, True)


def test_deadline_uses_injected_clock():
    now = [100.0]
    dl = Deadline(2.0, clock=lambda: now[0])
    assert not dl.expired and dl.remaining_s == pytest.approx(2.0)
    now[0] = 102.0
    assert dl.expired and dl.remaining_s == 0.0

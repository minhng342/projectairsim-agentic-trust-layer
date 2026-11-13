"""The offline pose-skew test fault: only the first long descent of a listed drone, only
inside its window, and the snapshot is INVALID there like the live adapter's."""
import asyncio

from atal.core.telemetry import ValidationStatus
from atal.simulation import kinematic_sim as ks
from atal.simulation.kinematic_sim import KinematicSim


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _sim(**kw):
    clock = Clock()
    sim = KinematicSim({"Drone1": (0.0, 0.0), "Drone2": (0.0, 3.0)}, speedup=1.0,
                       clock=clock, **kw)
    for v in sim._v.values():
        v.alt, v.grounded = 160.0, False
    return sim, clock


def _status(sim, clock, vid, t):
    clock.t = t
    return sim.get_snapshot(vid).validation_status


def _descend(sim, vid, to_m):
    async def go():
        await sim.drone(vid).move_to_position_async(0.0, 0.0, -to_m, 1.5)
    asyncio.run(go())


def test_the_first_long_descent_gets_one_invalid_window():
    sim, clock = _sim(pose_skew_on_descent=["Drone2"])
    _descend(sim, "Drone2", 1.5)
    t0 = ks.POSE_SKEW_AFTER_S
    assert _status(sim, clock, "Drone2", t0 - 0.1) == ValidationStatus.VALID
    snap_t = t0 + ks.POSE_SKEW_FOR_S / 2
    assert _status(sim, clock, "Drone2", snap_t) == ValidationStatus.INVALID
    assert "pose topic and kinematics differ by 732 ms" in " ".join(
        sim.get_snapshot("Drone2").validation_errors)
    assert _status(sim, clock, "Drone2", t0 + ks.POSE_SKEW_FOR_S + 0.1) == ValidationStatus.VALID
    assert _status(sim, clock, "Drone1", snap_t) == ValidationStatus.VALID
    assert len(sim.pose_skews) == 1


def test_short_descents_and_unlisted_drones_get_no_fault():
    sim, clock = _sim(pose_skew_on_descent=["Drone2"])
    _descend(sim, "Drone2", 150.0)                      # a 10 m recovery-sized descent
    _descend(sim, "Drone1", 1.5)                        # not listed
    for t in (1.0, 1.7, 2.0, 3.0):
        assert _status(sim, clock, "Drone2", t) == ValidationStatus.VALID
        assert _status(sim, clock, "Drone1", t) == ValidationStatus.VALID
    assert sim.pose_skews == []


def test_without_the_option_nothing_is_injected():
    sim, clock = _sim()
    _descend(sim, "Drone2", 1.5)
    assert _status(sim, clock, "Drone2", 1.7) == ValidationStatus.VALID

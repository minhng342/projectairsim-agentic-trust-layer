"""How often things happen in a run, and how those rates scale when the offline simulator
runs faster than real time."""
from atal.flight.executor import ExecutorConfig
from atal.flight.path_follower import CORNER_SLOWDOWN, FollowerConfig

OBSERVATION_HZ = 1.0          # the planner decides at sector boundaries, not every second
SENSOR_HZ = 20.0
TRUTH_HZ = 2.0
REMOTE_ID_HZ = 1.0
MAX_OBSERVATION_AGE_S = 5.0
DESCEND_ABOVE_LAUNCH_M = 1.5
SPAWN_TOLERANCE_M = 5.0
MAX_HOST_HZ = 250.0           # offline: cap host-side polling rates
PAD_WAIT_S = 600.0            # sim seconds a drone waits for the pad zone before landing outside it
SIM_STALL_S = 30.0            # live: host seconds with no sim-clock progress before a run is ended


class Timing:
    """Host-side rates and polls for live (speedup 1) or the kinematic sim (speedup k)."""

    def __init__(self, speedup: float = 1.0, corner_slowdown: bool = False):
        self.k = speedup
        self.corner_slowdown = corner_slowdown

    def hz(self, sim_hz):
        return min(sim_hz * self.k, MAX_HOST_HZ) if self.k > 1 else sim_hz

    def poll(self, sim_s):
        return max(sim_s / self.k, 1.0 / MAX_HOST_HZ) if self.k > 1 else sim_s

    def follower(self) -> FollowerConfig:
        extra = CORNER_SLOWDOWN if self.corner_slowdown else {}
        return FollowerConfig(**extra) if self.k == 1 else FollowerConfig(
            poll_interval_s=self.poll(0.1), hover_timeout_s=5.0, **extra)

    def executor(self) -> ExecutorConfig:
        return ExecutorConfig() if self.k == 1 else ExecutorConfig(poll_interval_s=self.poll(0.1))

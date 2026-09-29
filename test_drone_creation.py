"""
test_drone_creation.py
Connects to a running Project AirSim sim, loads a scene, creates the drone(s)
defined in it, and prints ground-truth telemetry (the AirSim equivalent of
BlueSky's ACDATA stream in the reference repo).

Run from client/python/example_user_scripts (so sim_config/ resolves), with the
sim already running.
"""
import asyncio
import math

import os

from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log

# Sample scene/robot configs shipped with Project AirSim (resolved from this file's location)
SIM_CONFIG = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "ProjectAirSim-v1.0.1", "client", "python",
    "example_user_scripts", "sim_config")) + os.sep
SCENE = "scene_basic_drone.jsonc"   # swap for your multi-drone scene
DRONE_NAMES = ["Drone1"]            # must match actor names in the scene


def summarize(name: str, kin: dict) -> str:
    p = kin["pose"]["position"]
    v = kin["twist"]["linear"]
    gs = math.hypot(v["x"], v["y"])
    trk = (math.degrees(math.atan2(v["y"], v["x"])) + 360) % 360
    # NED frame: z is down, so altitude = -z
    return (f"{name}: N={p['x']:.2f} E={p['y']:.2f} alt={-p['z']:.2f} m | "
            f"gs={gs:.2f} m/s trk={trk:.0f} deg vs={-v['z']:.2f} m/s")


async def main():
    client = ProjectAirSimClient()
    try:
        client.connect()
        projectairsim_log().info("Connected to Project AirSim")

        world = World(client, SCENE, delay_after_load_sec=2, sim_config_path=SIM_CONFIG)
        drones = {n: Drone(client, world, n) for n in DRONE_NAMES}
        projectairsim_log().info(f"Created drones: {list(drones)}")

        for name, d in drones.items():
            print(summarize(name, d.get_ground_truth_kinematics()))

        # Optional: streamed pose topic (push-based, like a BlueSky subscriber)
        latest = {}
        d1 = drones[DRONE_NAMES[0]]
        client.subscribe(
            d1.robot_info["actual_pose"],
            lambda topic, msg: latest.update(msg),
        )
        await asyncio.sleep(2)
        print("latest actual_pose msg:", latest)
    finally:
        client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

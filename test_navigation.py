"""
test_navigation.py
Takeoff -> heading/altitude/speed changes -> waypoint -> land, with telemetry
printed between steps. The helpers mirror BlueSkyAdapter's HDG / ALT / SPD
commands so the agent layer can swap adapters later.

Coordinates are NED (north, east, down) in meters; up is negative z.
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
SCENE = "scene_basic_drone.jsonc"
NAME = "Drone1"


def pos(d: Drone):
    p = d.get_ground_truth_kinematics()["pose"]["position"]
    return p["x"], p["y"], p["z"]


def report(d: Drone, label: str):
    n, e, z = pos(d)
    print(f"[{label}] N={n:.1f} E={e:.1f} alt={-z:.1f} m")


async def change_heading(d: Drone, heading_deg: float, speed: float, secs: float):
    """HDG equivalent: fly at `speed` m/s along `heading_deg` for `secs`."""
    r = math.radians(heading_deg)
    task = await d.move_by_velocity_async(
        v_north=speed * math.cos(r), v_east=speed * math.sin(r),
        v_down=0.0, duration=secs)
    await task


async def change_altitude(d: Drone, alt_m: float, speed: float = 2.0):
    """ALT equivalent: climb/descend to alt_m holding N/E."""
    n, e, _ = pos(d)
    await (await d.move_to_position_async(north=n, east=e, down=-alt_m,
                                          velocity=speed))


async def goto(d: Drone, n: float, e: float, alt_m: float, speed: float = 4.0):
    await (await d.move_to_position_async(north=n, east=e, down=-alt_m,
                                          velocity=speed))


async def main():
    client = ProjectAirSimClient()
    try:
        client.connect()
        world = World(client, SCENE, delay_after_load_sec=2, sim_config_path=SIM_CONFIG)
        d = Drone(client, world, NAME)

        d.enable_api_control()
        d.arm()
        report(d, "start")

        await (await d.takeoff_async())
        report(d, "takeoff")

        await change_altitude(d, 10.0)
        report(d, "ALT 10m")

        await change_heading(d, heading_deg=90, speed=3.0, secs=5)   # east
        report(d, "HDG 090")

        await change_heading(d, heading_deg=0, speed=5.0, secs=4)    # north, faster
        report(d, "HDG 000 / SPD 5")

        await goto(d, 0.0, 0.0, 10.0)
        report(d, "back home")

        await (await d.land_async())
        report(d, "landed")

        d.disarm()
        d.disable_api_control()
    finally:
        client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

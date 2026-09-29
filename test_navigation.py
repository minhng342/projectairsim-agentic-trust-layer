"""
test_navigation.py
Takeoff -> climb -> rotate nose -> move along tracks -> return to launch -> land,
printing heading AND track after each step so the two are never confused.

Command semantics (see models/telemetry.py):
- rotate_to_heading: turns the nose (yaw) in place; the flight path doesn't change.
- move_along_track:  moves in a direction at a speed for a duration; the nose
                     is left where it is.
- change_altitude:   climbs/descends to a local altitude, holding N/E.

Coordinates are NED (north, east, down) in meters; up is negative z.
Run from the repo root with Blocks.exe running:  python .\\test_navigation.py
"""
import asyncio
import math
import os

from projectairsim import Drone, ProjectAirSimClient, World

from adapters.projectairsim_adapter import track_deg_from_velocity, yaw_deg_from_quaternion
from models.telemetry import Quaternion, Vector3
from utils.flight_safety import safe_shutdown

SIM_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_config") + os.sep
SCENE = "scene_basic_drone.jsonc"
NAME = "Drone1"


def state(d: Drone):
    k = d.get_ground_truth_kinematics()
    p, q, v = k["pose"]["position"], k["pose"]["orientation"], k["twist"]["linear"]
    return p, Quaternion(**q), Vector3(**v)


def report(d: Drone, label: str):
    p, q, v = state(d)
    trk = track_deg_from_velocity(v, 0.5)
    print(f"[{label:<22}] N={p['x']:6.1f} E={p['y']:6.1f} alt={-p['z']:5.1f} m  "
          f"hdg={yaw_deg_from_quaternion(q):5.1f}  trk={'  -  ' if trk is None else f'{trk:5.1f}'}  "
          f"gs={math.hypot(v.x, v.y):4.1f} m/s")


async def rotate_to_heading(d: Drone, heading_deg: float):
    """Point the nose at heading_deg (0 = north, clockwise). Position is held."""
    await (await d.rotate_to_yaw_async(yaw=math.radians(heading_deg)))


async def move_along_track(d: Drone, track_deg: float, speed_mps: float, duration_s: float):
    """Move toward track_deg at speed_mps for duration_s. Heading is not changed."""
    r = math.radians(track_deg)
    await (await d.move_by_velocity_async(v_north=speed_mps * math.cos(r),
                                          v_east=speed_mps * math.sin(r),
                                          v_down=0.0, duration=duration_s))


async def change_altitude(d: Drone, alt_m: float, speed_mps: float = 2.0):
    """Climb/descend to local altitude alt_m, holding the current N/E position."""
    p, _, _ = state(d)
    await (await d.move_to_position_async(north=p["x"], east=p["y"], down=-alt_m,
                                          velocity=speed_mps))


async def goto(d: Drone, north: float, east: float, alt_m: float, speed_mps: float = 4.0):
    await (await d.move_to_position_async(north=north, east=east, down=-alt_m,
                                          velocity=speed_mps))


async def main():
    client = ProjectAirSimClient()
    drone = None
    try:
        client.connect()
        world = World(client, SCENE, delay_after_load_sec=2, sim_config_path=SIM_CONFIG)
        drone = Drone(client, world, NAME)
        launch, _, _ = state(drone)
        home_n, home_e = launch["x"], launch["y"]

        drone.enable_api_control()
        drone.arm()
        report(drone, "start")

        await (await drone.takeoff_async())
        report(drone, "takeoff")

        await change_altitude(drone, 10.0)
        report(drone, "altitude 10 m")

        await rotate_to_heading(drone, 90)
        report(drone, "nose -> 090")              # heading changes, position doesn't

        await move_along_track(drone, track_deg=0, speed_mps=3.0, duration_s=5)
        report(drone, "track 000 @ 3 m/s")        # moves north with nose still east

        await move_along_track(drone, track_deg=90, speed_mps=5.0, duration_s=3)
        report(drone, "track 090 @ 5 m/s")

        await goto(drone, home_n, home_e, 10.0)
        report(drone, "over launch point")

        await (await drone.land_async())
        report(drone, "landed")
    finally:
        await safe_shutdown(drone)
        client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

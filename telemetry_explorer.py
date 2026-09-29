"""
telemetry_explorer.py
Shows every non-camera telemetry stream Project AirSim exposes for Drone1,
so we know what the adapter can map into models/telemetry.py.

1. Subscribes to robot_info topics (actual_pose, collision_info, rotor_info)
   and sensor topics (IMU, GPS, barometer, magnetometer).
2. Flies a short pattern so the values actually change.
3. Prints, for each topic: message rate, every field path + type + sample value.
4. Calls the request/response getters (kinematics, geo location, sensor data).
5. Saves everything to telemetry_dump.json for reference.

Run from this folder with Blocks.exe running:
    python .\telemetry_explorer.py
"""
import asyncio
import json
import os
import time

from projectairsim import ProjectAirSimClient, Drone, World

from utils.flight_safety import safe_shutdown

SIM_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_config") + os.sep
SCENE = "scene_basic_drone.jsonc"
NAME = "Drone1"
OUT_FILE = os.path.join(os.path.dirname(__file__), "telemetry_dump.json")


class TopicRecorder:
    """Keeps the latest message, first message and count per topic (no printing)."""

    def __init__(self):
        self.latest, self.first, self.count, self.t0 = {}, {}, {}, {}

    def make_callback(self, label):
        def cb(_topic, msg):
            if label not in self.first:
                self.first[label] = msg
                self.t0[label] = time.time()
            self.latest[label] = msg
            self.count[label] = self.count.get(label, 0) + 1
        return cb

    def rate_hz(self, label):
        n = self.count.get(label, 0)
        dt = time.time() - self.t0.get(label, time.time())
        return n / dt if dt > 0 else 0.0


def flatten(obj, prefix=""):
    """Yield (path, type, sample) for every leaf field in a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], (dict, list)):
        yield from flatten(obj[0], f"{prefix}[0]")
        if len(obj) > 1:
            yield (f"{prefix}[...]", f"list[{len(obj)}]", "")
    else:
        sample = obj
        if isinstance(obj, (list, tuple)) and len(obj) > 6:
            sample = f"{list(obj[:6])} ... ({len(obj)} items)"
        yield (prefix, type(obj).__name__, sample)


def print_schema(title, msg, extra=""):
    print(f"\n=== {title} {extra}")
    if msg is None:
        print("    (no message received)")
        return
    for path, typ, sample in flatten(msg):
        print(f"    {path:<45} {typ:<8} {sample}")


def safe_call(fn, *args):
    try:
        return fn(*args)
    except Exception as err:  # some getters depend on the robot config
        return {"error": str(err)}


async def main():
    client = ProjectAirSimClient()
    rec = TopicRecorder()
    drone = None
    try:
        client.connect()
        world = World(client, SCENE, delay_after_load_sec=2, sim_config_path=SIM_CONFIG)
        drone = Drone(client, world, NAME)

        # ---- subscribe to every non-image topic ----
        topics = {f'robot_info["{k}"]': v for k, v in drone.robot_info.items()}
        for sensor, streams in drone.sensors.items():
            for stream, topic in streams.items():
                if "camera" in stream:
                    continue  # images are large; not needed for the adapter
                topics[f'sensors["{sensor}"]["{stream}"]'] = topic
        for label, topic in topics.items():
            client.subscribe(topic, rec.make_callback(label))

        # ---- fly a little so velocities/attitude are non-zero ----
        drone.enable_api_control()
        drone.arm()
        await (await drone.takeoff_async())
        await (await drone.move_by_velocity_async(v_north=3.0, v_east=2.0,
                                                  v_down=-1.0, duration=4.0))
        # snapshot the pull-based getters while still moving
        moving = drone.move_by_velocity_async(v_north=0.0, v_east=3.0,
                                              v_down=0.0, duration=3.0)
        task = await moving
        await asyncio.sleep(1.0)
        getters = {
            "get_ground_truth_kinematics()": safe_call(drone.get_ground_truth_kinematics),
            "get_estimated_kinematics()": safe_call(drone.get_estimated_kinematics),
            "get_ground_truth_geo_location()": safe_call(drone.get_ground_truth_geo_location),
            "get_estimated_geo_location()": safe_call(drone.get_estimated_geo_location),
            'get_gps_data("GPS")': safe_call(drone.get_gps_data, "GPS"),
            'get_imu_data("IMU1")': safe_call(drone.get_imu_data, "IMU1"),
            'get_barometer_data("Barometer")': safe_call(drone.get_barometer_data, "Barometer"),
            'get_magnetometer_data("Magnetometer")': safe_call(drone.get_magnetometer_data, "Magnetometer"),
        }
        await task
        await (await drone.land_async())
        await safe_shutdown(drone)
        drone = None  # already shut down cleanly

        # ---- report ----
        print("\n################ SUBSCRIBED TOPICS (push) ################")
        for label in topics:
            print_schema(label, rec.latest.get(label),
                         f"| {rec.count.get(label, 0)} msgs, ~{rec.rate_hz(label):.1f} Hz")
        print("\n################ REQUEST GETTERS (pull) ################")
        for name, val in getters.items():
            print_schema(name, val)

        with open(OUT_FILE, "w") as f:
            json.dump({
                "topics": {l: {"topic": topics[l], "count": rec.count.get(l, 0),
                               "rate_hz": round(rec.rate_hz(l), 2),
                               "first": rec.first.get(l), "latest": rec.latest.get(l)}
                           for l in topics},
                "getters": getters,
            }, f, indent=2, default=str)
        print(f"\nSaved full samples to {OUT_FILE}")
    finally:
        await safe_shutdown(drone)  # no-op if the flight already shut down cleanly
        client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

"""
Live check of ProjectAirSimAdapter against the running simulator.

    python .\\test_adapter_live.py          # stationary: snapshots at 2 Hz for 5 s
    python .\\test_adapter_live.py --fly    # also takeoff, move, land while sampling

The --fly option uses the raw drone handle only to make values change; the
adapter itself has no command methods yet.
"""
import argparse
import asyncio
import json

from adapters.projectairsim_adapter import ProjectAirSimAdapter

VEHICLE = "Drone1"


def line(s) -> str:
    trk = f"{s.track_deg:5.1f}" if s.track_deg is not None else "  -  "
    c = s.collision
    coll = (f"{c.object_name}({'rest' if c.is_resting_contact else 'IMPACT' if c.is_resting_contact is False else '?'})"
            if c.in_contact else "-")
    return (f"t={s.sim_time_s:7.2f}s  {s.validation_status.value:<7} "
            f"N={s.position_ned_m.x:6.1f} E={s.position_ned_m.y:6.1f} alt={s.altitude_local_m:5.1f}m "
            f"hdg={s.heading_deg:5.1f} trk={trk} gs={s.ground_speed_mps:4.1f} vs={s.vertical_speed_mps:+4.1f} "
            f"age={s.telemetry_age_ms or 0:5.1f}ms coll={coll}")


async def sample(adapter, seconds: float, hz: float = 2.0):
    for _ in range(int(seconds * hz)):
        s = adapter.get_snapshot(VEHICLE)
        print(line(s))
        for e in s.validation_errors:
            print("    ERROR:", e)
        for w in s.validation_warnings:
            print("    WARN: ", w)
        await asyncio.sleep(1.0 / hz)


async def fly(adapter):
    d = adapter.drone(VEHICLE)
    d.enable_api_control()
    d.arm()
    await (await d.takeoff_async())
    await (await d.move_by_velocity_async(v_north=3.0, v_east=2.0, v_down=-1.0, duration=4.0))
    await (await d.land_async())
    d.disarm()
    d.disable_api_control()


async def main(do_fly: bool):
    with ProjectAirSimAdapter(vehicle_ids=[VEHICLE]) as adapter:
        await asyncio.sleep(0.5)  # let the pose cache fill
        print("--- stationary ---")
        await sample(adapter, 5)
        if do_fly:
            print("--- flying ---")
            flight = asyncio.create_task(fly(adapter))
            while not flight.done():
                await sample(adapter, 0.5)
            await flight
            print("--- after landing ---")
            await sample(adapter, 2)
        print("\n--- full snapshot (JSON) ---")
        print(adapter.get_snapshot(VEHICLE).model_dump_json(indent=2))
        print("\ntopic stats:", json.dumps(adapter.topic_stats()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", action="store_true")
    asyncio.run(main(ap.parse_args().fly))

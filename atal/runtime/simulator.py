"""Choosing the simulator for a run: the offline kinematic simulator or Project AirSim."""
from atal.core.paths import ConfigNotFound, check_scene, scene_location


def find_scene(spec):
    """(folder, file name) of a live scene, after checking that it exists and that every
    robot file it refers to sits next to it. Raises ConfigNotFound saying what is missing."""
    folder, name = scene_location(spec)
    missing = check_scene(folder, name)
    if missing:
        raise ConfigNotFound(f"{spec}: refers to {', '.join(missing)}, not found in "
                             f"{folder} (robot files must sit next to the scene)")
    return folder, name


def open_simulator(args, starts: dict, timing):
    """A context manager that yields the adapter. `starts` maps each vehicle to its (x, y)
    start; only the offline simulator needs it (live, the scene file places the drones)."""
    if args.offline:
        from atal.simulation.kinematic_sim import KinematicSim
        return KinematicSim(dict(starts), speedup=timing.k,
                            obstacles=getattr(args, "obstacle", None) or ())
    from atal.simulation.airsim import ProjectAirSimAdapter
    folder, name = find_scene(args.scene)
    return ProjectAirSimAdapter(vehicle_ids=list(starts), scene=name,
                                sim_config_path=str(folder))

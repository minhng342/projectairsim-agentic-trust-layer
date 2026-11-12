"""Choosing the simulator for a run: the offline kinematic simulator or Project AirSim."""


def open_simulator(args, starts: dict, timing):
    """A context manager that yields the adapter. `starts` maps each vehicle to its (x, y)
    start; only the offline simulator needs it (live, the scene file places the drones)."""
    if args.offline:
        from atal.simulation.kinematic_sim import KinematicSim
        return KinematicSim(dict(starts), speedup=timing.k,
                            obstacles=getattr(args, "obstacle", None) or ())
    from atal.simulation.airsim import ProjectAirSimAdapter
    return ProjectAirSimAdapter(vehicle_ids=list(starts), scene=args.scene)

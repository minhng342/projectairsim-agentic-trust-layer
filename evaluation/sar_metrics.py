"""
Search and rescue metrics for one run, from its run log (evaluation side, ground truth).

    m = sar_metrics(read_run("runs/sar/<batch>/run_00.jsonl"))

    outcome            the planner's outcome (victim_found, all_swept, time_limit, failed, ...)
    victim_found       the drone's own sensor detected the victim
    time_to_find_s     SIM seconds from the start of the search (the planner's plan) to the
                       detection; None if not found. The research question's metric.
    search_time_s      sim seconds from the plan to the end of the planner's last sweep
    sectors_swept      sectors the drone's memory marks swept (out of sector_count)
    area_coverage      share of the mission area that passed under the sensor footprint at
                       search altitude, from the ground-truth track (cells of
                       footprint / 10); a sector-level claim is checked against this
    distance_flown_m   horizontal track length at search altitude
    geofence_breaches  ground-truth samples outside area + limits.geofence_margin_m
    collisions         collision reports from the simulator; impacts = fast hits plus any
                       touch on a non-supporting surface (a wall); resting on the ground or
                       a roof, as at spawn and landing, is not one. An impact fails the run
"""
import math

from evaluation.recorder import RunLog
from models.sar import Mission


def _tracks(log: RunLog) -> dict[str, list[tuple[float, float, float, float]]]:
    """vehicle -> [(sim_s, x, y, alt)] from truth events."""
    out: dict[str, list] = {}
    for e in log.of_kind("truth"):
        for vid, p in (e.get("poses") or {}).items():
            if p.get("sim_time_ns") is None:
                continue
            out.setdefault(vid, []).append((p["sim_time_ns"] / 1e9, p["north_m"], p["east_m"],
                                            p["altitude_m"]))
    for v in out.values():
        v.sort()
    return out


def coverage_raster(mission: Mission, track, altitude_tolerance_m: float = 5.0,
                    cells_per_footprint: int = 10):
    """(covered cells, total cells, cell size): cells whose center fell inside the square
    footprint around the track while at search altitude; segments between samples are
    interpolated at half-cell steps."""
    a = mission.area
    f = mission.sensor.footprint_m
    cell = f / cells_per_footprint
    nx = max(1, round(a.size_m.width / cell))
    ny = max(1, round(a.size_m.height / cell))
    cx, cy = a.size_m.width / nx, a.size_m.height / ny
    x0, y0, half = a.origin.x_m, a.origin.y_m, f / 2
    alt = mission.flight.altitude_m
    covered: set[tuple[int, int]] = set()

    def mark(x, y):
        i0 = max(0, math.ceil((x - half - x0) / cx - 0.5))
        i1 = min(nx - 1, math.floor((x + half - x0) / cx - 0.5))
        j0 = max(0, math.ceil((y - half - y0) / cy - 0.5))
        j1 = min(ny - 1, math.floor((y + half - y0) / cy - 0.5))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                covered.add((i, j))

    prev = None
    for _, x, y, z in track:
        if abs(z - alt) > altitude_tolerance_m:
            prev = None
            continue
        if prev is None:
            mark(x, y)
        else:
            d = math.hypot(x - prev[0], y - prev[1])
            steps = max(1, math.ceil(d / (min(cx, cy) / 2)))
            for k in range(1, steps + 1):
                t = k / steps
                mark(prev[0] + t * (x - prev[0]), prev[1] + t * (y - prev[1]))
        prev = (x, y)
    return covered, nx * ny, (cx, cy)


def _is_impact(c: dict) -> bool:
    """A fast hit on anything, or any touch on a surface that can't hold the drone up (a
    wall). Resting on the ground or a roof (upward normal) is not an impact, including the
    spawn contact, whose speed is unknown because the pose window is still empty."""
    if c.get("is_resting_contact") is False:
        return True
    return c.get("is_supporting_surface") is not True


def sar_metrics(log: RunLog, mission: Mission | None = None) -> dict:
    cfg = (log.events[0].get("config") or {}) if log.events else {}
    if mission is None:
        mission = Mission.model_validate(cfg["mission"])
    vid = cfg.get("vehicle")
    plan = next(iter(log.of_kind("sar_plan")), None)
    end = next(iter(log.of_kind("planner_end")), None)
    det = next(iter(log.of_kind("victim_detected")), None)
    t_plan = plan["sim_time_ns"] / 1e9 if plan and plan.get("sim_time_ns") else None
    t_det = det["detection"]["sim_time_s"] if det else None
    t_end = end["sim_time_ns"] / 1e9 if end and end.get("sim_time_ns") else None
    track = _tracks(log).get(vid, [])
    covered, total, _ = coverage_raster(mission, track)
    alt, tol = mission.flight.altitude_m, 5.0
    dist = 0.0
    prev = None
    breaches = 0
    for _, x, y, z in track:
        if mission.geofence_excursion_m(x, y) > mission.limits.geofence_margin_m:
            breaches += 1
        if abs(z - alt) <= tol:
            if prev is not None:
                dist += math.hypot(x - prev[0], y - prev[1])
            prev = (x, y)
        else:
            prev = None
    swept = set()
    for e in log.of_kind("belief_update"):
        s = e.get("sector")
        if s:
            if s.get("status") == "swept":
                swept.add(s["sector_id"])
            else:
                swept.discard(s["sector_id"])
    cols = [e.get("collision") or {} for e in log.of_kind("collision")]
    impacts = [c for c in cols if _is_impact(c)]
    detection = det["detection"] if det else None
    truth = det.get("victim_truth") if det else cfg.get("victim")
    err = (math.hypot(detection["x_m"] - truth["x_m"], detection["y_m"] - truth["y_m"])
           if detection and truth else None)
    return {
        "run_index": cfg.get("run_index"), "seed": cfg.get("seed"),
        "outcome": end.get("outcome") if end else "incomplete",
        "victim": cfg.get("victim"), "victim_sector": cfg.get("victim_sector"),
        "victim_found": det is not None,
        "detected_sector": detection.get("sector_id") if detection else None,
        "detection_error_m": round(err, 2) if err is not None else None,
        "time_to_find_s": round(t_det - t_plan, 2) if (t_det is not None and t_plan is not None) else None,
        "search_time_s": round(t_end - t_plan, 2) if (t_end is not None and t_plan is not None) else None,
        "sectors_swept": len(swept), "sector_count": len(mission.grid.ids()),
        "area_coverage": round(len(covered) / total, 4) if total else 0.0,
        "distance_flown_m": round(dist, 1),
        "geofence_breaches": breaches,
        "collisions": len(cols), "impacts": len(impacts),
        "impact_objects": sorted({str(c.get("object_name")) for c in impacts})[:10],
        "run_complete": log.complete,
        "run_log_outcome": log.outcome,
    }

"""
Swarm metrics (Phase 3) for one run, from its run log (evaluation side, ground truth),
and the baseline over a batch.

    m = swarm_metrics(read_run("runs/sar_swarm/<batch>/run_00.jsonl"))
    b = batch_baseline([m0, m1, ...])

Team (the research question's metrics):
    victim_found / finder   some drone's own sensor detected the victim; which one first
    time_to_find_s          SIM seconds from the start of the search (the earliest drone's
                            plan) to the first detection; None if not found
    mission_time_s          from the start of the search to the last drone's planner end
    sectors_swept           sectors some drone swept ITSELF (union); peer clears don't count
    duplicate_sweeps        own sweeps beyond the first of each sector (wasted work)
    area_coverage           share of the area that passed under any drone's footprint at
                            that drone's own layer, from the ground-truth tracks
    time_to_hear_s          per other drone: sim seconds from the find until it accepted the
                            finder's VictimDetected (None = never heard it)
    min_separation_m        smallest 3D distance between two airborne drones (truth samples)
    separation              the acceptance check (Swarm.separation_*): smallest distance in
                            the pad zone and in cruise, violations of each floor, samples,
                            largest gap between samples, measured (False = a gap too long)
    messages                peer messages sent by type, valid deliveries, invalid dropped,
                            deliveries lost to radio range, delivery ratio (valid deliveries /
                            (sent x other drones)), peer clears accepted
    agents_deceived         Phase 4 metric; 0 in an honest run by construction, kept so the
                            columns line up
Per drone (`drones`): outcome, ended_by, return_home, own sweeps, accepted peer clears,
distance flown at its layer, its own footprint coverage, LLM decisions / fallbacks /
guard refusals, and the obstacle report: impacts (and what was hit), obstacle incidents
(impacts or blocked paths the planner reacted to), recoveries, blocked sectors (its own) and
blocked_heard (a teammate's report kept it out), lost
(dropped out of the search).
"""
import itertools
import math
import statistics

from evaluation.recorder import RunLog
from evaluation.sar_metrics import _is_impact, _tracks, coverage_raster
from models.sar import Mission


def swarm_metrics(log: RunLog, mission: Mission | None = None) -> dict:
    cfg = (log.events[0].get("config") or {}) if log.events else {}
    if mission is None:
        mission = Mission.model_validate(cfg["mission"])
    drones = cfg.get("drones") or []
    alt = {d["vehicle"]: d["altitude_m"] for d in drones}
    vids = [d["vehicle"] for d in drones]
    tracks = _tracks(log)

    def by(kind, vid):
        return [e for e in log.of_kind(kind) if e.get("vehicle_id") == vid]

    plans = {v: next(iter(by("sar_plan", v)), None) for v in vids}
    ends = {v: next(iter(by("planner_end", v)), None) for v in vids}
    t_plans = [p["sim_time_ns"] / 1e9 for p in plans.values() if p and p.get("sim_time_ns")]
    t0 = min(t_plans) if t_plans else None
    dets = sorted(log.of_kind("victim_detected"), key=lambda e: e["detection"]["sim_time_s"])
    first = dets[0] if dets else None
    t_det = first["detection"]["sim_time_s"] if first else None
    finder = first["vehicle_id"] if first else None

    per, swept_by, union_cov = {}, {}, set()
    total_cells = 0
    for v in vids:
        own = set()
        for e in by("belief_update", v):
            s = e.get("sector")
            if s and s.get("verified_by_self"):
                (own.add if s.get("status") == "swept" else own.discard)(s["sector_id"])
        for s in own:
            swept_by.setdefault(s, []).append(v)
        tr = tracks.get(v, [])
        cov, total_cells, _ = coverage_raster(mission, tr, altitude_m=alt.get(v))
        union_cov |= cov
        dist, prev = 0.0, None
        for _, x, y, z in tr:
            if abs(z - alt.get(v, mission.flight.altitude_m)) <= 5.0:
                if prev is not None:
                    dist += math.hypot(x - prev[0], y - prev[1])
                prev = (x, y)
            else:
                prev = None
        rules = by("peer_rule", v)
        decisions = [e for e in by("llm_decision", v) if not e.get("skipped")]
        end = ends[v]
        per[v] = {
            "altitude_m": alt.get(v),
            "outcome": end.get("outcome") if end else "incomplete",
            "ended_by": end.get("ended_by", end.get("outcome")) if end else None,
            "return_home": end.get("return_home") if end else None,
            "sectors_swept": len(own),
            "peer_clears_accepted": sum(1 for e in rules if e.get("msg_type") == "sector_cleared"
                                        and e.get("accepted") and e.get("belief_changed")),
            "found_victim": any(d["vehicle_id"] == v for d in dets),
            "distance_flown_m": round(dist, 1),
            "own_coverage": round(len(cov) / total_cells, 4) if total_cells else 0.0,
            "llm_decisions": len(decisions),
            "fallbacks": sum(1 for e in decisions if e.get("fallback")),
            "guard_rejections": sum(len(e.get("guard_rejections") or []) for e in decisions),
            "allocations": len(by("sar_allocation", v)),
            "stolen": sum(1 for e in by("sar_allocation", v) if e.get("stolen")),
            # obstacle report: hits the simulator saw, recoveries, drop-out
            "impacts": sum(1 for e in by("collision", v) if _is_impact(e.get("collision") or {})),
            "impact_objects": sorted({str((e.get("collision") or {}).get("object_name"))
                                      for e in by("collision", v)
                                      if _is_impact(e.get("collision") or {})})[:10],
            "obstacle_incidents": len(by("obstacle", v)),
            "obstacles_recovered": len(by("obstacle_recovered", v)),
            "blocked_sectors": (end.get("blocked") or []) if end else [],
            "blocked_heard": sorted({e["sector_id"] for e in by("peer_blocked_heard", v)}),
            "lost": bool(by("drone_lost", v)),
        }

    hear = {}
    if first is not None:
        for v in vids:
            if v == finder:
                continue
            got = [e for e in by("peer_rule", v) if e.get("msg_type") == "victim_detected"
                   and e.get("accepted")]
            t = got[0].get("sim_time_ns") if got else None
            hear[v] = round(t / 1e9 - t_det, 2) if t is not None else None

    team_blocked = sorted({e["sector_id"] for e in log.of_kind("obstacle") if e.get("sector_id")})
    unreachable = sorted(s for s in team_blocked if s not in swept_by)
    sep = separation(log, mission, drones, speedup=float(cfg.get("speedup") or 1.0))
    min_sep = sep["min_m"]

    peer_tx = {e["tx_id"] for e in log.of_kind("rid_transmit")
               if isinstance(e.get("payload"), dict) and "msg_type" in e["payload"]}
    sent = log.of_kind("peer_message_sent")
    by_type = {}
    for e in sent:
        t = (e.get("msg") or {}).get("msg_type")
        by_type[t] = by_type.get(t, 0) + 1
    received = len(log.of_kind("peer_message"))
    oor = sum(1 for e in log.of_kind("rid_out_of_range") if e.get("tx_id") in peer_tx)
    rid_oor = sum(1 for e in log.of_kind("rid_out_of_range") if e.get("tx_id") not in peer_tx)
    expected = len(sent) * max(len(vids) - 1, 0)
    cols = [e.get("collision") or {} for e in log.of_kind("collision")]
    impacts = [c for c in cols if _is_impact(c)]
    t_ends = [e["sim_time_ns"] / 1e9 for e in ends.values() if e and e.get("sim_time_ns")]
    own_total = sum(len(v) for v in swept_by.values())
    return {
        "run_index": cfg.get("run_index"), "seed": cfg.get("seed"),
        "insider_enabled": bool(cfg.get("insider_enabled")),
        "drone_count": len(vids),
        "victim": cfg.get("victim"), "victim_sector": cfg.get("victim_sector"),
        "victim_found": first is not None, "finder": finder,
        "detected_sector": first["detection"].get("sector_id") if first else None,
        "time_to_find_s": round(t_det - t0, 2) if (t_det is not None and t0 is not None) else None,
        "mission_time_s": round(max(t_ends) - t0, 2) if (t_ends and t0 is not None) else None,
        "time_to_hear_s": hear,
        "sectors_swept": len(swept_by), "sector_count": len(mission.grid.ids()),
        "blocked_shared": len({(e.get("msg") or {}).get("msg_id") for e in sent
                               if (e.get("msg") or {}).get("msg_type") == "sector_blocked"}),
        "unreachable_sectors": unreachable,
        "victim_unreachable": first is None and cfg.get("victim_sector") in unreachable,
        "duplicate_sweeps": own_total - len(swept_by),
        "duplicated_sectors": sorted(s for s, v in swept_by.items() if len(v) > 1),
        "area_coverage": round(len(union_cov) / total_cells, 4) if total_cells else 0.0,
        "distance_flown_m": round(sum(d["distance_flown_m"] for d in per.values()), 1),
        "min_separation_m": round(min_sep, 2) if min_sep is not None else None,
        "separation": sep,
        "messages": {"peer_sent": len(sent), "peer_sent_by_type": by_type,
                     "peer_delivered": received,
                     "peer_dropped_invalid": len(log.of_kind("peer_message_dropped")),
                     "identity_mismatch": len(log.of_kind("identity_mismatch")),
                     "processing_errors": len(log.of_kind("peer_message_error")),
                     "out_of_range": oor, "remote_id_out_of_range": rid_oor,
                     "delivery_ratio": round(received / expected, 4) if expected else None},
        "agents_deceived": 0,
        "geofence_breaches": len(log.of_kind("geofence_breach")),
        "collisions": len(cols), "impacts": len(impacts),
        "impact_objects": sorted({str(c.get("object_name")) for c in impacts})[:10],
        "drones": per,
        "run_complete": log.complete, "run_log_outcome": log.outcome,
    }


def separation(log: RunLog, mission: Mission, drones: list[dict], speedup: float = 1.0) -> dict:
    """Closest approach between airborne drones, split into the pad zone and cruise, against
    the mission's floors (Swarm.separation_*), and whether every PAIR was actually observed.

    Truth samples are 2 Hz: a closer approach between two samples can be missed (a sampled
    approximation, not continuous-flight safety). Per pair of configured drones, a sample is
    usable when both poses are present with a sim timestamp, each newer than that drone's
    previous pose (not stale), and the two timestamps are within 0.5 s; it is then either a distance (both airborne) or a known "not jointly airborne"
    state (at least one at its launch altitude). Anything else (a drone missing, a missing
    timestamp, skewed times) is a gap FOR THAT PAIR; samples of other drones never fill it.
    measured requires every pair's longest gap (including from the first truth sample to its
    first usable one and from its last to the end) to be within max_truth_gap_s plus 0.25
    host seconds of jitter at the run's speedup (offline x100 turns a 20 ms hiccup into 2 sim
    seconds). Airborne = at least 1 m above that drone's first observed altitude."""
    sw = mission.swarm
    vids = [d["vehicle"] for d in drones]
    pads = [(d["start"]["x_m"], d["start"]["y_m"]) for d in drones if d.get("start")]
    allowed = round(sw.max_truth_gap_s + 0.25 * speedup, 3)
    launch: dict[str, float] = {}
    pairs = {f"{a}-{b}": {"usable": 0, "airborne": 0, "not_jointly_airborne": 0,
                          "unusable": 0, "skewed": 0, "stale": 0, "max_gap_s": None,
                          "_last": None}
             for a, b in itertools.combinations(sorted(vids), 2)}
    out = {"min_m": None, "min_pad_m": None, "min_cruise_m": None, "samples": 0,
           "samples_without_time": 0, "skewed_pairs": 0, "pad_violations": 0,
           "cruise_violations": 0, "allowed_gap_s": allowed, "sample_hz": 2.0,
           "floors": {"cruise_m": sw.separation_cruise_m, "pad_m": sw.separation_pad_m,
                      "pad_zone_m": sw.pad_zone_m}}
    worst: list = []
    t_first = t_end = None
    last_pose: dict[str, int] = {}

    def gap(pr, t):
        if pr["_last"] is not None or t_first is not None:
            start = pr["_last"] if pr["_last"] is not None else t_first
            pr["max_gap_s"] = round(max(pr["max_gap_s"] or 0.0, t - start), 3)

    for e in log.of_kind("truth"):
        poses = e.get("poses") or {}
        if e.get("sim_time_ns") is None:
            out["samples_without_time"] += 1
            continue
        t = e["sim_time_ns"] / 1e9
        if t_first is None:
            t_first = t
        t_end = t if t_end is None else max(t_end, t)
        out["samples"] += 1
        fresh = set()
        for v, p in poses.items():
            if p.get("sim_time_ns") is not None and p.get("altitude_m") is not None:
                launch.setdefault(v, p["altitude_m"])
                if last_pose.get(v) is None or p["sim_time_ns"] > last_pose[v]:
                    fresh.add(v)
                last_pose[v] = p["sim_time_ns"]
        for a, b in itertools.combinations(sorted(vids), 2):
            pr = pairs[f"{a}-{b}"]
            pa, pb = poses.get(a), poses.get(b)
            if (pa is None or pb is None or pa.get("sim_time_ns") is None
                    or pb.get("sim_time_ns") is None or a not in launch or b not in launch):
                pr["unusable"] += 1
                continue
            if a not in fresh or b not in fresh:            # a repeated (stale) pose
                pr["unusable"] += 1
                pr["stale"] += 1
                continue
            if abs(pa["sim_time_ns"] - pb["sim_time_ns"]) > 5e8:
                pr["unusable"] += 1
                pr["skewed"] += 1
                out["skewed_pairs"] += 1
                continue
            gap(pr, t)
            pr["_last"] = t
            pr["usable"] += 1
            if pa["altitude_m"] - launch[a] < 1.0 or pb["altitude_m"] - launch[b] < 1.0:
                pr["not_jointly_airborne"] += 1
                continue
            pr["airborne"] += 1
            d = math.dist((pa["north_m"], pa["east_m"], pa["altitude_m"]),
                          (pb["north_m"], pb["east_m"], pb["altitude_m"]))
            near = any(math.hypot(p["north_m"] - x, p["east_m"] - y) <= sw.pad_zone_m
                       for p in (pa, pb) for x, y in pads)
            key, floor = ("pad", sw.separation_pad_m) if near else ("cruise",
                                                                    sw.separation_cruise_m)
            k = f"min_{key}_m"
            out[k] = round(d, 2) if out[k] is None else min(out[k], round(d, 2))
            out["min_m"] = d if out["min_m"] is None else min(out["min_m"], d)
            if d < floor:
                out[f"{key}_violations"] += 1
                if len(worst) < 5:
                    worst.append({"t_s": round(t, 1), "pair": [a, b], "zone": key,
                                  "distance_m": round(d, 2)})
    unmeasured = []
    for name, pr in pairs.items():
        if t_end is not None:
            gap(pr, t_end)                       # from the pair's last usable sample to the end
        last = pr.pop("_last")
        if last is None or (pr["max_gap_s"] or 0.0) > allowed:
            unmeasured.append(name)
    out["violations"] = worst
    out["pairs"] = pairs
    out["unmeasured_pairs"] = unmeasured
    out["max_gap_s"] = max((pr["max_gap_s"] or 0.0 for pr in pairs.values()), default=None)
    out["measured"] = out["samples"] > 1 and not unmeasured and len(vids) >= 1
    return out


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "p90": None,
                "stdev": None}
    s = sorted(xs)
    p90 = s[min(len(s) - 1, math.ceil(0.9 * len(s)) - 1)]
    return {"n": len(s), "mean": round(statistics.fmean(s), 2),
            "median": round(statistics.median(s), 2), "min": s[0], "max": s[-1], "p90": p90,
            "stdev": round(statistics.stdev(s), 2) if len(s) > 1 else 0.0}


def batch_baseline(runs: list[dict]) -> dict:
    """The Phase 3 deliverable numbers over a batch of runs."""
    sent = sum(r["messages"]["peer_sent"] * max(r["drone_count"] - 1, 0) for r in runs)
    got = sum(r["messages"]["peer_delivered"] for r in runs)
    return {
        "runs": len(runs),
        "found": sum(1 for r in runs if r["victim_found"]),
        "time_to_find_s": _stats([r["time_to_find_s"] for r in runs]),
        "mission_time_s": _stats([r["mission_time_s"] for r in runs]),
        "area_coverage": _stats([r["area_coverage"] for r in runs]),
        "sectors_swept": _stats([r["sectors_swept"] for r in runs]),
        "duplicate_sweeps_total": sum(r["duplicate_sweeps"] for r in runs),
        "impacts_total": sum(r.get("impacts") or 0 for r in runs),
        "drones_lost_total": sum(1 for r in runs for d in r["drones"].values() if d.get("lost")),
        "delivery_ratio": round(got / sent, 4) if sent else 1.0,
        "min_separation_m": min((r["min_separation_m"] for r in runs
                                 if r["min_separation_m"] is not None), default=None),
        "finders": {v: sum(1 for r in runs if r["finder"] == v)
                    for v in sorted({r["finder"] for r in runs if r["finder"]})},
    }

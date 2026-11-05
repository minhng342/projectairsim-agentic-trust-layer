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
    messages                peer messages sent by type, valid deliveries, invalid dropped,
                            deliveries lost to radio range, delivery ratio (valid deliveries /
                            (sent x other drones)), peer clears accepted
    agents_deceived         Phase 4 metric; 0 in an honest run by construction, kept so the
                            columns line up
Per drone (`drones`): outcome, ended_by, return_home, own sweeps, accepted peer clears,
distance flown at its layer, its own footprint coverage, LLM decisions / fallbacks /
guard refusals.
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

    # separation between airborne drones, from truth samples that hold several drones
    min_sep = None
    for e in log.of_kind("truth"):
        poses = e.get("poses") or {}
        for a, b in itertools.combinations(sorted(poses), 2):
            pa, pb = poses[a], poses[b]
            if pa["altitude_m"] < 2.0 or pb["altitude_m"] < 2.0:
                continue
            d = math.dist((pa["north_m"], pa["east_m"], pa["altitude_m"]),
                          (pb["north_m"], pb["east_m"], pb["altitude_m"]))
            min_sep = d if min_sep is None else min(min_sep, d)

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
        "drone_count": len(vids),
        "victim": cfg.get("victim"), "victim_sector": cfg.get("victim_sector"),
        "victim_found": first is not None, "finder": finder,
        "detected_sector": first["detection"].get("sector_id") if first else None,
        "time_to_find_s": round(t_det - t0, 2) if (t_det is not None and t0 is not None) else None,
        "mission_time_s": round(max(t_ends) - t0, 2) if (t_ends and t0 is not None) else None,
        "time_to_hear_s": hear,
        "sectors_swept": len(swept_by), "sector_count": len(mission.grid.ids()),
        "duplicate_sweeps": own_total - len(swept_by),
        "duplicated_sectors": sorted(s for s, v in swept_by.items() if len(v) > 1),
        "area_coverage": round(len(union_cov) / total_cells, 4) if total_cells else 0.0,
        "distance_flown_m": round(sum(d["distance_flown_m"] for d in per.values()), 1),
        "min_separation_m": round(min_sep, 2) if min_sep is not None else None,
        "messages": {"peer_sent": len(sent), "peer_sent_by_type": by_type,
                     "peer_delivered": received,
                     "peer_dropped_invalid": len(log.of_kind("peer_message_dropped")),
                     "out_of_range": oor, "remote_id_out_of_range": rid_oor,
                     "delivery_ratio": round(received / expected, 4) if expected else None},
        "agents_deceived": 0,
        "geofence_breaches": len(log.of_kind("geofence_breach")),
        "collisions": len(cols), "impacts": len(impacts),
        "impact_objects": sorted({str(c.get("object_name")) for c in impacts})[:10],
        "drones": per,
        "run_complete": log.complete, "run_log_outcome": log.outcome,
    }


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
        "delivery_ratio": round(got / sent, 4) if sent else 1.0,
        "min_separation_m": min((r["min_separation_m"] for r in runs
                                 if r["min_separation_m"] is not None), default=None),
        "finders": {v: sum(1 for r in runs if r["finder"] == v)
                    for v in sorted({r["finder"] for r in runs if r["finder"]})},
    }

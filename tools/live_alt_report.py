"""Compare live runs on altitude hold: one line per drone and one per run.

    python tools/live_alt_report.py runs/sar_swarm/<batch A> runs/sar_swarm/<batch B> ...

Reads each batch's run_*.jsonl and batch_summary.json. Altitude error comes from the path
follower's own trace (follower_trace: every command it sends, written at 2 Hz sim with the
extremes in between), so an excursion between samples still counts; the 2 Hz truth track
is shown beside it. Error = altitude - follower target, > 0 above. Only CRUISE and HOLD
commands are counted (a vertical move has its own target and reservation).
"""
import glob
import json
import sys
from pathlib import Path

E_Z = 1.0     # atal/airspace/bounds.py: per-drone altitude bound


def drone_lines(log):
    per = {}
    for line in Path(log).read_text().splitlines():
        e = json.loads(line)
        if e["kind"] == "follower_trace" and e.get("mode") in ("cruise", "hold"):
            d = per.setdefault(e["vehicle_id"], {"hi": [], "lo": [], "gap": [], "age": []})
            d["hi"].append(e["alt_err_max_m"])
            d["lo"].append(e["alt_err_min_m"])
            if e.get("cmd_gap_max_s") is not None:
                d["gap"].append(e["cmd_gap_max_s"])
            if e.get("telemetry_age_ms") is not None:
                d["age"].append(e["telemetry_age_ms"])
        elif e["kind"] == "unauthorised_motion":
            per.setdefault(e["vehicle_id"], {"hi": [], "lo": [], "gap": [], "age": []})[
                "flag"] = f"{e.get('alt_m')} vs {e.get('cleared_m')}"
    return per


def main(batches):
    for b in batches:
        b = Path(b)
        s = json.loads((b / "batch_summary.json").read_text())
        mf = b / "manifest.json"
        man = json.loads(mf.read_text()) if mf.exists() else {}
        args = (man.get("command") or {}).get("args") or {}
        seeds = [x.get("seed") for x in man.get("seeds") or []]
        print(f"== {b.name}  commit {(man.get('git') or {}).get('commit_short', '?')}  "
              f"corner_slowdown={bool(args.get('corner_slowdown'))}  "
              f"planner={args.get('planner')}  model={args.get('model')}  seeds={seeds}")
        for i, log in enumerate(sorted(glob.glob(str(b / "run_*.jsonl")))):
            r = s["runs"][i]
            sep = r.get("separation") or {}
            print(f"  run {i}: found={r['victim_found']} t_find={r['time_to_find_s']} "
                  f"mission={r['mission_time_s']} s  min_sep={sep.get('min_cruise_m')} m "
                  f"(pad {sep.get('min_pad_m')})  all_landed={r.get('all_landed')}  "
                  f"pad_timeouts={r.get('pad_zone_timeouts')}  "
                  f"fail={'; '.join(r.get('fail_reasons') or []) or 'none'}")
            for vid, d in sorted(drone_lines(log).items()):
                hi = max(d["hi"], default=None)
                lo = min(d["lo"], default=None)
                over = sum(1 for x in d["hi"] if x > E_Z) + sum(1 for x in d["lo"] if x < -E_Z)
                gaps = sorted(d["gap"])
                p99 = gaps[int(0.99 * (len(gaps) - 1))] if gaps else None
                dr = r["drones"].get(vid, {})
                print(f"    {vid}: alt err max {hi:+.2f} min {lo:+.2f} m, trace windows over "
                      f"e_z {over}/{len(d['hi'])}; cmd gap max {max(gaps, default=0):.2f} "
                      f"p99 {p99} s; telemetry age max {max(d['age'], default=0):.0f} ms; "
                      f"flag {d.get('flag', 'none')}; landed {dr.get('landed')} "
                      f"recoveries {dr.get('recoveries')}")


if __name__ == "__main__":
    main(sys.argv[1:])

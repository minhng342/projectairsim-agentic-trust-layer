"""
Reference cases for the restructuring work: a small set of offline runs whose observable
behaviour must stay the same while code is moved and reorganised.

    python tools/reference_cases.py --record docs/refactor/reference_baseline.json
    python tools/reference_cases.py --check  docs/refactor/reference_baseline.json
    python tools/reference_cases.py --check  docs/refactor/reference_baseline.json --only team_fixed

Each case is one runner command on the offline kinematic simulator with three seeded runs.
The exit status is 0 only when nothing below is reported.

Checked in every run of every case, whatever the reference says (run health):
    every drone landed and disarmed, and its descent succeeded
    the run log is complete and did not end "aborted"
    no run error, no failed background task, no stalled simulator
    team runs: no peer-message processing error, no pad zone left held
    the runner's exit status agrees with its own verdicts (0 = all runs passed, 1 = some
    run failed acceptance); any other status, or no batch summary, is a crash

Compared with the reference, per run:
    run index, seed and victim sector    must be identical (the same experiment)
    victim_found                         must match
    passed                               must match. In the obstacle case, whether a drone
                                         touches the obstacle varies between executions, so
                                         there a run may fail, but only for the reason
                                         "N impacts"; any other failure reason is reported
    time_to_find_s                       within the tolerance (default: 15 % or 30 s,
                                         whichever is larger), only in the cases with a
                                         fixed sweep order: one drone, and the static split

Why not exact numbers: the offline simulator is paced by the host clock, so the same seed
gives slightly different flight times and task ordering from one execution to the next. One
drone repeats within about 2 %. A cooperating team does not: the drones re-split the open
sectors at every decision, so a small timing difference changes who sweeps what, and the
time to find moved by up to 150 % between two executions of the same commit (measured
2026-11-12). Team times, sector counts and finders are recorded for information only. The
same seed fixes the victim, not the team's execution.

No simulator, model server or network is needed. About 6 minutes for all eight cases.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ONE = "run_sar_mission.py"
TEAM = "run_sar_swarm.py"
DEV = ["--mission", "missions/sar/mission_dev.yaml", "--scenario", "missions/sar/scenario_dev.yaml"]

# Cases whose sweep order does not depend on timing, so the time to find is comparable.
TIME_COMPARED = {"one_fixed", "one_llm", "team_static"}
# Cases where a run may fail acceptance, and the only failure reasons allowed there.
EXPECTED_FAILURES = {"team_obstacle": re.compile(r"^\d+ impacts$")}

# name -> (script, extra arguments, what the case exercises)
CASES = {
    "one_fixed": (ONE, DEV, "one drone, fixed sector order"),
    "one_llm": (ONE, DEV + ["--planner", "llm", "--model", "scripted"],
                "one drone, scripted stand-in for the LLM, invariant guard on"),
    "team_fixed": (TEAM, [], "three drones cooperating over the bus"),
    "team_llm": (TEAM, ["--planner", "llm", "--model", "scripted"],
                 "three drones, scripted stand-in for the LLM"),
    "team_dropout": (TEAM, ["--dropout", "Drone2@200"], "one drone fails 200 s into the search"),
    "team_static": (TEAM, ["--allocation", "static"], "fixed pre-mission split, no claims"),
    "team_obstacle": (TEAM, ["--obstacle", "tower:60,80,60,80,165"],
                      "an obstacle in the grid: impacts, recovery, shared blocked sectors"),
    "team_self_only": (TEAM, ["--peer-clear-rule", "self_only"],
                       "peer clears are recorded but never accepted"),
}
RUNS = 3


def run_health(r: dict) -> dict:
    """The health facts of one run, from its entry in batch_summary.json. A one-drone run has
    them at the top level, a team run per drone."""
    drones = r.get("drones") or {"drone": r}
    msgs = r.get("messages") or {}
    return {
        "landed": {v: bool(d.get("landed")) for v, d in sorted(drones.items())},
        "descend": {v: d.get("descend") for v, d in sorted(drones.items())},
        "drone_errors": {v: d["error"] for v, d in sorted(drones.items()) if d.get("error")},
        "run_complete": bool(r.get("run_complete")),
        "run_log_outcome": r.get("run_log_outcome"),
        "run_error": r.get("run_error"),
        "background_errors": list(r.get("background_errors") or []),
        "sim_stalled": r.get("sim_stalled"),
        "processing_errors": msgs.get("processing_errors", 0),
        "pad_zone_held_unconfirmed": list(r.get("pad_zone_held_unconfirmed") or []),
    }


def health_problems(h: dict) -> list[str]:
    why = [f"{v} not landed" for v, ok in h["landed"].items() if not ok]
    why += [f"{v} descend {st}" for v, st in h["descend"].items() if st != "succeeded"]
    why += [f"{v} error: {e}" for v, e in h["drone_errors"].items()]
    if not h["run_complete"]:
        why.append("run log incomplete")
    if h["run_log_outcome"] in (None, "aborted"):
        why.append(f"run log ended {h['run_log_outcome']}")
    if h["run_error"]:
        why.append(f"run error: {h['run_error']}")
    if h["background_errors"]:
        why.append(f"background task failed: {'; '.join(h['background_errors'])}")
    if h["sim_stalled"]:
        why.append(f"simulator stalled: {h['sim_stalled']}")
    if h["processing_errors"]:
        why.append(f"{h['processing_errors']} peer message processing error(s)")
    if h["pad_zone_held_unconfirmed"]:
        why.append(f"pad zone left held by {', '.join(h['pad_zone_held_unconfirmed'])}")
    return why


def run_case(name: str) -> dict:
    script, extra, what = CASES[name]
    with tempfile.TemporaryDirectory(prefix=f"refcase-{name}-") as out:
        cmd = [sys.executable, script, *extra, "--offline", "--runs", str(RUNS), "--out", out]
        started = time.monotonic()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        summaries = list(Path(out).glob("*/batch_summary.json"))
        if len(summaries) != 1:
            raise RuntimeError(f"{name}: the runner crashed (exit status {proc.returncode}, "
                               f"{len(summaries)} batch summaries)\n{proc.stdout[-2000:]}\n"
                               f"{proc.stderr[-2000:]}")
        batch = json.loads(summaries[0].read_text())
    runs = []
    for r in batch["runs"]:
        runs.append({
            "run_index": r["run_index"], "seed": r["seed"],
            "victim_sector": r["victim_sector"], "victim_found": r["victim_found"],
            "passed": not r["fail_reasons"], "fail_reasons": r["fail_reasons"],
            "health": run_health(r),
            "time_to_find_s": r["time_to_find_s"],
            "sectors_swept": r["sectors_swept"],          # information only
            "finder": r.get("finder"),                     # information only
            "duplicate_sweeps": r.get("duplicate_sweeps"),  # information only
        })
    return {"what": what, "command": " ".join(["python", script, *extra, "--offline",
                                               "--runs", str(RUNS)]),
            "exit_code": proc.returncode, "host_seconds": round(time.monotonic() - started),
            "runs": runs}


def own_problems(name: str, result: dict) -> list[str]:
    """What is wrong with one execution of a case on its own, with no reference: run health,
    the exit status, and failure reasons the case does not allow."""
    problems = []
    passed = [r["passed"] for r in result["runs"]]
    expected_exit = 0 if all(passed) else 1
    if result["exit_code"] != expected_exit or len(passed) != RUNS:
        problems.append(f"{name}: exit status {result['exit_code']} with {sum(passed)} of "
                        f"{len(passed)} runs passed (expected status {expected_exit}, {RUNS} runs)")
    allowed = EXPECTED_FAILURES.get(name)
    for r in result["runs"]:
        tag = f"{name} run {r['run_index']}"
        problems += [f"{tag}: {why}" for why in health_problems(r["health"])]
        if allowed is not None:
            problems += [f"{tag}: failure not expected in this case: {why}"
                         for why in r["fail_reasons"] if not allowed.match(why)]
    return problems


def compare(name: str, ref: dict, new: dict, rel: float, abs_s: float) -> list[str]:
    """Differences between a reference execution and a new one, plus everything that is
    wrong with the new one on its own."""
    problems = own_problems(name, new)
    if len(ref["runs"]) != len(new["runs"]):
        return problems + [f"{name}: {len(new['runs'])} runs, reference has {len(ref['runs'])}"]
    for a, b in zip(ref["runs"], new["runs"]):
        tag = f"{name} run {a['run_index']}"
        for key in ("run_index", "seed", "victim_sector"):
            if a[key] != b[key]:
                problems.append(f"{tag}: {key} {b[key]}, reference {a[key]} (not the same "
                                f"experiment)")
        if a["victim_found"] != b["victim_found"]:
            problems.append(f"{tag}: victim_found {b['victim_found']}, reference "
                            f"{a['victim_found']}")
        if name not in EXPECTED_FAILURES and a["passed"] != b["passed"]:
            problems.append(f"{tag}: passed {b['passed']} ({b['fail_reasons']}), reference "
                            f"{a['passed']} ({a['fail_reasons']})")
        ta, tb = a["time_to_find_s"], b["time_to_find_s"]
        if name in TIME_COMPARED and ta is not None and tb is not None:
            allowed = max(rel * ta, abs_s)
            if abs(ta - tb) > allowed:
                problems.append(f"{tag}: time to find {tb:.1f} s, reference {ta:.1f} s "
                                f"(allowed difference {allowed:.1f} s)")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", metavar="FILE", help="run the cases and write their results")
    mode.add_argument("--check", metavar="FILE", help="run the cases and compare with FILE")
    ap.add_argument("--only", action="append", choices=sorted(CASES), default=None,
                    help="run only this case (repeatable)")
    ap.add_argument("--rel-tolerance", type=float, default=0.15)
    ap.add_argument("--abs-tolerance-s", type=float, default=30.0)
    args = ap.parse_args(argv)

    names = args.only or list(CASES)
    results = {}
    for name in names:
        print(f"{name}: {CASES[name][2]} ...", flush=True)
        results[name] = run_case(name)
        r = results[name]
        found = sum(x["victim_found"] for x in r["runs"])
        passed = sum(x["passed"] for x in r["runs"])
        print(f"    found {found}/{RUNS}, passed {passed}/{RUNS}, {r['host_seconds']} s")

    if args.record:
        # a reference must itself be sound: nothing is written if any case is not
        problems = [p for name in names for p in own_problems(name, results[name])]
        for p in problems:
            print(f"NOT RECORDED: {p}")
        if problems:
            return 1
        path = Path(args.record)
        existing = json.loads(path.read_text()) if path.exists() and args.only else {}
        existing.update(results)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, indent=2) + "\n")
        print(f"recorded {len(results)} case(s) in {path}")
        return 0

    reference = json.loads(Path(args.check).read_text())
    problems = []
    for name in names:
        if name not in reference:
            problems.append(f"{name}: not in the reference file")
            continue
        problems += compare(name, reference[name], results[name],
                            args.rel_tolerance, args.abs_tolerance_s)
    for p in problems:
        print(f"DIFFERENT: {p}")
    print("RESULT: " + ("SAME BEHAVIOUR" if not problems else f"{len(problems)} difference(s)"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

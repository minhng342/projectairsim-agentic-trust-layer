"""
Paired comparison of SAR batches on the same victims (Phase 3 evaluation).

Every batch of the dev scenarios draws run i's victim from the same seed, so run i of a
one-drone batch and run i of a three-drone batch search for the same victim: the fair
comparison is per victim (paired), not of two means.

    pc = paired_comparison(load_batch("runs/sar/<one drone>"), three_drone_runs)
    pc["speedup"]      {"mean", "median", "ci95": [lo, hi], "ci95_median": [lo, hi]}
                       speedup of a pair = reference time to find / batch time to find
    pc["time_saved_s"] {"mean", "ci95"}: reference - batch, seconds
    pc["faster"]       pairs where the batch found the victim sooner

Pairs are matched on the victim's position (any batch format with run["victim"]); a run
whose victim has no match in the reference is "unpaired". A pair where either side did
not find the victim has no speedup: it is counted under not_found and left out of the
interval (with every run found, as in the offline batches, nothing is left out).

The interval is a percentile bootstrap over pairs: resample the pairs with replacement
(n_boot times, fixed seed so the numbers are reproducible), take the mean (and median) of
each resample, report the 2.5th and 97.5th percentiles. With 20 pairs it is a rough
interval, which is the point: it shows how much the 20-victim mean can move.

Command line, the comparison table for several batches against one reference:

    python -m evaluation.compare REFERENCE BATCH [BATCH ...] [--labels "1 drone" ...]
        [--md table.md]
"""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path


def load_batch(path) -> dict:
    """A batch_summary.json (or the batch folder holding one)."""
    p = Path(path)
    if p.is_dir():
        p = p / "batch_summary.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
        raise ValueError(f"{p} is not a batch_summary.json (no runs list)")
    return data


def _runs(batch) -> list[dict]:
    runs = batch["runs"] if isinstance(batch, dict) else list(batch)
    return [r for r in runs if not r.get("sim_stalled")]


def _key(run: dict):
    v = run.get("victim") or {}
    if "x_m" in v and "y_m" in v:
        return round(float(v["x_m"]), 3), round(float(v["y_m"]), 3)
    return ("seed", run.get("seed"), run.get("run_index"))


def _pct(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def bootstrap_ci(values, stat=statistics.fmean, n_boot=10000, seed=0, level=0.95):
    """Percentile bootstrap interval of stat(values); None for fewer than 2 values."""
    vals = list(values)
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    n = len(vals)
    boots = sorted(stat([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    a = (1 - level) / 2
    return [_pct(boots, a), _pct(boots, 1 - a)]


def _r(x, nd=3):
    return None if x is None else round(x, nd)


def paired_comparison(reference, runs, reference_label=None, n_boot=10000, seed=0) -> dict:
    ref = {}
    for r in _runs(reference):
        ref.setdefault(_key(r), r)
    pairs, unpaired, nf_ref, nf_batch = [], 0, 0, 0
    for r in runs:
        if r.get("sim_stalled"):
            continue
        q = ref.get(_key(r))
        if q is None:
            unpaired += 1
            continue
        a, b = q.get("time_to_find_s"), r.get("time_to_find_s")
        entry = {"run_index": r.get("run_index"), "victim_sector": r.get("victim_sector"),
                 "reference_s": a, "batch_s": b,
                 "speedup": _r(a / b) if (a is not None and b) else None}
        if a is None:
            nf_ref += 1
        if b is None:
            nf_batch += 1
        pairs.append(entry)
    good = [p for p in pairs if p["speedup"] is not None]
    ratios = [p["reference_s"] / p["batch_s"] for p in good]
    saved = [p["reference_s"] - p["batch_s"] for p in good]
    ci = bootstrap_ci(ratios, n_boot=n_boot, seed=seed)
    ci_med = bootstrap_ci(ratios, stat=statistics.median, n_boot=n_boot, seed=seed)
    ci_saved = bootstrap_ci(saved, n_boot=n_boot, seed=seed)
    return {
        "reference": str(reference_label) if reference_label is not None else
        (reference.get("batch") if isinstance(reference, dict) else None),
        "pairs": len(good), "matched": len(pairs), "unpaired": unpaired,
        "not_found": {"reference": nf_ref, "batch": nf_batch},
        "faster": sum(1 for p in good if p["batch_s"] < p["reference_s"]),
        "speedup": {"mean": _r(statistics.fmean(ratios)) if ratios else None,
                    "median": _r(statistics.median(ratios)) if ratios else None,
                    "ci95": [_r(x) for x in ci] if ci else None,
                    "ci95_median": [_r(x) for x in ci_med] if ci_med else None},
        "time_saved_s": {"mean": _r(statistics.fmean(saved), 1) if saved else None,
                         "ci95": [_r(x, 1) for x in ci_saved] if ci_saved else None},
        "method": f"pairs matched on victim position; speedup = reference / batch time to "
                  f"find; percentile bootstrap over pairs, {n_boot} resamples, seed {seed}",
        "per_run": pairs,
    }


def batch_row(batch: dict) -> dict:
    """The comparison-table numbers of one batch (either runner's format)."""
    runs = _runs(batch)
    t = [r["time_to_find_s"] for r in runs if r.get("time_to_find_s") is not None]
    dup = [r["duplicate_sweeps"] for r in runs if "duplicate_sweeps" in r]
    msgs = [r.get("messages") or {} for r in runs]
    sent = sum((m.get("peer_sent") or 0) * max((r.get("drone_count") or 1) - 1, 0)
               for m, r in zip(msgs, runs))
    got = sum(m.get("peer_delivered") or 0 for m in msgs)
    passed = batch.get("passed") or []
    return {"runs": len(runs), "found": sum(1 for r in runs if r.get("victim_found")),
            "passed": sum(1 for p in passed if p),
            "t_mean": _r(statistics.fmean(t), 1) if t else None,
            "t_median": _r(statistics.median(t), 1) if t else None,
            "t_min": min(t, default=None), "t_max": max(t, default=None),
            "coverage_mean": _r(statistics.fmean(r["area_coverage"] for r in runs), 4)
            if runs else None,
            "duplicates": sum(dup) if dup else None,
            "delivery": _r(got / sent, 4) if sent else None,
            "impacts": sum(r.get("impacts") or 0 for r in runs)}


def table(reference: dict, batches: list[dict], labels: list[str]) -> str:
    head = ("| Condition | Found | Passed | Time to find, mean / median (s) | min / max (s) | "
            "Paired speedup, mean (95% CI) | Median speedup (95% CI) | Faster | Coverage | "
            "Duplicate sweeps | Delivery |")
    lines = [head, "|" + "---|" * 11]

    def fmt(x, pct=False):
        if x is None:
            return "n/a"
        return f"{x:.1%}" if pct else f"{x:g}"
    for i, (b, label) in enumerate(zip([reference, *batches], labels)):
        row = batch_row(b)
        if i == 0:
            sp = med = "1.0 (reference)"
            faster = "n/a"
        else:
            pc = paired_comparison(reference, _runs(b))
            s = pc["speedup"]
            sp = (f"{s['mean']}x ({s['ci95'][0]}-{s['ci95'][1]}x)" if s["ci95"]
                  else f"{s['mean']}x")
            med = (f"{s['median']}x ({s['ci95_median'][0]}-{s['ci95_median'][1]}x)"
                   if s["ci95_median"] else f"{s['median']}x")
            faster = f"{pc['faster']}/{pc['pairs']}"
        lines.append(f"| {label} | {row['found']}/{row['runs']} | {row['passed']}/{row['runs']} "
                     f"| {fmt(row['t_mean'])} / {fmt(row['t_median'])} | {fmt(row['t_min'])} / "
                     f"{fmt(row['t_max'])} | {sp} | {med} | {faster} | "
                     f"{fmt(row['coverage_mean'], True)} | {fmt(row['duplicates'])} | "
                     f"{fmt(row['delivery'], True)} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="paired comparison of SAR batches")
    ap.add_argument("reference", help="reference batch (folder or batch_summary.json)")
    ap.add_argument("batches", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="row labels: the reference first, then each batch")
    ap.add_argument("--md", default=None, help="also write the table to this file")
    args = ap.parse_args(argv)
    ref = load_batch(args.reference)
    batches = [load_batch(b) for b in args.batches]
    labels = args.labels or [args.reference, *args.batches]
    if len(labels) != len(batches) + 1:
        print("--labels needs one label for the reference and one per batch")
        return 2
    out = table(ref, batches, labels)
    print(out)
    if args.md:
        Path(args.md).write_text(out + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

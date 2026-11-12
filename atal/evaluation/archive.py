"""
Pack batches of runs into one evidence archive.

    python -m atal.evaluation.archive BATCH [BATCH ...] [--reference BATCH] [--labels ...]
                                 [--out runs/evidence]

For each batch folder: check it against its manifest.json (written by the runner at the
end of the batch: commit, config hashes, model and inference settings, seeds, attempted /
valid / stalled / passed counts, a sha256 per file), refuse to pack one that is missing a
manifest or whose files changed since (--force packs it anyway and says so). With
--reference, also write the paired comparison table (atal/evaluation/compare.py) for the
batches against it. The result is runs/evidence/<UTC time>.tar.gz with every batch folder,
an EVIDENCE.md index (one line per batch: condition, commit, counts, config hash) and the
comparison table, plus the same EVIDENCE.md next to it for a quick read.
"""
import argparse
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from atal.evaluation.compare import load_batch, table
from atal.evaluation.evidence import MANIFEST, verify
from atal.evaluation.recorder import REPO_ROOT


def _describe(summary: dict) -> str:
    sw = summary.get("swarm") or {}
    drones = summary.get("drones")
    n = len(drones) if isinstance(drones, list) else 1
    bits = [f"{n} drone{'s' if n != 1 else ''}", summary.get("planner") or "?"]
    if summary.get("model"):
        bits.append(summary["model"])
    if n > 1:
        if sw.get("allocation") == "static":
            bits.append("static split")
        if sw.get("radio_range_m") is not None and sw["radio_range_m"] < 5:
            bits.append(f"radio {sw['radio_range_m']:g} m (no messages)")
        if sw.get("peer_clear_rule") not in (None, "single"):
            bits.append(f"clears {sw['peer_clear_rule']}")
    bits.append(summary.get("mode") or "?")
    return ", ".join(bits)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="pack SAR batches into an evidence archive")
    ap.add_argument("batches", nargs="+", help="batch folders")
    ap.add_argument("--reference", default=None,
                    help="reference batch folder for the paired comparison (also packed)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="comparison row labels: the reference first, then each batch")
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "evidence"))
    ap.add_argument("--force", action="store_true",
                    help="pack batches whose manifest is missing or does not match")
    args = ap.parse_args(argv)

    folders = ([Path(args.reference)] if args.reference else []) + [Path(b) for b in args.batches]
    lines, problems_any = [], False
    for f in folders:
        if not f.is_dir():
            print(f"{f}: not a folder")
            return 2
        probs = verify(f)
        summary = load_batch(f)
        mf = json.loads((f / MANIFEST).read_text()) if (f / MANIFEST).exists() else {}
        g, c = mf.get("git") or {}, mf.get("counts") or {}
        status = "intact" if not probs else f"{len(probs)} PROBLEM(S): {'; '.join(probs[:5])}"
        problems_any |= bool(probs)
        lines.append(
            f"- `{f.name}`: {_describe(summary)}; commit {g.get('commit_short')}"
            f"{' (DIRTY)' if g.get('dirty') else ''}; config "
            f"{((mf.get('config') or {}).get('config_sha256') or '?')[:12]}; "
            f"{c.get('attempted', len(summary['runs']))} attempted, {c.get('valid', '?')} valid, "
            f"{c.get('stalled', '?')} stalled, {c.get('passed', '?')} passed; files {status}")
        print(lines[-1])
    if problems_any and not args.force:
        print("not packed: a batch has no manifest or changed since its run (--force to pack "
              "anyway, marked in EVIDENCE.md)")
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md = [f"# SAR evidence {stamp}", "", "Batches (manifest.json in each folder):", "", *lines]
    if args.reference:
        ref = load_batch(args.reference)
        batches = [load_batch(b) for b in args.batches]
        labels = args.labels or [_describe(ref), *(_describe(b) for b in batches)]
        if len(labels) != len(batches) + 1:
            print("--labels needs one label for the reference and one per batch")
            return 2
        md += ["", f"Paired comparison against `{Path(args.reference).name}` (pairs matched on "
                   f"victim position; 95% percentile bootstrap over pairs):", "",
               table(ref, batches, labels)]
    text = "\n".join(md) + "\n"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = out / f"{stamp}.EVIDENCE.md"
    index.write_text(text, encoding="utf-8")
    tar_path = out / f"{stamp}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(index, arcname=f"{stamp}/EVIDENCE.md")
        for f in folders:
            tar.add(f, arcname=f"{stamp}/{f.name}")
    print(f"\n{text}\narchive: {tar_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Evidence manifest for a batch of runs: what exactly produced these numbers.

    write_manifest(out_dir, argv=sys.argv, args=vars(args), files={...}, model=..., results=...,
                   passed=..., runs_planned=N)  ->  out_dir/manifest.json (also returned)

Written by the runner at the end of every batch (so the commit is the one that flew, not
the one checked out later), and checked / packed by scripts/archive_batch.py.

    git          commit, branch, dirty (uncommitted changes to tracked files at run time:
                 such a batch is not reproducible from the commit alone)
    command      the command line and the parsed arguments
    config       sha256 of each input file (mission, scenario, scene) and one config_sha256
                 over those hashes plus the arguments that change the run
    model        the LLM and its inference settings (thinking, extra body), None for the
                 fixed planner
    seeds        run index -> seed and victim sector
    counts       planned, attempted, valid (attempted minus stalled), stalled, passed
    environment  python, platform, host, package versions
    files        sha256 and size of every file in the batch folder (the run logs, memories,
                 summaries, plots), so an archived copy can be checked
"""
import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from atal.evaluation.recorder import REPO_ROOT

MANIFEST = "manifest.json"
# arguments that do not change what is flown (output location only)
_NOT_CONFIG = {"out", "reference"}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*cmd) -> str | None:
    try:
        r = subprocess.run(["git", *cmd], cwd=REPO_ROOT, capture_output=True, text=True,
                           timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def git_state() -> dict:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "commit_short": commit[:7] if commit else None,
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None,
            "changed_files": status.splitlines()[:50] if status else []}


def _versions() -> dict:
    out = {}
    for pkg in ("pydantic", "pydantic-ai", "pydantic-ai-slim", "projectairsim", "logfire"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def batch_files(out_dir: Path) -> dict:
    return {p.relative_to(out_dir).as_posix(): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in sorted(out_dir.rglob("*")) if p.is_file() and p.name != MANIFEST}


def write_manifest(out_dir, argv, args: dict, files: dict, model, results: list[dict],
                   passed: list[bool], runs_planned: int) -> dict:
    out_dir = Path(out_dir)
    inputs = {}
    for name, rel in files.items():
        p = Path(rel)
        if not p.is_absolute():
            p = (REPO_ROOT / p) if not p.exists() else p
        inputs[name] = {"path": str(rel), "sha256": sha256_file(p) if p.exists() else None}
    cfg_args = {k: v for k, v in sorted(args.items()) if k not in _NOT_CONFIG}
    config_sha = hashlib.sha256(json.dumps({"inputs": {k: v["sha256"] for k, v in inputs.items()},
                                            "args": cfg_args}, sort_keys=True,
                                           default=str).encode()).hexdigest()
    stalled = [m["run_index"] for m in results if m.get("sim_stalled")]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "batch": out_dir.name,
        "git": git_state(),
        "command": {"argv": list(argv), "args": cfg_args},
        "config": {"inputs": inputs, "config_sha256": config_sha},
        "model": model,
        "seeds": [{"run_index": m.get("run_index"), "seed": m.get("seed"),
                   "victim_sector": m.get("victim_sector")} for m in results],
        "counts": {"planned": runs_planned, "attempted": len(results),
                   "valid": len(results) - len(stalled), "stalled": len(stalled),
                   "stalled_runs": stalled, "passed": sum(1 for p in passed if p),
                   "failed": sum(1 for p in passed if not p)},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "host": socket.gethostname(), "packages": _versions()},
        "files": batch_files(out_dir),
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return {k: manifest[k] for k in ("git", "config", "counts")}


def verify(out_dir) -> list[str]:
    """Differences between the batch folder and its manifest (empty = intact)."""
    out_dir = Path(out_dir)
    p = out_dir / MANIFEST
    if not p.exists():
        return [f"{p} missing"]
    want = json.loads(p.read_text(encoding="utf-8")).get("files") or {}
    have = batch_files(out_dir)
    problems = [f"missing: {f}" for f in sorted(set(want) - set(have))]
    problems += [f"changed: {f}" for f in sorted(set(want) & set(have))
                 if want[f]["sha256"] != have[f]["sha256"]]
    # the runner writes the manifest last, so an extra file was added afterwards
    problems += [f"not in manifest: {f}" for f in sorted(set(have) - set(want))]
    return problems

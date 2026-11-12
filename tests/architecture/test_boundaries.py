"""The folder rules of the package (docs: Repo architecture plan, "Dependency rules").

Each folder of atal/ may import only the folders listed for it in ALLOWED. The two
root utilities (atal/env.py, atal/observability.py) may be imported anywhere. On top of
that, three rules the project is about:

  - truth stays on the simulator and runtime side: agents/, trust_layer/, search/,
    channels/, flight/ and airspace/ never import the scenario (the hidden victim) or
    simulation/
  - the transition manager is trusted flight-safety infrastructure: only flight/ and
    runtime/ use airspace/, so it can never become part of an agent's decision or of the
    trust layer's evidence
  - no private borrowing: a name starting with "_" is never imported from another file

The rules are read from the source with ast, so the test needs no simulator and imports
nothing it checks.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "atal"
UTILITIES = {"env", "observability"}

ALLOWED = {
    "core": set(),
    "search": {"core"},
    "simulation": {"core"},
    "airspace": {"core"},
    "flight": {"core", "search", "airspace"},
    "channels": {"core"},
    "trust_layer": {"core", "search"},
    "agents": {"core", "search", "trust_layer"},
    "attacks": {"core"},
    "runtime": {"core", "search", "simulation", "airspace", "flight", "channels",
                "trust_layer", "agents", "attacks", "evaluation"},
    "evaluation": {"core"},
}
NO_TRUTH = {"agents", "trust_layer", "search", "channels", "flight", "airspace"}
TRUTH_MODULES = ("atal.core.scenario", "atal.simulation")


def _imports():
    """(file, its folder, imported module, imported names) for every atal import."""
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        folder = rel.parts[0] if len(rel.parts) > 1 else rel.stem
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                if node.module == "atal" or node.module.startswith("atal."):
                    yield rel, folder, node.module, [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith("atal."):
                        yield rel, folder, a.name, []


def _target(module: str) -> str:
    return module.split(".")[1] if "." in module else ""


def test_every_folder_has_a_rule():
    folders = {p.name for p in ROOT.iterdir()                 # code folders; configs/ is data
               if p.is_dir() and not p.name.startswith("_") and any(p.glob("*.py"))}
    assert folders == set(ALLOWED), folders ^ set(ALLOWED)


def test_each_folder_imports_only_what_its_rule_allows():
    bad = [f"{rel}: {mod}" for rel, folder, mod, _ in _imports()
           if folder in ALLOWED and _target(mod) != folder
           and _target(mod) not in ALLOWED[folder] | UTILITIES]
    assert not bad, "\n".join(bad)


def test_truth_never_reaches_agent_trust_or_flight_code():
    bad = [f"{rel}: {mod}" for rel, folder, mod, _ in _imports()
           if folder in NO_TRUTH and mod.startswith(TRUTH_MODULES)]
    assert not bad, "\n".join(bad)


def test_only_flight_and_runtime_use_the_transition_manager():
    users = {folder for _, folder, mod, _ in _imports()
             if _target(mod) == "airspace" and folder != "airspace"}
    assert users <= {"flight", "runtime"}, users


def test_no_file_imports_another_files_private_names():
    bad = [f"{rel}: {mod}.{n}" for rel, _, mod, names in _imports()
           for n in names if n.startswith("_")]
    assert not bad, "\n".join(bad)

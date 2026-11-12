"""The reading documents stay true to the code: every module is listed in
docs/architecture.md, every repo path the current documents name exists, every local link
resolves, and the README stays short. docs/history/ and the refactor records are a record
of the past and are not checked."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CURRENT = [REPO / "README.md", *sorted((REPO / "docs").glob("*.md")),
           REPO / "docs" / "recovery" / "recovery_policy.md"]
PATH = re.compile(r"`((?:atal|tests|tools|legacy|docs)/[\w/.\-]+?\.(?:py|md|yaml|jsonc|json))`")
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def _every_file_block() -> str:
    text = (REPO / "docs" / "architecture.md").read_text(encoding="utf-8")
    return text[text.index("## Every file"):]


def test_every_module_is_listed_in_the_architecture_document():
    block = _every_file_block()
    missing = []
    for p in sorted((REPO / "atal").rglob("*.py")):
        if p.name == "__init__.py":
            continue
        rel = p.relative_to(REPO / "atal")
        name = "/".join(rel.parts[1:]) if len(rel.parts) > 1 else rel.name
        listed = re.search(rf"(?<![\w/]){re.escape(name)}\b", block) or rel.as_posix() in block
        if not listed:
            missing.append(str(rel))
    assert not missing, missing


def test_every_path_the_current_documents_name_exists():
    bad = [f"{doc.relative_to(REPO)}: {m}" for doc in CURRENT
           for m in PATH.findall(doc.read_text(encoding="utf-8")) if not (REPO / m).exists()]
    assert not bad, "\n".join(bad)


def test_every_local_link_resolves():
    bad = []
    for doc in CURRENT:
        for target in LINK.findall(doc.read_text(encoding="utf-8")):
            if "://" in target:
                continue
            if not (doc.parent / target).exists():
                bad.append(f"{doc.relative_to(REPO)}: {target}")
    assert not bad, "\n".join(bad)


def test_the_readme_stays_under_100_lines():
    assert len((REPO / "README.md").read_text(encoding="utf-8").splitlines()) < 100

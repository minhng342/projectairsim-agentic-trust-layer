"""Load KEY=VALUE lines from the repo's .env into os.environ (never overriding
variables that are already set). No dependency on python-dotenv. Values are
never printed or logged; callers learn only which keys were set."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env_file(path: Path | str = REPO_ROOT / ".env") -> list[str]:
    path = Path(path)
    if not path.is_file():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded

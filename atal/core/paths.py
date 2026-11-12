"""Where things are: the bundled mission and scene files, and the run folder.

BUNDLED FILES. The sample missions, scenarios and simulator scenes ship inside the
package, in atal/configs/ (missions/sar/*.yaml, sim_config/*.jsonc), and are read through
importlib.resources, so an installed run finds them from any folder. They are read-only:
nothing ever writes into atal/configs/.

NAMING A FILE. A --mission, --scenario or --scene value is either
    bundled:<path inside atal/configs>   e.g. bundled:missions/sar/mission_dev.yaml
    any other path                       your own file, relative to the folder you run in
The runners' defaults are bundled files. A path that does not exist is an error that
names it, the folder it was looked up from, and the bundled files of that kind.

RUN OUTPUT goes to ./runs under the folder the command is run from, unless --out says
otherwise. REPO_ROOT is used only for git provenance and means something only in a
checkout.
"""
import re
from importlib import resources
from pathlib import Path

BUNDLED = "bundled:"
PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_DIR.parent


class ConfigNotFound(FileNotFoundError):
    """A mission, scenario or scene that names no file; the message says what exists."""


def config_root():
    """atal/configs/ as an importlib.resources Traversable (read-only)."""
    return resources.files("atal").joinpath("configs")


def bundled_names(folder: str, suffix: str) -> list[str]:
    """Every bundled file under atal/configs/<folder> ending in suffix, as bundled: names."""
    out, stack = [], [(config_root().joinpath(*folder.split("/")), folder)]
    while stack:
        node, rel = stack.pop()
        for child in node.iterdir():
            name = f"{rel}/{child.name}"
            if child.is_dir():
                stack.append((child, name))
            elif child.name.endswith(suffix):
                out.append(BUNDLED + name)
    return sorted(out)


def _missing(spec: str, folder: str, suffix: str) -> ConfigNotFound:
    names = ", ".join(bundled_names(folder, suffix)) or "(none)"
    return ConfigNotFound(f"{spec}: no such file (looked up from {Path.cwd()}). "
                          f"Bundled {folder} files: {names}")


def read_config(spec, folder: str = "missions", suffix: str = ".yaml") -> tuple[str, bytes]:
    """(label, contents) of a bundled: name or a path (see NAMING A FILE)."""
    spec = str(spec)
    if spec.startswith(BUNDLED):
        node = config_root().joinpath(*spec[len(BUNDLED):].split("/"))
        if not node.is_file():
            raise _missing(spec, folder, suffix)
        return spec, node.read_bytes()
    p = Path(spec)
    if not p.is_file():
        raise _missing(spec, folder, suffix)
    return str(p), p.read_bytes()


def scene_location(spec) -> tuple[Path, str]:
    """(folder, file name) of a scene: a bundled: name or a path. The simulator loads
    scenes by name from a folder, so the robot files a scene refers to must sit next to
    it; check_scene() says which are missing."""
    spec = str(spec)
    if spec.startswith(BUNDLED):
        node = config_root().joinpath(*spec[len(BUNDLED):].split("/"))
        if not node.is_file():
            raise _missing(spec, "sim_config", ".jsonc")
        path = Path(str(node))        # a regular install keeps package files on disk
        if not path.is_file():
            raise ConfigNotFound(f"{spec}: the package is not installed as plain files; "
                                 f"pass the scene as a path instead")
    else:
        path = Path(spec)
        if not path.is_file():
            raise _missing(spec, "sim_config", ".jsonc")
    return path.resolve().parent, path.name


def check_scene(folder: Path, name: str) -> list[str]:
    """The robot-config files `name` refers to that are not in `folder`."""
    text = (folder / name).read_text(encoding="utf-8")
    refs = re.findall(r'"robot-config"\s*:\s*"([^"]+)"', text)
    return sorted({r for r in refs if not (folder / r).is_file()})


def bundled_scene_dir() -> Path:
    """The bundled sim_config folder on disk (the simulator adapter's default)."""
    return scene_location(BUNDLED + "sim_config/scene_basic_drone.jsonc")[0]


def runs_dir() -> Path:
    """Default parent folder for run output: ./runs in the current folder."""
    return Path.cwd() / "runs"

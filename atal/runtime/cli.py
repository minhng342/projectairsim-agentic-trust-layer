"""The `atal` command (installed by pyproject.toml). Each subcommand runs one module's
own command line, with the same options:

    atal mission ...     one-drone run             = python run_sar_mission.py ...
    atal swarm ...       team run                  = python run_sar_swarm.py ...
    atal compare ...     paired batch comparison   = python -m atal.evaluation.compare ...
    atal archive ...     evidence archive          = python -m atal.evaluation.archive ...
    atal plot ...        top-down SVG of a run     = python -m atal.evaluation.plot_tracks ...
    atal summarize ...   readable run summary      = python -m atal.evaluation.summarize_run ...

`atal <subcommand> --help` shows that command's options.
"""
import runpy
import sys

COMMANDS = {
    "mission": "atal.runtime.run_single",
    "swarm": "atal.runtime.run_team",
    "compare": "atal.evaluation.compare",
    "archive": "atal.evaluation.archive",
    "plot": "atal.evaluation.plot_tracks",
    "summarize": "atal.evaluation.summarize_run",
}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__.strip())
        return 0 if argv[:1] in (["-h"], ["--help"]) else 2
    name, rest = argv[0], argv[1:]
    sys.argv = [f"atal {name}", *rest]
    try:
        runpy.run_module(COMMANDS[name], run_name="__main__", alter_sys=True)
    except SystemExit as done:
        return done.code if isinstance(done.code, int) else (0 if done.code is None else 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

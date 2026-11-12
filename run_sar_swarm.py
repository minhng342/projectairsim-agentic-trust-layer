"""Three-drone search and rescue run. The code is in atal/runtime/run_team.py.

    python run_sar_swarm.py --offline --runs 1
"""
import asyncio
import sys

from atal.runtime.run_team import main, parse_args

if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))

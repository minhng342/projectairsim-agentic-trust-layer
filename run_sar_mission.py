"""One-drone search and rescue run. The code is in atal/runtime/run_single.py.

    python run_sar_mission.py --offline --runs 1
"""
import asyncio
import sys

from atal.runtime.run_single import main, parse_args

if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))

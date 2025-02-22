"""One-command experiment: python demo.py  (trains both reward variants in ~2 min on CPU).

Trains PPO on the intended and the misspecified reward, runs the reward-hacking detector and
writes plots, an exploit GIF and a JSON report to ./results (override with HG_OUT_DIR).
Set HG_TOTAL_STEPS to shorten or lengthen training.
"""
import sys
from pathlib import Path

try:
    import hackablegrid  # noqa: F401  (installed via `pip install .`)
except ImportError:  # running from a source checkout after `make build`
    sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))

from hackablegrid.demo import main

if __name__ == "__main__":
    main()

"""CLI entry point for the Kaggle leaderboard snapshot monitor."""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.execv(
        sys.executable,
        (
            sys.executable,
            "-m",
            "ptcg_rl.data.kaggle.leaderboard_monitor",
            *sys.argv[1:],
        ),
    )

"""CLI entry point for recent Kaggle BC source selection."""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.execv(
        sys.executable,
        (
            sys.executable,
            "-m",
            "ptcg_rl.data.kaggle.recent_bc_selection",
            *sys.argv[1:],
        ),
    )

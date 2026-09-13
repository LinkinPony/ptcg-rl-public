"""CLI entry point for compact Kaggle Daily public-environment refreshes."""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.execv(
        sys.executable,
        (
            sys.executable,
            "-m",
            "ptcg_rl.data.kaggle.public_environment_refresh",
            *sys.argv[1:],
        ),
    )

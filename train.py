"""Repository-level launcher for reusable RL training runs."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Run the training launcher with ``src`` available on ``sys.path``."""
    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"
    sys.path.insert(0, str(src_path))

    from ptcg_rl.training.launcher import main as launcher_main

    return launcher_main()


if __name__ == "__main__":
    raise SystemExit(main())

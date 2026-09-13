"""CLI wrapper for exporting runtime checkpoint assets.

Run with an immutable bundle manifest. The evaluated checkpoint bytes are
copied unchanged; this command never resolves ``latest``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ptcg_rl.submission import (
    ReleaseRuntimeCheckpointExportConfig,
    export_release_runtime_checkpoint,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Export a compact runtime checkpoint for Kaggle packaging."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    config = ReleaseRuntimeCheckpointExportConfig(
        release_manifest_path=args.release_manifest,
        output_checkpoint=args.output,
    )
    summary = export_release_runtime_checkpoint(config)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

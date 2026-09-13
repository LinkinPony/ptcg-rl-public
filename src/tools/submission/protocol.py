"""CLI wrapper for Kaggle package building and validation.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/submission/protocol.py \
      --profile random_baseline \
      --allow-draft \
      --output dist/random_baseline.tar.gz \
      --message "random baseline"

Use ``submit_release.py`` for the normal protected upload workflow.
"""

from __future__ import annotations

from ptcg_rl.submission.protocol import main

if __name__ == "__main__":
    raise SystemExit(main())

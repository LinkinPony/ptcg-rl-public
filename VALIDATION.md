# Validation snapshot — 2026-09-13

This snapshot covers the initial sanitized source distribution and the exact LFS
archives identified below. It is evidence from this release, not a permanent
hardware or performance requirement.

| Deck ID | Sanitized archive SHA-256 | Bytes |
|---|---|---:|
| `1628cfd93d1e` | `6a631e2ae905b8a9aaa86b4447cac3ee7efcb6fc46eee7907685966feb8315e8` | 161796494 |
| `13f8a7262a7b` | `24352fee4d1cc24981ceaf00e7e867610a20eb7122874dbdab0aff495b87204b` | 161795559 |

Checks performed:

- Each original archive was bound to its actual Kaggle submission and the original
  SHA-256. Both sanitized archives passed `python -m tools.public_assets.verify`.
- For each archive, all 679 declared unchanged members were compared byte for byte
  with the original. This includes weights, decklist, runtime temperature, and
  inference catalog. Seventy members were removed and two Python files had
  internal remote-path defaults removed. See each `public_release_manifest.json`.
- Both checkpoint pickle metadata sections were inspected without unpickling.
  No internal host paths or credential markers were found in that metadata.
- Gitleaks 8.30.1 scanned the distribution with archive traversal and reported no
  leaks. A separate internal-path/host review found only a false positive in the
  public `nvidia-curand` dependency version `10.4.0.35`. This is a bounded audit,
  not a guarantee that every possible sensitive value can be recognized.
- Both checkpoints loaded with strict state-dict matching on CUDA, using PyTorch
  2.13.0+cu130 on an NVIDIA H200. Each produced finite value-head outputs on a
  synthetic input. These checks do not establish full-game deployment parity.
- The two synthetic family-migration tests passed. These small tests ran on CPU;
  they do not perform training or benchmark inference.
- The fresh-run Hydra configuration passed Pydantic validation and resolved
  without launching a learner. The root launcher's `--dry-run` passed.
- The Python wheel built successfully. CLI examples assume an editable install
  from the repository checkout, where Hydra configuration files remain available.
- Ruff passed for all distributed Python source and tests. Focused Mypy passed
  for the 12 new/modified release, deployment, and test files with imported modules
  checked separately by the full-source run.

## Existing typing findings

The full `mypy src tests` run checked 858 source files and reported **12 existing
errors in eight files**. They are not hidden with global ignores. The findings
include dynamic PyArrow compute typing, optional Kaggle imports, obsolete ignore
comments, a dataclass replacement annotation, and legacy diagnostic/ladder protocol
mismatches. They were present in the input implementation and were not expanded
into unrelated behavior repairs during this migration.

The simulator and engine-linked libraries were deliberately excluded. No complete
match, training run, or Kaggle resubmission was performed as part of this release.

## Privacy review follow-up — 2026-09-13

A further review removed an account-bound Kaggle diagnostic and a private native
build record from the source selection. Native build documentation now requires
locally supplied, authorized artifacts. These changes affect source distribution
only; both LFS archives and their original checkpoint/deck fingerprints remain
unchanged. The distribution retains an independent single-commit history.

The follow-up inspection covered every tracked file, both remotely fetched LFS
archives (including normalized archive headers), checkpoint pickle strings
without unpickling, and all catalog string arrays. Gitleaks 8.30.1 found no
credentials in source history, nested archives, or decoded checkpoint metadata.
Numeric model tensors were inventoried, not interpreted as arbitrary text.
Dependency attribution and hash-domain strings were reviewed separately from
personal contact details and absolute filesystem paths.

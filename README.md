# PTCG RL

Source code and final model artifacts for our Pokemon TCG AI Battle Kaggle agent.
The project contains shared and deck-specific Transformer policies, temporal
memory, behavior cloning, PPO training, engine-backed action features, distributed
collection, evaluation, and deployment tooling.

This distribution starts with a clean Git history. Internal operations documents,
run histories, credentials, and the writeup are excluded. The Python source tree
is the maintained implementation; each submission archive preserves the runtime
source that accompanied that particular submitted policy, with the sanitization
described below.

## Final submissions

| Deck | Original compact deck ID | Kaggle submission | Download with Git LFS |
|---|---|---|---|
| Hydrapple | `1628cfd93d1e` | `55564817` | [Sanitized v1032 bundle](submissions/55564817_1628cfd93d1e.tar.gz) |
| Dragapult | `13f8a7262a7b` | `55565209` | [Sanitized v1032 bundle](submissions/55565209_13f8a7262a7b.tar.gz) |

These are sanitized distributions of our final two submissions, **not the exact
archives uploaded to Kaggle**. The checkpoint, decklist, runtime temperature, and
inference catalog retain their original bytes. Restricted simulator components,
vendored opponent agents, internal provenance, and unrelated frontend files were
removed. Two runtime modules have machine-specific remote paths removed.

[The manifest](submissions/manifest.json) records original submission fingerprints,
original archive SHA-256 values, checkpoint and deck checksums, and the checksums
of these new sanitized archives. The original `deck_hash` is preserved independently
of the full internal `deck_digest`.

```bash
git lfs install
git clone https://github.com/LinkinPony/ptcg-rl-public.git
cd ptcg-rl-public
git lfs pull --include="submissions/*.tar.gz"
```

GitHub's source ZIP should not be used as a substitute for retrieving LFS objects.
A tiny text file beginning with `version https://git-lfs.github.com/spec/v1` is
an LFS pointer, not a submission archive.

## Install and verify

Use Python 3.11 or newer and a PyTorch installation appropriate for your CUDA
driver. The source uses PyTorch APIs for packed attention and compilation; the
exact validation environment is recorded in [VALIDATION.md](VALIDATION.md).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
PYTHONPATH=src python -m tools.public_assets.verify
PYTHONPATH=src python -m pytest -q tests
```

The verification command streams both archives, checks hashes and member safety,
and confirms that the original checkpoint and deck bytes are intact. It does not
load pickle files or require the simulator.

## Source map

| Directory | Purpose |
|---|---|
| `src/ptcg_rl/model/` | Shared, family, exact-deck, temporal, and action-decoder models |
| `src/ptcg_rl/training/`, `src/ptcg_rl/rl/` | Behavior cloning, PPO, checkpoints, and collection |
| `src/native/` | Project-owned C++ adapters; requires a separately authorized engine |
| `src/ptcg_rl/agent/` | Deployment policy and runtime |
| `src/ptcg_rl/evaluation/` | Match runners, bundle evaluation, and scoring |
| `src/ptcg_rl/dashboard/` | Local monitoring API and frontend sources |
| `src/tools/` | Command-line training, data, evaluation, and release tools |
| `configs/` | Selected Hydra examples and final model references |

Configuration is managed with Hydra and validated by Pydantic. Operational examples
are starting points for new experiments, not exact-resume profiles for our private
training history. The final artifacts are fixed-deck inference exports and do not
contain optimizer state or the complete routed training checkpoint.

Resolve the example training configuration without starting a run:

```bash
PYTHONPATH=src python src/tools/rl_train.py --cfg job --resolve
python train.py --dry-run --run-version example
```

The final-family model example uses native CUDA collection and fresh expert
lineages. Training requires the separately authorized simulator, its matching
card metadata, and locally prepared training inputs. The included catalog can be
prepared at the example's expected path with:

```bash
mkdir -p data/external/final
tar -xzf submissions/55564817_1628cfd93d1e.tar.gz \
  -C data/external/final public_catalog
```

Generate static card features only in a separately authorized SDK environment:

```bash
PYTHONPATH=data/sample_submission:src python -m ptcg_rl.cards.static_features
```

The example's dormant random-opponent manifest records the simulator fingerprint
used when preparing this distribution. Rebind it explicitly if using a different
authorized simulator build. Historical frozen opponent
checkpoints and raw replay datasets are not included.

## Simulator and inference

The original deployment needs the competition's `cg` SDK and a matching native
probe library. Neither is redistributed here because of the simulator license.
Consequently, extracting one of these archives alone does **not** produce a
standalone runnable agent. See [NOTICE.md](NOTICE.md) for the licensing boundary.

Where separately authorized, the build interfaces accept the engine directory:

```bash
make -C src/native/cg_probe ENGINE_DIR="/path/to/authorized/engine"
make -C src/native/cg_train ENGINE_DIR="/path/to/authorized/engine"
```

An authorized runtime must supply the compatible `cg` package and
`src/native/cg_probe/libcg_probe.so` in the extracted submission layout. The
checkpoint uses engine-derived facts; disabling those features changes the policy
and is not an equivalent reproduction. Runtime code bundled with the final
artifacts is the reference for their deployment behavior.

For a long training run, start the launcher in a named tmux session. Distributed
collection must use a private network, explicit worker configuration, and immutable
source snapshots. GPU capacity and batch sizes must be selected for the available
hardware. Remote evaluation requires explicit host/repository arguments; local
evaluation uses `--local-only`.

## License

Our code and trained weights use the [MIT license](LICENSE). Third-party and
game-content rights are described in [NOTICE.md](NOTICE.md).

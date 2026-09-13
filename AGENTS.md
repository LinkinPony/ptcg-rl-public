# Repository boundaries

- This is an independent distribution with clean history. Never merge or push
  private development history into it.
- Keep `docs/writeup/` and all writeup drafts, exports, assets, and generation
  tools out of this repository.
- Do not add the competition engine, `cg` SDK, engine-linked native binaries,
  or third-party agents without establishing redistribution permission.
- Keep source code under `src/`, configuration under `configs/`, disposable
  files under `tmp/`, and reusable external inputs under ignored `data/external/`.
- Do not add credentials, local host inventories, internal absolute paths,
  raw operational logs, or private run histories.
- Submission archives belong in Git LFS. Preserve original weights, decklists,
  submission references, and fingerprints. Document sanitization explicitly;
  never label a modified archive as the exact original Kaggle upload.
- Preserve the authoritative compact `deck_hash` for display and traceability.
  A canonical `deck_digest` or its prefix is not a substitute.
- Game semantics come from the authorized simulator, not new per-card formulas.
- Use Hydra and Pydantic for configuration. Run Ruff and Mypy for changed code.
- Tests must use minimal synthetic inputs and check behavior. Never load real
  repository profiles or assert operational configuration values in tests.
- Prefer CUDA for supported model workloads. Keep formal training in tmux;
  a failed learner or worker must not be automatically restarted.
- Repository visibility is an operator decision. Do not change it implicitly.

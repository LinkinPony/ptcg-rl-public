# cg_probe native backend

This directory builds the native dynamic-effect probe backend used by training
and engine-fact-enabled Kaggle inference. It includes the bundled engine headers
from `data/ptcg_engine/ptcgProgram 22` and does not modify engine sources or
`data/sample_submission/cg`.

Build:

```bash
make -C src/native/cg_probe
```

This distribution does not include the simulator, a compiled probe, or
private build records. Obtain the required simulator authorization and
produce a compatible native library and its validation manifest locally.
The release builder validates the supplied library against that manifest;
it must not be treated as an included deployment artifact.

Production engine facts require the ragged `CgProbeFactBatch` ABI and fail
closed when the symbol, ABI version, feature width, or library fingerprint
differs. There is no Python payload/feature fallback in the training path.

The same library exports the production `CgPlanner*` ABI v6. A planner lane owns
one mutable engine session and accepts one ragged complete-selection-by-world
grid. Each cell advances exactly one complete selection and any following
uniquely forced prompts, stopping at the next strategic prompt, same-seat
`MAIN`, turn handoff, chance prompt, or terminal result. The payload is
candidate-major and contains contiguous metadata plus root-visible and
leaf-actor-visible JSON bytes. Each request supplies an explicit 64-bit
stochastic seed. Candidates for one world start from the same derived engine
RNG stream, different worlds use independent derived streams, and repeating
the identical request is reproducible. The payload also carries two independent
SHA-256 identities: a digest recomputed from every raw buffer and execution
limit consumed by native code (including the stochastic seed), and the
producer's semantic contract digest. Python verifies the exported ABI
descriptor, both identities, and the loaded library fingerprint before
accepting evidence.

The separate `CgPlannerSession*` ABI v5 supports exact continuation across
later strategic or manual-coin prompts without putting engine state on a
Python or IPC wire. `CgPlannerOpenSession` pins one bounded generation to a
lane and returns opaque `(generation, state_slot)` handles only for
strategic/chance endpoints. Aligned continuation calls may branch a handle;
callers explicitly release unused handles and close the generation. Slots are
not reused within a generation, and the Python wrapper destroys a tainted lane
after any ambiguous call or payload failure. Continuation starts from the
stored fully determinized state; it does not reapply the initial hidden-prize
face-down repair and thereby overwrite legitimate later reveal state.

The planner intentionally returns error `92` when a branch consumes engine RNG
that cannot be represented by the request's common scenario support. Manual
coin prompts remain visible as chance endpoints. This is an explicit base-policy
fallback surface, not an approximate rules result.

`CgProbeMacroBatch` is an additional benchmark-oriented C ABI for executing
fully specified multi-step action macros through the same native Search
implementation. Build it under a separate output path when profiling so an
active training process never observes a library replacement:

```bash
make -C src/native/cg_probe \
  TARGET=../../../outputs/engine/native_exact_consequence_benchmark/libcg_probe_benchmark.so
```

The reproducible replay benchmark is
`src/tools/native_exact_consequence_benchmark.py`. The macro API does not
choose strategically meaningful continuations; callers must provide them, or
explicitly request the minimum legal deterministic completion used by the cost
probe.

An explicit macro is not allowed to cross its root semantic boundary. If the
root player hands off, reaches `MAIN` again, or reaches terminal while supplied
steps remain, the backend stops before the trailing action and returns
per-transition error `1001` with the reached endpoint preserved. Python callers
should compare against `NATIVE_MACRO_TRAILING_ACTION_ERROR`.

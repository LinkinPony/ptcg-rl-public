# Native training arena

`libcg_train.so` is a training-only C++20 vector arena over the bundled
simulator. It owns opaque, fixed-capacity lanes whose initialized
`BattleData` objects keep the same address across resets and steps.

Build from the repository root:

```bash
make -C src/native/cg_train
```

The ABI is declared in `cg_train.h`. `CgTrainReset` consumes contiguous
`decks[N][2][60]` plus explicit `uint32` seeds. `CgTrainStep` consumes slot
ids and complete selections in CSR form. Both write caller-owned SoA buffers;
no API path serializes JSON.

Calls validate the whole request before mutation. Each valid transition is
first evaluated on a private copy, including the `selectMax == 0` forced
chain. Output option capacity is checked against those copies before any slot
is committed, so an insufficient-capacity return leaves the arena unchanged.
Illegal actions and invalid decks are reported per row and leave that row's
slot unchanged.

## Stateful rollout encoder ABI v1

`cg_rollout_encoder.h` declares a separate training-only ABI layered on the
public projection produced by `cg_train.h`. It does not call or replace game
rules. One `CgTrainRolloutEncoder` consumes raw, unselected arena outputs,
incrementally tracks perspective-specific public history and catalog evidence,
then writes selected ready slots directly into model-ready state, option, deck
and belief tensors. Learner-only known evidence is available through a separate
CSR plan/write pair.

The lifecycle is:

1. Check `CgTrainRolloutGetAbiDescriptor`, then create one encoder for an arena
   slot capacity with `CgTrainRolloutCreate`.
2. After each successful arena reset or step, call the matching
   `CgTrainRolloutConsumeReset` or `CgTrainRolloutConsumeStep` before selecting
   rows. Clear retired slots with `CgTrainRolloutClearSlots`.
3. Call `CgTrainRolloutPlanRows`, allocate a caller-owned contiguous output,
   and call `CgTrainRolloutEncodeRows`. No consume or clear may intervene
   between plan and encode. `row_offset` and `belief_row_offset` allow several
   encoder sources to fill one shared batch.
4. Destroy the handle exactly once with `CgTrainRolloutDestroy`.

Creation copies every catalog column and supporter ID, so creation-time input
pointers need not remain alive. The encoder owns its catalog copy, posterior
cache, per-slot current snapshot and both perspective histories. Arena source
buffers and every `CgTrainRolloutOutput` tensor remain caller-owned and need
only remain valid for the call. The C wrapper serializes individual calls.
Multi-call sequences such as plan/encode still require external sequencing so
no mutation can intervene. Destruction must happen exactly once after all calls
have returned and must never race an operation on the same handle.

Consume calls validate the complete source CSR, slot set, reset decks, public
metadata and model schema against staged slot copies. A rejected call does not
commit any slot snapshot or history; `ClearSlots` likewise validates the full
slot set before clearing it. Plan and encode require unique, in-range slots
whose requested perspective matches the current ready prompt. Encode validates
the complete output descriptor, pointers, widths, offsets and capacities
before writing caller memory. Posterior-cache warming is not semantic state and
may occur during otherwise read-only planning.

The rollout descriptor binds more than C struct sizes. Loaders must require the
ABI magic/version, all declared struct sizes and tensor widths, the complete
feature-bit set, and the `model_encoding_fingerprint` published by
`rollout_export.cpp`. That fingerprint must change whenever token ordering,
feature indices, scaling, masks, option semantics or belief layout changes.
The public-deck catalog fingerprint and model input-contract fingerprint remain
control-plane identities: the Python bridge validates and carries them with the
model batch, while the C++ constructor validates and copies the corresponding
numeric catalog. A matching ABI fingerprint alone must never be used as proof
that a checkpoint and catalog belong together.

The native implementation must retain the Python path as a correctness oracle.
The focused live-engine parity test compares reordered selected slots across
successive steps, including state/options/deck/belief tensors and known public
evidence:

```bash
make -C src/native/cg_train
PYTHONPATH=data/sample_submission:src \
  python -m pytest -q \
  tests/test_native_rollout_encoder.py \
  tests/test_public_deck_catalog.py
```

Benchmark evidence must use the production
`encode_native_rollout_sources` path and record the checkpoint, catalog and
shared-library hashes. A short benchmark proves wiring and stage attribution;
only a production-shaped complete rollout window can decide training
throughput.

## ABI v4 public state, log deltas, and transition accounting

ABI v2 appended public-current-state columns to the v1 output prefix. ABI v3
preserved that prefix and appended public log deltas. ABI v4 preserves the
entire v3 prefix and appends one `selection_advance_count` scalar per row.
Scalar columns are absolute player 0/1 values; `select_player` identifies the
acting perspective. Cards and attachments are flattened CSR tables:

- `visible_card_offsets` partitions card rows by game. Card `area` uses the
  engine's public integer area values. Select-deck cards use `Deck`; select
  `contextCard` and `effect` use virtual area zero with indices zero and one.
- Hidden active, bench, prize, or redacted-looking entries preserve the public
  list position but have `card_id == 0` and `serial == 0`. No other field is
  read from the hidden card for those rows.
- The opponent hand has no card rows. Its public count remains available in
  `player1_hand_count` or `player0_hand_count`.
- `attachment_offsets` partitions attachments by game and
  `attachment_parent` points into the absolute flattened card table. Energy
  rows also carry the effective energy type and unit count used by
  `PokemonJson`; tools and pre-evolutions use `-1, 0`.
- `looking_mode` distinguishes JSON `null`, a visible list, and a same-length
  list of JSON nulls. `context_card_row` and `effect_card_row` use
  `UINT32_MAX` when absent.

The implementation deliberately follows `CardJson`, `PokemonJson`,
`PlayerJson`, `Current`, and `SelectJson` in the bundled `ToJson.h`, while
writing primitive arrays directly. In particular, it does not expose
`State`, the opponent's hidden hand/deck identities, or face-down card
identities.

`log_offsets` partitions logs by output row. `log_type`, `log_param_count`,
and `log_p0` through `log_p6` are the fixed-width SoA equivalent of
`LogJson(..., select_player, false)`. Parameters are in the same order as the
JSON object's fields after `type`; unused parameter columns are zero.
Privacy projection is applied before writing: opponent `Draw` becomes
`DrawReverse`, and a non-public `MoveCard` becomes `MoveCardReverse` with
only player/from/to parameters.

Each engine state retains one cursor per player, matching
`State::nextLogStart()`. A successful reset or step emits only the acting
perspective's delta and advances only that perspective's cursor. Invalid
actions, engine/reset errors, terminal re-steps, and any capacity failure emit
no delta and consume no cursor. `recordLog` is enabled, but raw `Log` objects
and their private parameters never cross the ABI.

On a successful reset, `selection_advance_count` is the number of
`selectMax == 0` callbacks consumed after the initial engine advance. On a
successful step it is one for the submitted complete selection plus every
subsequent `selectMax == 0` callback consumed before the next strategic
prompt. Invalid actions, reset/engine errors, terminal re-steps, and calls
rejected before commit report zero. As with every other output column, a
whole-call capacity failure writes nothing, including this counter.

For parity debugging, replay an identical explicit-seed action trace through
the arena and a debug-only engine build that calls `ToJsonApi`, then compare:
global/select scalars, zone list lengths and null positions, visible
`id/serial/playerIndex`, Pokémon HP fields, attachment lists, and effective
energy values. Also assert that every JSON null maps to zero identity and that
no opponent-hand row exists. JSON generation is intentionally not compiled
into or callable from the training ABI.

`make -C src/native/cg_train parity-probe` builds that debug-only reference
executable at `tmp/cg_train_parity_probe`. It accepts `SEED MAX_STEPS`, reads
two whitespace-separated 60-card decks from stdin, follows the deterministic
minimum-legal-selection trace, and emits one canonical `ToJsonApi` object per
line, including perspective-specific log deltas. Its stderr emits one
`CG_TRAIN_SELECTION_ADVANCE_COUNT <n>` record aligned with each stdout
observation so debug parity checks can independently verify the ABI v4
counter. The target is deliberately separate from `libcg_train.so`.

`make -C src/native/cg_train log-parity-probe` builds an exhaustive synthetic
projection reference at `tmp/cg_train_log_parity_probe`. It emits paired
`LogJson` and fixed-column projections for every public log type and both
acting perspectives, with all `Draw` and `MoveCard` privacy branches. These
debug executables are evidence tools, not training dependencies.

The v4 descriptor advertises `PUBLIC_LOG_DELTA` and
`SELECTION_ADVANCE_COUNT`; the legacy v2 `NO_HISTORY` feature bit is
intentionally unset.

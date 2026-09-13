// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®,
// and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use
// only; the full license is in data/ptcg_engine/ptcgProgram 22/LICENSES/.

#ifndef SRC_NATIVE_CG_TRAIN_CG_TRAIN_H_
#define SRC_NATIVE_CG_TRAIN_CG_TRAIN_H_

#include <stdint.h>

#include "quota_scheduler.h"

#if defined(_WIN32)
#define CG_TRAIN_API __declspec(dllexport)
#else
#define CG_TRAIN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CgTrainLane CgTrainLane;

enum CgTrainReturnCode {
  CG_TRAIN_OK = 0,
  CG_TRAIN_INVALID_ARGUMENT = -1,
  CG_TRAIN_INSUFFICIENT_CAPACITY = -2,
  CG_TRAIN_ENGINE_EXCEPTION = -3,
};

enum CgTrainSlotStatus {
  CG_TRAIN_SLOT_UNINITIALIZED = 0,
  CG_TRAIN_SLOT_READY = 1,
  CG_TRAIN_SLOT_FINISHED = 2,
  CG_TRAIN_SLOT_ACTION_ERROR = 3,
  CG_TRAIN_SLOT_RESET_ERROR = 4,
  CG_TRAIN_SLOT_ENGINE_ERROR = 5,
};

enum CgTrainSlotError {
  CG_TRAIN_SLOT_ERROR_NONE = 0,

  // Values 4, 5, and 6 are returned unchanged from State::checkPlayerSelect().
  CG_TRAIN_SLOT_ERROR_WRONG_ACTION_COUNT = 4,
  CG_TRAIN_SLOT_ERROR_OPTION_OUT_OF_RANGE = 5,
  CG_TRAIN_SLOT_ERROR_DUPLICATE_OPTION = 6,

  CG_TRAIN_SLOT_ERROR_UNKNOWN_CARD = 101,
  CG_TRAIN_SLOT_ERROR_TOO_MANY_COPIES = 102,
  CG_TRAIN_SLOT_ERROR_NO_BASIC_POKEMON = 103,
  CG_TRAIN_SLOT_ERROR_TOO_MANY_ACE_SPEC = 104,
  CG_TRAIN_SLOT_ERROR_TERMINAL = 105,
  CG_TRAIN_SLOT_ERROR_ENGINE_EXCEPTION = 199,
};

enum CgTrainFeature {
  CG_TRAIN_FEATURE_EXPLICIT_SEEDS = UINT64_C(1) << 0,
  CG_TRAIN_FEATURE_CSR_ACTIONS = UINT64_C(1) << 1,
  CG_TRAIN_FEATURE_CALLER_OWNED_SOA = UINT64_C(1) << 2,
  CG_TRAIN_FEATURE_STABLE_SLOT_ADDRESS = UINT64_C(1) << 3,
  CG_TRAIN_FEATURE_FORCED_CHAIN = UINT64_C(1) << 4,
  CG_TRAIN_FEATURE_NO_JSON = UINT64_C(1) << 5,
  CG_TRAIN_FEATURE_ATOMIC_CAPACITY_FAILURE = UINT64_C(1) << 6,
  CG_TRAIN_FEATURE_PUBLIC_CURRENT_STATE = UINT64_C(1) << 7,
  CG_TRAIN_FEATURE_VISIBLE_CARD_CSR = UINT64_C(1) << 8,
  CG_TRAIN_FEATURE_ATTACHMENT_CSR = UINT64_C(1) << 9,
  // ABI v2 advertised this bit. ABI v3 and later leave it unset.
  CG_TRAIN_FEATURE_NO_HISTORY = UINT64_C(1) << 10,
  CG_TRAIN_FEATURE_PUBLIC_LOG_DELTA = UINT64_C(1) << 11,
  CG_TRAIN_FEATURE_SELECTION_ADVANCE_COUNT = UINT64_C(1) << 12,
  CG_TRAIN_FEATURE_PARALLEL_ROWS = UINT64_C(1) << 13,
  CG_TRAIN_FEATURE_PUBLIC_STATE_TOKEN_EXPORT = UINT64_C(1) << 14,
  CG_TRAIN_FEATURE_PUBLIC_OBSERVATION_EXPORT = UINT64_C(1) << 15,
  CG_TRAIN_FEATURE_SLOT_RELEASE = UINT64_C(1) << 16,
};

enum CgTrainTurnFlag {
  CG_TRAIN_TURN_SUPPORTER_PLAYED = UINT32_C(1) << 0,
  CG_TRAIN_TURN_STADIUM_PLAYED = UINT32_C(1) << 1,
  CG_TRAIN_TURN_ENERGY_ATTACHED = UINT32_C(1) << 2,
  CG_TRAIN_TURN_RETREATED = UINT32_C(1) << 3,
};

enum CgTrainPlayerStatusFlag {
  CG_TRAIN_PLAYER_POISONED = UINT32_C(1) << 0,
  CG_TRAIN_PLAYER_BURNED = UINT32_C(1) << 1,
  CG_TRAIN_PLAYER_ASLEEP = UINT32_C(1) << 2,
  CG_TRAIN_PLAYER_PARALYZED = UINT32_C(1) << 3,
  CG_TRAIN_PLAYER_CONFUSED = UINT32_C(1) << 4,
};

// The public card area uses the integer values exposed by the engine API.
// Area zero is reserved for the two Select virtual cards:
// area_index == 0 is contextCard, and area_index == 1 is effect.
enum CgTrainPublicCardArea {
  CG_TRAIN_CARD_AREA_VIRTUAL = 0,
  CG_TRAIN_CARD_AREA_DECK = 1,
  CG_TRAIN_CARD_AREA_HAND = 2,
  CG_TRAIN_CARD_AREA_DISCARD = 3,
  CG_TRAIN_CARD_AREA_ACTIVE = 4,
  CG_TRAIN_CARD_AREA_BENCH = 5,
  CG_TRAIN_CARD_AREA_PRIZE = 6,
  CG_TRAIN_CARD_AREA_STADIUM = 7,
  CG_TRAIN_CARD_AREA_LOOKING = 12,
};

enum CgTrainLookingMode {
  // Matches Current(...): "looking": null.
  CG_TRAIN_LOOKING_NULL = 0,
  // Matches a visible array of Card objects.
  CG_TRAIN_LOOKING_VISIBLE = 1,
  // Matches an array whose entries are null. The corresponding visible-card
  // rows have zero card id and serial.
  CG_TRAIN_LOOKING_REDACTED = 2,
};

enum CgTrainAttachmentKind {
  CG_TRAIN_ATTACHMENT_ENERGY = 1,
  CG_TRAIN_ATTACHMENT_TOOL = 2,
  CG_TRAIN_ATTACHMENT_PRE_EVOLUTION = 3,
};

typedef struct CgTrainAbiDescriptor {
  uint32_t magic;
  uint32_t abi_version;
  uint32_t descriptor_size;
  uint32_t output_size;
  uint32_t deck_size;
  uint32_t players_per_game;
  uint32_t option_param_count;
  uint32_t option_type_count;
  uint32_t max_lane_capacity;
  uint32_t reserved;
  uint64_t features;
} CgTrainAbiDescriptor;

// Every pointer is caller-owned. Scalar row arrays have at least slot_capacity
// elements. *_offsets arrays have at least slot_capacity + 1 elements.
// Option, visible-card, and attachment value arrays have their corresponding
// capacities. Public player columns use absolute player 0/1; select_player is
// the acting perspective.
typedef struct CgTrainOutput {
  uint32_t struct_size;
  uint32_t slot_capacity;
  uint32_t option_capacity;
  uint32_t reserved;

  int32_t* status;
  int32_t* error;
  int32_t* select_player;
  int32_t* select_type;
  int32_t* select_context;
  int32_t* select_min;
  int32_t* select_max;
  int32_t* result;
  int32_t* turn;

  uint32_t* option_offsets;
  int32_t* option_type;
  int32_t* option_p0;
  int32_t* option_p1;
  int32_t* option_p2;
  int32_t* option_p3;
  int32_t* option_p4;

  // ABI v2 capacities. Capacity validation happens before any slot or output
  // buffer is modified.
  uint32_t visible_card_capacity;
  uint32_t attachment_capacity;
  uint32_t reserved_v2_0;
  uint32_t reserved_v2_1;

  int32_t* turn_action_count;
  int32_t* first_player;
  uint32_t* turn_flags;
  int32_t* remain_damage_counter;
  int32_t* remain_energy_cost;

  int32_t* player0_deck_count;
  int32_t* player1_deck_count;
  int32_t* player0_hand_count;
  int32_t* player1_hand_count;
  int32_t* player0_prize_count;
  int32_t* player1_prize_count;
  int32_t* player0_bench_max;
  int32_t* player1_bench_max;
  uint32_t* player0_status_flags;
  uint32_t* player1_status_flags;

  int32_t* looking_mode;
  int32_t* select_deck_visible;
  // Absolute indices into the flattened visible-card arrays, or UINT32_MAX.
  uint32_t* context_card_row;
  uint32_t* effect_card_row;

  uint32_t* visible_card_offsets;
  int32_t* visible_card_owner;
  int32_t* visible_card_area;
  int32_t* visible_card_area_index;
  int32_t* visible_card_id;
  int32_t* visible_card_serial;
  int32_t* visible_card_hp;
  int32_t* visible_card_max_hp;
  int32_t* visible_card_appear_this_turn;

  // attachment_offsets is CSR by output row. attachment_parent is an absolute
  // index into the flattened visible-card arrays.
  uint32_t* attachment_offsets;
  uint32_t* attachment_parent;
  int32_t* attachment_kind;
  int32_t* attachment_card_id;
  int32_t* attachment_card_serial;
  // For energy cards these reproduce PokemonJson's effective `energies`
  // values without JSON. Non-energy attachments use -1 and zero.
  int32_t* attachment_energy_type;
  int32_t* attachment_energy_units;

  // ABI v3 public log delta. Logs are CSR by output row. Each row is
  // projected for that row's select_player and contains only entries since
  // the previous successful output for that same perspective.
  uint32_t log_capacity;
  uint32_t reserved_v3;
  uint32_t* log_offsets;
  int32_t* log_type;
  uint32_t* log_param_count;
  int32_t* log_p0;
  int32_t* log_p1;
  int32_t* log_p2;
  int32_t* log_p3;
  int32_t* log_p4;
  int32_t* log_p5;
  int32_t* log_p6;

  // ABI v4 transition accounting. A successful reset reports the number of
  // selectMax == 0 callbacks consumed after the initial engine advance. A
  // successful step reports one submitted callback plus every subsequently
  // consumed selectMax == 0 callback. Failed or uncommitted rows report zero.
  uint32_t* selection_advance_count;
} CgTrainOutput;

CG_TRAIN_API const CgTrainAbiDescriptor* CgTrainGetAbiDescriptor(void);
CG_TRAIN_API const char* CgTrainLastError(void);
CG_TRAIN_API int32_t CgTrainInitialize(void);

CG_TRAIN_API CgTrainLane* CgTrainCreate(uint32_t capacity);
// worker_count == 0 selects the affinity-sized automatic value. This optional
// constructor is ABI-compatible with CgTrainCreate and is primarily useful for
// deterministic parity probes and process-level resource partitioning.
CG_TRAIN_API CgTrainLane* CgTrainCreateWithWorkers(
    uint32_t capacity, uint32_t worker_count);
CG_TRAIN_API void CgTrainDestroy(CgTrainLane* lane);
CG_TRAIN_API uint32_t CgTrainCapacity(const CgTrainLane* lane);
CG_TRAIN_API uint32_t CgTrainWorkerCount(const CgTrainLane* lane);

// Destroy selected live BattleData objects once their games have retired.
// Each slot id must be unique and less than lane capacity. The fixed lane and
// its caller-owned output buffers remain reusable by a later CgTrainReset.
CG_TRAIN_API int32_t CgTrainClearSlots(
    CgTrainLane* lane, uint32_t batch_count, const uint32_t* slots);

// decks is contiguous row-major [batch_count][2][60]. seeds has batch_count
// entries. Each slot id must be unique and less than lane capacity.
CG_TRAIN_API int32_t CgTrainReset(
    CgTrainLane* lane, uint32_t batch_count, const uint32_t* slots,
    const int32_t* decks, const uint32_t* seeds, CgTrainOutput* output);

// Actions are complete selections encoded as CSR. action_offsets has
// batch_count + 1 entries, begins at zero, is monotonic, and ends at
// action_count. Each slot id must be unique and initialized.
CG_TRAIN_API int32_t CgTrainStep(
    CgTrainLane* lane, uint32_t batch_count, const uint32_t* slots,
    const uint32_t* action_offsets, const int32_t* actions,
    uint32_t action_count, CgTrainOutput* output);

// Export the same privacy-erased base64 state token as ApiGetBattleData for
// each selected slot. token_offsets has batch_count + 1 entries. On
// CG_TRAIN_INSUFFICIENT_CAPACITY, the required offsets are still returned and
// token_data is not modified. The operation never mutates engine state or log
// cursors.
CG_TRAIN_API int32_t CgTrainExportPublicStateTokens(
    CgTrainLane* lane, uint32_t batch_count, const uint32_t* slots,
    uint32_t token_capacity, uint32_t* token_offsets, char* token_data);

// Export the exact privacy-safe JSON observation captured by the most recent
// reset/step output for each slot, before its public-log cursor was advanced.
// observation_offsets follows the same CSR capacity protocol as state tokens.
CG_TRAIN_API int32_t CgTrainExportPublicObservations(
    CgTrainLane* lane, uint32_t batch_count, const uint32_t* slots,
    uint32_t observation_capacity, uint32_t* observation_offsets,
    char* observation_data);

// Quota scheduler ABI. Rows are aggregate assignment cells; only returned row
// indices are materialized into per-game leases by the controller.
CG_TRAIN_API const char* CgTrainQuotaSchedulerLastError(void);
CG_TRAIN_API CgTrainQuotaScheduler* CgTrainQuotaSchedulerCreate(
    const CgTrainQuotaRow* rows, uint32_t row_count, uint32_t artifact_count);
CG_TRAIN_API void CgTrainQuotaSchedulerDestroy(
    CgTrainQuotaScheduler* scheduler);
CG_TRAIN_API uint64_t CgTrainQuotaSchedulerRemaining(
    const CgTrainQuotaScheduler* scheduler);
CG_TRAIN_API int32_t CgTrainQuotaSchedulerCanTake(
    const CgTrainQuotaScheduler* scheduler, uint32_t game_count,
    uint32_t frozen_artifact_limit, uint32_t required_artifact_slot);
CG_TRAIN_API int32_t CgTrainQuotaSchedulerTake(
    CgTrainQuotaScheduler* scheduler, uint32_t game_count,
    uint32_t frozen_artifact_limit, uint32_t required_artifact_slot,
    uint32_t* output_row_indices, uint32_t output_capacity,
    uint32_t* output_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SRC_NATIVE_CG_TRAIN_CG_TRAIN_H_

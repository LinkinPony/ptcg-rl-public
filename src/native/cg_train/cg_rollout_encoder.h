// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®,
// and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use
// only; the full license is in data/ptcg_engine/ptcgProgram 22/LICENSES/.

#ifndef SRC_NATIVE_CG_TRAIN_CG_ROLLOUT_ENCODER_H_
#define SRC_NATIVE_CG_TRAIN_CG_ROLLOUT_ENCODER_H_

#include <stdint.h>

#include "cg_train.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CgTrainRolloutEncoder CgTrainRolloutEncoder;

enum CgTrainRolloutFeature {
  CG_TRAIN_ROLLOUT_FEATURE_TRANSACTIONAL_HISTORY = UINT64_C(1) << 0,
  CG_TRAIN_ROLLOUT_FEATURE_DIRECT_SLOT_SELECTION = UINT64_C(1) << 1,
  CG_TRAIN_ROLLOUT_FEATURE_MODEL_READY_TENSORS = UINT64_C(1) << 2,
  CG_TRAIN_ROLLOUT_FEATURE_PUBLIC_CATALOG = UINT64_C(1) << 3,
  CG_TRAIN_ROLLOUT_FEATURE_CALLER_OWNED_OUTPUT = UINT64_C(1) << 4,
};

typedef struct CgTrainRolloutAbiDescriptor {
  uint32_t magic;
  uint32_t abi_version;
  uint32_t descriptor_size;
  uint32_t catalog_size;
  uint32_t shape_size;
  uint32_t output_size;
  uint32_t token_scalar_size;
  uint32_t option_scalar_size;
  uint32_t dynamic_effect_size;
  uint32_t maximum_entity_slots;
  uint32_t belief_scalar_size;
  uint32_t history_size;
  uint32_t deck_flow_size;
  uint32_t reserved;
  uint64_t features;
  // SHA-256 of the complete token/option/scalar encoding semantics. Width
  // checks alone cannot detect a meaning-preserving ABI size with changed
  // feature indices.
  uint8_t model_encoding_fingerprint[32];
} CgTrainRolloutAbiDescriptor;

// Immutable catalog columns are copied during encoder creation. entry_counts
// is row-major [deck_count][card_vocab_size + 1]. The remaining arrays use
// float64 so native posterior arithmetic has the same precision as the Python
// reference implementation.
typedef struct CgTrainRolloutCatalog {
  uint32_t struct_size;
  uint32_t deck_count;
  uint32_t card_vocab_size;
  uint32_t supporter_count;
  uint32_t posterior_cache_capacity;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;

  const int16_t* entry_counts;
  const double* exact_log_priors;
  const double* log_combinations;
  const double* log_factorials;
  const double* unknown_card_probabilities;
  const double* unknown_log_card_probabilities;
  double unknown_log_prior;
  const int32_t* supporter_card_ids;
} CgTrainRolloutCatalog;

// Exact dimensions required for one direct slot-selection request.
typedef struct CgTrainRolloutShape {
  uint32_t struct_size;
  uint32_t row_count;
  uint32_t state_token_width;
  uint32_t state_attachment_width;
  uint32_t option_width;
  uint32_t deck_width;
  uint32_t belief_row_count;
  uint32_t belief_width;
} CgTrainRolloutShape;

// Every tensor pointer is caller-owned and contiguous. Row-major strides are
// derived from the declared widths. EncodeRows can write a segment into a
// larger shared batch using row_offset and belief_row_offset.
typedef struct CgTrainRolloutOutput {
  uint32_t struct_size;
  uint32_t row_capacity;
  uint32_t state_token_width;
  uint32_t state_attachment_width;
  uint32_t option_width;
  uint32_t deck_width;
  uint32_t belief_row_capacity;
  uint32_t belief_width;

  int64_t* state_card_ids;
  int64_t* state_areas;
  int64_t* state_owner_roles;
  int64_t* state_token_kinds;
  float* state_scalars;
  int64_t* state_last_attack_ids;
  uint8_t* state_padding_mask;
  uint16_t* state_attachment_card_ids;
  uint16_t* state_attachment_parent_indices;
  uint8_t* state_attachment_kinds;
  uint8_t* state_entity_slots;
  uint32_t* state_sequence_lengths;

  int64_t* option_types;
  int64_t* option_contexts;
  int64_t* option_entity_slots;
  uint8_t* option_entity_slot_mask;
  int64_t* option_attack_ids;
  int64_t* option_card_ids;
  float* option_scalars;
  float* option_dynamic_effect_features;
  uint8_t* option_dynamic_effect_masks;
  uint8_t* option_valid;
  int64_t* option_min_counts;
  int64_t* option_max_counts;
  uint32_t* option_lengths;
  uint32_t* option_maximum_counts;

  int64_t* deck_card_ids;
  float* deck_counts;
  uint8_t* deck_valid;

  int64_t* belief_card_ids;
  float* belief_expected_counts;
  uint8_t* belief_valid;
  float* belief_scalars;
  int64_t* belief_row_indices;
} CgTrainRolloutOutput;

CG_TRAIN_API const CgTrainRolloutAbiDescriptor*
CgTrainRolloutGetAbiDescriptor(void);
CG_TRAIN_API const char* CgTrainRolloutLastError(void);

CG_TRAIN_API CgTrainRolloutEncoder* CgTrainRolloutCreate(
    uint32_t slot_capacity, const CgTrainRolloutCatalog* catalog);
CG_TRAIN_API void CgTrainRolloutDestroy(CgTrainRolloutEncoder* encoder);

// Reset decks are contiguous row-major [batch_count][2][60]. Both functions
// stage and fully validate the entire request, including model schema and
// catalog vocabulary constraints, before committing any tracker or snapshot
// state.
CG_TRAIN_API int32_t CgTrainRolloutConsumeReset(
    CgTrainRolloutEncoder* encoder, uint32_t batch_count,
    const uint32_t* slots, const int32_t* decks,
    const CgTrainOutput* source);
CG_TRAIN_API int32_t CgTrainRolloutConsumeStep(
    CgTrainRolloutEncoder* encoder, uint32_t batch_count,
    const uint32_t* slots, const CgTrainOutput* source);
CG_TRAIN_API int32_t CgTrainRolloutClearSlots(
    CgTrainRolloutEncoder* encoder, uint32_t slot_count,
    const uint32_t* slots);

// perspectives must match each selected slot's current acting perspective.
// Duplicate or out-of-range slots are rejected. Calls other than Destroy are
// internally serialized. Destroy must be called exactly once only after every
// operation on the handle has returned; it must never race another call.
CG_TRAIN_API int32_t CgTrainRolloutPlanRows(
    CgTrainRolloutEncoder* encoder, uint32_t row_count,
    const uint32_t* slots, const int32_t* perspectives,
    CgTrainRolloutShape* shape);
CG_TRAIN_API int32_t CgTrainRolloutEncodeRows(
    CgTrainRolloutEncoder* encoder, uint32_t row_count,
    const uint32_t* slots, const int32_t* perspectives,
    uint32_t row_offset, uint32_t belief_row_offset,
    CgTrainRolloutOutput* output);

// Learner-only public evidence. The plan call returns the required flattened
// value count. WriteKnown emits canonical uint32 CSR offsets and sorted int32
// card/count values in caller row order.
CG_TRAIN_API int32_t CgTrainRolloutPlanKnown(
    CgTrainRolloutEncoder* encoder, uint32_t row_count,
    const uint32_t* slots, const int32_t* perspectives,
    uint32_t* value_count);
CG_TRAIN_API int32_t CgTrainRolloutWriteKnown(
    CgTrainRolloutEncoder* encoder, uint32_t row_count,
    const uint32_t* slots, const int32_t* perspectives,
    uint32_t value_capacity, uint32_t* offsets,
    int32_t* card_ids, int32_t* counts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SRC_NATIVE_CG_TRAIN_CG_ROLLOUT_ENCODER_H_

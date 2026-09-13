// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®,
// and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only

#include "rollout_internal.h"

#include <exception>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>

namespace {

const CgTrainRolloutAbiDescriptor kDescriptor = {
    .magic = cg_train_rollout::kRolloutMagic,
    .abi_version = cg_train_rollout::kRolloutAbiVersion,
    .descriptor_size = sizeof(CgTrainRolloutAbiDescriptor),
    .catalog_size = sizeof(CgTrainRolloutCatalog),
    .shape_size = sizeof(CgTrainRolloutShape),
    .output_size = sizeof(CgTrainRolloutOutput),
    .token_scalar_size = cg_train_rollout::kTokenScalarSize,
    .option_scalar_size = cg_train_rollout::kOptionScalarSize,
    .dynamic_effect_size = cg_train_rollout::kDynamicEffectSize,
    .maximum_entity_slots = cg_train_rollout::kMaximumEntitySlots,
    .belief_scalar_size = cg_train_rollout::kBeliefScalarSize,
    .history_size = cg_train_rollout::kHistorySize,
    .deck_flow_size = cg_train_rollout::kDeckFlowSize,
    .reserved = 0,
    .features =
        CG_TRAIN_ROLLOUT_FEATURE_TRANSACTIONAL_HISTORY |
        CG_TRAIN_ROLLOUT_FEATURE_DIRECT_SLOT_SELECTION |
        CG_TRAIN_ROLLOUT_FEATURE_MODEL_READY_TENSORS |
        CG_TRAIN_ROLLOUT_FEATURE_PUBLIC_CATALOG |
        CG_TRAIN_ROLLOUT_FEATURE_CALLER_OWNED_OUTPUT,
    .model_encoding_fingerprint = {
        0xcf, 0x1d, 0x0c, 0xba, 0x60, 0x50, 0x14, 0x2d,
        0x2b, 0xa4, 0x0e, 0x9a, 0x09, 0xef, 0x07, 0x25,
        0x1a, 0xc5, 0x28, 0x5c, 0x1b, 0xbd, 0xf9, 0xce,
        0xf8, 0xcf, 0xd8, 0x63, 0x7f, 0xff, 0x26, 0x4a,
    },
};

template <typename Callback>
std::int32_t Guard(Callback callback) {
  try {
    callback();
    cg_train_rollout::LastError().clear();
    return CG_TRAIN_OK;
  } catch (const std::invalid_argument& error) {
    return cg_train_rollout::Fail(CG_TRAIN_INVALID_ARGUMENT, error.what());
  } catch (const std::length_error& error) {
    return cg_train_rollout::Fail(CG_TRAIN_INSUFFICIENT_CAPACITY,
                                  error.what());
  } catch (const std::exception& error) {
    return cg_train_rollout::Fail(CG_TRAIN_ENGINE_EXCEPTION, error.what());
  } catch (...) {
    return cg_train_rollout::Fail(
        CG_TRAIN_ENGINE_EXCEPTION,
        "unknown native rollout encoder exception");
  }
}

CgTrainRolloutEncoder& RequireEncoder(
    CgTrainRolloutEncoder* encoder) {
  if (encoder == nullptr || encoder->core == nullptr) {
    throw std::invalid_argument("native rollout encoder is null");
  }
  return *encoder;
}

template <typename T>
std::span<const T> InputSpan(const T* values, std::uint32_t size,
                             const char* name) {
  if (size > 0 && values == nullptr) {
    throw std::invalid_argument(std::string("null rollout input: ") + name);
  }
  return std::span<const T>(values, size);
}

}  // namespace

extern "C" {

const CgTrainRolloutAbiDescriptor* CgTrainRolloutGetAbiDescriptor(void) {
  return &kDescriptor;
}

const char* CgTrainRolloutLastError(void) {
  return cg_train_rollout::LastError().c_str();
}

CgTrainRolloutEncoder* CgTrainRolloutCreate(
    std::uint32_t slot_capacity,
    const CgTrainRolloutCatalog* catalog) {
  try {
    if (catalog == nullptr) {
      throw std::invalid_argument("native rollout catalog is null");
    }
    auto core =
        std::make_unique<cg_train_rollout::RolloutEncoderCore>(
            slot_capacity, *catalog);
    auto* encoder = new CgTrainRolloutEncoder(std::move(core));
    cg_train_rollout::LastError().clear();
    return encoder;
  } catch (const std::exception& error) {
    cg_train_rollout::LastError() = error.what();
    return nullptr;
  } catch (...) {
    cg_train_rollout::LastError() =
        "unknown native rollout encoder creation exception";
    return nullptr;
  }
}

void CgTrainRolloutDestroy(CgTrainRolloutEncoder* encoder) {
  delete encoder;
}

std::int32_t CgTrainRolloutConsumeReset(
    CgTrainRolloutEncoder* encoder, std::uint32_t batch_count,
    const std::uint32_t* slots, const std::int32_t* decks,
    const CgTrainOutput* source) {
  return Guard([&] {
    if (source == nullptr) {
      throw std::invalid_argument("native rollout source is null");
    }
    CgTrainRolloutEncoder& handle = RequireEncoder(encoder);
    std::scoped_lock lock(handle.mutex);
    handle.core->ConsumeReset(batch_count, slots, decks, *source);
  });
}

std::int32_t CgTrainRolloutConsumeStep(
    CgTrainRolloutEncoder* encoder, std::uint32_t batch_count,
    const std::uint32_t* slots, const CgTrainOutput* source) {
  return Guard([&] {
    if (source == nullptr) {
      throw std::invalid_argument("native rollout source is null");
    }
    CgTrainRolloutEncoder& handle = RequireEncoder(encoder);
    std::scoped_lock lock(handle.mutex);
    handle.core->ConsumeStep(batch_count, slots, *source);
  });
}

std::int32_t CgTrainRolloutClearSlots(
    CgTrainRolloutEncoder* encoder, std::uint32_t slot_count,
    const std::uint32_t* slots) {
  return Guard([&] {
    CgTrainRolloutEncoder& handle = RequireEncoder(encoder);
    std::scoped_lock lock(handle.mutex);
    handle.core->ClearSlots(InputSpan(slots, slot_count, "slots"));
  });
}

std::int32_t CgTrainRolloutPlanRows(
    CgTrainRolloutEncoder* encoder, std::uint32_t row_count,
    const std::uint32_t* slots, const std::int32_t* perspectives,
    CgTrainRolloutShape* shape) {
  return Guard([&] {
    if (shape == nullptr ||
        shape->struct_size != sizeof(CgTrainRolloutShape)) {
      throw std::invalid_argument(
          "native rollout shape descriptor is invalid");
    }
    CgTrainRolloutEncoder& handle = RequireEncoder(encoder);
    std::scoped_lock lock(handle.mutex);
    const CgTrainRolloutShape planned = handle.core->PlanRows(
        InputSpan(slots, row_count, "slots"),
        InputSpan(perspectives, row_count, "perspectives"));
    *shape = planned;
  });
}

std::int32_t CgTrainRolloutEncodeRows(
    CgTrainRolloutEncoder* encoder, std::uint32_t row_count,
    const std::uint32_t* slots, const std::int32_t* perspectives,
    std::uint32_t row_offset, std::uint32_t belief_row_offset,
    CgTrainRolloutOutput* output) {
  return Guard([&] {
    CgTrainRolloutEncoder& handle = RequireEncoder(encoder);
    std::scoped_lock lock(handle.mutex);
    handle.core->EncodeRows(
        InputSpan(slots, row_count, "slots"),
        InputSpan(perspectives, row_count, "perspectives"), row_offset,
        belief_row_offset, output);
  });
}

std::int32_t CgTrainRolloutPlanKnown(
    CgTrainRolloutEncoder* encoder, std::uint32_t row_count,
    const std::uint32_t* slots, const std::int32_t* perspectives,
    std::uint32_t* value_count) {
  return Guard([&] {
    if (value_count == nullptr) {
      throw std::invalid_argument(
          "native rollout known shape output is null");
    }
    CgTrainRolloutEncoder& handle = RequireEncoder(encoder);
    std::scoped_lock lock(handle.mutex);
    const std::uint32_t planned = handle.core->PlanKnown(
        InputSpan(slots, row_count, "slots"),
        InputSpan(perspectives, row_count, "perspectives"));
    *value_count = planned;
  });
}

std::int32_t CgTrainRolloutWriteKnown(
    CgTrainRolloutEncoder* encoder, std::uint32_t row_count,
    const std::uint32_t* slots, const std::int32_t* perspectives,
    std::uint32_t value_capacity, std::uint32_t* offsets,
    std::int32_t* card_ids, std::int32_t* counts) {
  return Guard([&] {
    CgTrainRolloutEncoder& handle = RequireEncoder(encoder);
    std::scoped_lock lock(handle.mutex);
    handle.core->WriteKnown(
        InputSpan(slots, row_count, "slots"),
        InputSpan(perspectives, row_count, "perspectives"),
        value_capacity, offsets, card_ids, counts);
  });
}

}  // extern "C"

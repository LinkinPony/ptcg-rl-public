// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Pokémon TCG.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "All.h"
#include "planner_session_internal.h"
#include "state_token_validation.h"

#ifdef _MSC_VER
#define CG_PLANNER_SESSION_API __declspec(dllexport)
#else
#define CG_PLANNER_SESSION_API __attribute__((visibility("default")))
#endif

extern "C" void CgProbeInitialize();

namespace {

using planner_session::CellMetadata;
using planner_session::RequestKind;
using planner_session::SessionLane;
using planner_session::TransitionCaps;

bool ValidTransitionCaps(const TransitionCaps& caps) {
  return caps.max_engine_steps > 0 &&
         caps.max_engine_steps <=
             planner_session::kMaximumEngineStepsPerCall &&
         caps.max_forced_steps >= 0 &&
         caps.max_forced_steps <= planner_session::kMaximumForcedSteps &&
         caps.max_observation_bytes > 0 &&
         caps.max_observation_bytes <=
             planner_session::kMaximumObservationBytes;
}

bool FingerprintMatches(
    const std::array<std::uint8_t, planner_session::kFingerprintBytes>& expected,
    const unsigned char* actual) {
  return std::equal(expected.begin(), expected.end(), actual);
}

}  // namespace

extern "C" {

struct CgPlannerSessionResult {
  int error;
  const unsigned char* data;
  int size;
};

CG_PLANNER_SESSION_API int CgPlannerSessionPayloadVersion() {
  return planner_session::kPayloadVersion;
}

CG_PLANNER_SESSION_API const char* CgPlannerSessionAbiDescriptor() {
  return planner_session::kAbiDescriptor;
}

CG_PLANNER_SESSION_API const char* CgPlannerSessionLastError() {
  return planner_session::LastError().c_str();
}

CG_PLANNER_SESSION_API void* CgPlannerCreateSessionLane() {
  try {
    CgProbeInitialize();
    planner_session::SetLastError("");
    return new SessionLane();
  } catch (const std::exception& error) {
    planner_session::SetLastError(error.what());
    return nullptr;
  } catch (...) {
    planner_session::SetLastError(
        "unknown planner session lane creation failure");
    return nullptr;
  }
}

CG_PLANNER_SESSION_API void CgPlannerDestroySessionLane(void* raw_lane) {
  delete static_cast<SessionLane*>(raw_lane);
}

CG_PLANNER_SESSION_API CgPlannerSessionResult CgPlannerOpenSession(
    void* raw_lane,
    const unsigned char* producer_contract_fingerprint,
    int producer_contract_fingerprint_count,
    const char* state_token,
    int state_token_count,
    const int* hidden_counts,
    int hidden_count_count,
    const int* hidden_values,
    int hidden_value_count,
    int world_count,
    const int* candidate_counts,
    int candidate_count_count,
    const int* candidate_values,
    int candidate_value_count,
    int candidate_count,
    int root_player,
    int manual_coin,
    int max_state_slots,
    int max_engine_steps,
    int max_forced_steps,
    int max_observation_bytes) {
  auto* lane = static_cast<SessionLane*>(raw_lane);
  const TransitionCaps caps = {
      max_engine_steps,
      max_forced_steps,
      max_observation_bytes,
  };
  try {
    if (lane == nullptr || producer_contract_fingerprint == nullptr ||
        producer_contract_fingerprint_count !=
            planner_session::kFingerprintBytes ||
        state_token == nullptr || state_token_count <= 0 ||
        state_token_count > planner_session::kMaximumStateTokenBytes ||
        hidden_counts == nullptr ||
        (hidden_values == nullptr && hidden_value_count != 0) ||
        candidate_counts == nullptr ||
        (candidate_values == nullptr && candidate_value_count != 0) ||
        world_count <= 0 ||
        world_count > planner_session::kMaximumCellsPerCall ||
        candidate_count <= 0 ||
        candidate_count > planner_session::kMaximumCellsPerCall ||
        hidden_count_count !=
            world_count * planner_session::kHiddenListCount ||
        candidate_count_count != candidate_count ||
        root_player < 0 || root_player > 1 ||
        (manual_coin != 0 && manual_coin != 1) ||
        max_state_slots <= 0 ||
        max_state_slots > planner_session::kMaximumStateSlots ||
        !ValidTransitionCaps(caps)) {
      planner_session::SetLastError("invalid CgPlannerOpenSession input");
      return {planner_session::kErrorInvalidInput, nullptr, 0};
    }
    const std::int64_t cell_count =
        static_cast<std::int64_t>(world_count) * candidate_count;
    if (cell_count > planner_session::kMaximumCellsPerCall ||
        cell_count > max_state_slots || cell_count > max_engine_steps) {
      planner_session::SetLastError(
          "planner session root grid exceeds a resolved capacity");
      return {planner_session::kErrorInvalidInput, nullptr, 0};
    }
    int expected_hidden_values = 0;
    int expected_candidate_values = 0;
    if (!planner_session::CheckedSum(
            hidden_counts,
            hidden_count_count,
            DECK_SIZE,
            &expected_hidden_values) ||
        expected_hidden_values != hidden_value_count ||
        !planner_session::CheckedSum(
            candidate_counts,
            candidate_count_count,
            planner_session::kMaximumSelectCount,
            &expected_candidate_values) ||
        expected_candidate_values != candidate_value_count) {
      planner_session::SetLastError(
          "planner session ragged root inputs do not match their buffers");
      return {planner_session::kErrorInvalidInput, nullptr, 0};
    }
    const auto request_fingerprint =
        planner_session::OpenRequestFingerprint(
            state_token,
            state_token_count,
            hidden_counts,
            hidden_count_count,
            hidden_values,
            hidden_value_count,
            world_count,
            candidate_counts,
            candidate_count_count,
            candidate_values,
            candidate_value_count,
            candidate_count,
            root_player,
            manual_coin != 0,
            max_state_slots,
            caps);

    std::scoped_lock<std::mutex> lock(lane->mutex);
    if (lane->active_generation != 0) {
      planner_session::SetLastError(
          "planner session lane already has an active generation");
      return {planner_session::kErrorInvalidInput, nullptr, 0};
    }
    std::vector<std::uint8_t> decoded_state;
    std::string state_error;
    if (!DecodeValidatedStateToken(
            state_token,
            state_token_count,
            &decoded_state,
            &state_error)) {
      planner_session::SetLastError(state_error);
      return {planner_session::kErrorInvalidInput, nullptr, 0};
    }
    lane->data->reader.buf = std::move(decoded_state);
    lane->data->reader.pos = 0;
    lane->data->state.deserialize(lane->data->reader);
    if (!ValidateDeserializedPlannerRoot(lane->data->state, root_player)) {
      planner_session::SetLastError(
          "state token does not contain a valid live planner root");
      return {planner_session::kErrorInvalidInput, nullptr, 0};
    }
    const int generation = lane->BeginSession(
        root_player,
        manual_coin != 0,
        max_state_slots,
        producer_contract_fingerprint);
    if (generation <= 0) {
      planner_session::SetLastError(
          "planner session generation capacity is exhausted");
      return {planner_session::kErrorInvalidInput, nullptr, 0};
    }

    try {
      const State base_state = lane->data->state;
      const std::vector<std::vector<int>> candidates =
          planner_session::ReadActions(
              candidate_counts, candidate_values, candidate_count);
      std::vector<CellMetadata> metadata(
          static_cast<std::size_t>(cell_count));
      std::vector<unsigned char> observation_scratch;
      int hidden_offset = 0;
      int engine_steps = 0;
      for (int world_index = 0; world_index < world_count; ++world_index) {
        SearchStartConfig config;
        const bool valid_counts = planner_session::HiddenCountsMatchState(
            hidden_counts, world_index, base_state, root_player);
        const bool valid_values = planner_session::FillSearchConfig(
            hidden_counts,
            hidden_values,
            world_index,
            &hidden_offset,
            manual_coin != 0,
            &config);
        lane->data->search.clear();
        const SearchInfo root =
            valid_counts && valid_values
                ? lane->data->search.start(config, base_state)
                : SearchInfo::error(planner_session::kErrorInvalidInput);
        if (root.errorCode == 0) {
          planner_session::RestoreDeterminizedHiddenZones(root.state);
        }
        const std::mt19937 root_rng = lane->data->game.rng;
        for (int candidate_index = 0; candidate_index < candidate_count;
             ++candidate_index) {
          const std::size_t cell_index =
              static_cast<std::size_t>(candidate_index) * world_count +
              world_index;
          planner_session::ExecuteTransition(
              lane,
              root,
              root_rng,
              candidates[candidate_index],
              caps,
              &engine_steps,
              &observation_scratch,
              &metadata[cell_index]);
        }
        lane->data->search.clear();
      }
      std::vector<unsigned char> payload = planner_session::BuildPayload(
          RequestKind::kOpen,
          generation,
          request_fingerprint,
          producer_contract_fingerprint,
          std::move(metadata),
          observation_scratch);
      auto* output = new unsigned char[payload.size()];
      std::copy(payload.begin(), payload.end(), output);
      planner_session::SetLastError("");
      return {0, output, static_cast<int>(payload.size())};
    } catch (...) {
      lane->ResetActiveSession();
      throw;
    }
  } catch (const std::exception& error) {
    planner_session::SetLastError(error.what());
    return {planner_session::kErrorException, nullptr, 0};
  } catch (...) {
    planner_session::SetLastError("unknown planner session open failure");
    return {planner_session::kErrorException, nullptr, 0};
  }
}

CG_PLANNER_SESSION_API CgPlannerSessionResult CgPlannerContinueSession(
    void* raw_lane,
    const unsigned char* producer_contract_fingerprint,
    int producer_contract_fingerprint_count,
    int generation,
    const int* parent_slots,
    int parent_count,
    const int* action_counts,
    int action_count_count,
    const int* action_values,
    int action_value_count,
    int max_engine_steps,
    int max_forced_steps,
    int max_observation_bytes) {
  auto* lane = static_cast<SessionLane*>(raw_lane);
  const TransitionCaps caps = {
      max_engine_steps,
      max_forced_steps,
      max_observation_bytes,
  };
  try {
    if (lane == nullptr || producer_contract_fingerprint == nullptr ||
        producer_contract_fingerprint_count !=
            planner_session::kFingerprintBytes ||
        generation <= 0 || parent_slots == nullptr || parent_count <= 0 ||
        parent_count > planner_session::kMaximumCellsPerCall ||
        action_counts == nullptr || action_count_count != parent_count ||
        (action_values == nullptr && action_value_count != 0) ||
        parent_count > max_engine_steps || !ValidTransitionCaps(caps)) {
      planner_session::SetLastError(
          "invalid CgPlannerContinueSession input");
      return {planner_session::kErrorInvalidInput, nullptr, 0};
    }
    int expected_action_values = 0;
    if (!planner_session::CheckedSum(
            action_counts,
            action_count_count,
            planner_session::kMaximumSelectCount,
            &expected_action_values) ||
        expected_action_values != action_value_count) {
      planner_session::SetLastError(
          "planner continuation actions do not match their buffers");
      return {planner_session::kErrorInvalidInput, nullptr, 0};
    }
    const auto request_fingerprint =
        planner_session::ContinueRequestFingerprint(
            generation,
            parent_slots,
            parent_count,
            action_counts,
            action_count_count,
            action_values,
            action_value_count,
            caps);

    std::scoped_lock<std::mutex> lock(lane->mutex);
    if (generation != lane->active_generation ||
        !FingerprintMatches(
            lane->producer_fingerprint, producer_contract_fingerprint)) {
      planner_session::SetLastError(
          "planner continuation session identity is stale or mismatched");
      return {planner_session::kErrorStaleHandle, nullptr, 0};
    }
    if (static_cast<std::int64_t>(lane->slots.size()) + parent_count >
        lane->max_state_slots) {
      planner_session::SetLastError(
          "planner continuation would exceed the state-slot arena");
      return {planner_session::kErrorArenaCapacity, nullptr, 0};
    }
    for (int index = 0; index < parent_count; ++index) {
      if (lane->Get(generation, parent_slots[index]) == nullptr) {
        planner_session::SetLastError(
            "planner continuation references a stale state handle");
        return {planner_session::kErrorStaleHandle, nullptr, 0};
      }
    }
    const std::vector<std::vector<int>> actions =
        planner_session::ReadActions(
            action_counts, action_values, parent_count);
    std::vector<CellMetadata> metadata(
        static_cast<std::size_t>(parent_count));
    std::vector<unsigned char> observation_scratch;
    int engine_steps = 0;
    for (int index = 0; index < parent_count; ++index) {
      // Branch from a value copy so every aligned action starts from the exact
      // same immutable parent even when the request repeats a handle.
      const planner_session::SessionSlot parent =
          *lane->Get(generation, parent_slots[index]);
      lane->data->search.clear();
      SearchStartConfig config;
      config.manualCoin = lane->manual_coin;
      const SearchInfo root = lane->data->search.start(config, parent.state);
      planner_session::ExecuteTransition(
          lane,
          root,
          parent.rng,
          actions[index],
          caps,
          &engine_steps,
          &observation_scratch,
          &metadata[index]);
      lane->data->search.clear();
    }
    std::vector<unsigned char> payload = planner_session::BuildPayload(
        RequestKind::kContinue,
        generation,
        request_fingerprint,
        producer_contract_fingerprint,
        std::move(metadata),
        observation_scratch);
    auto* output = new unsigned char[payload.size()];
    std::copy(payload.begin(), payload.end(), output);
    planner_session::SetLastError("");
    return {0, output, static_cast<int>(payload.size())};
  } catch (const std::exception& error) {
    planner_session::SetLastError(error.what());
    return {planner_session::kErrorException, nullptr, 0};
  } catch (...) {
    planner_session::SetLastError("unknown planner continuation failure");
    return {planner_session::kErrorException, nullptr, 0};
  }
}

CG_PLANNER_SESSION_API int CgPlannerReleaseSessionHandles(
    void* raw_lane,
    int generation,
    const int* slots,
    int slot_count) {
  auto* lane = static_cast<SessionLane*>(raw_lane);
  try {
    if (lane == nullptr || generation <= 0 || slots == nullptr ||
        slot_count <= 0 ||
        slot_count > planner_session::kMaximumStateSlots) {
      planner_session::SetLastError(
          "invalid CgPlannerReleaseSessionHandles input");
      return planner_session::kErrorInvalidInput;
    }
    std::scoped_lock<std::mutex> lock(lane->mutex);
    std::vector<int> unique(slots, slots + slot_count);
    std::sort(unique.begin(), unique.end());
    if (std::adjacent_find(unique.begin(), unique.end()) != unique.end()) {
      planner_session::SetLastError(
          "planner session handle release contains duplicates");
      return planner_session::kErrorInvalidInput;
    }
    for (int slot : unique) {
      if (lane->Get(generation, slot) == nullptr) {
        planner_session::SetLastError(
            "planner session handle release is stale");
        return planner_session::kErrorStaleHandle;
      }
    }
    for (int slot : unique) {
      if (!lane->Release(generation, slot)) {
        throw std::runtime_error("validated planner handle could not be released");
      }
    }
    planner_session::SetLastError("");
    return 0;
  } catch (const std::exception& error) {
    planner_session::SetLastError(error.what());
    return planner_session::kErrorException;
  } catch (...) {
    planner_session::SetLastError("unknown planner handle release failure");
    return planner_session::kErrorException;
  }
}

CG_PLANNER_SESSION_API int CgPlannerCloseSession(
    void* raw_lane,
    int generation) {
  auto* lane = static_cast<SessionLane*>(raw_lane);
  try {
    if (lane == nullptr || generation <= 0) {
      planner_session::SetLastError("invalid CgPlannerCloseSession input");
      return planner_session::kErrorInvalidInput;
    }
    std::scoped_lock<std::mutex> lock(lane->mutex);
    if (lane->active_generation != generation) {
      planner_session::SetLastError("planner session close is stale");
      return planner_session::kErrorStaleHandle;
    }
    lane->ResetActiveSession();
    planner_session::SetLastError("");
    return 0;
  } catch (const std::exception& error) {
    planner_session::SetLastError(error.what());
    return planner_session::kErrorException;
  } catch (...) {
    planner_session::SetLastError("unknown planner session close failure");
    return planner_session::kErrorException;
  }
}

CG_PLANNER_SESSION_API void CgPlannerSessionFree(const unsigned char* data) {
  delete[] data;
}

}  // extern "C"

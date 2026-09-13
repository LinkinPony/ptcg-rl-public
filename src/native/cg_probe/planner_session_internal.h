// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Pokémon TCG.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "All.h"

namespace planner_session {

constexpr std::int32_t kMagic = 0x31534745;  // "EGS1", little-endian.
constexpr std::int32_t kPayloadVersion = 5;
constexpr int kFingerprintBytes = 32;
constexpr int kHiddenListCount = 6;
constexpr int kMetadataWidth = 14;
constexpr int kMaximumCellsPerCall = 1 << 16;
constexpr int kMaximumEngineStepsPerCall = 1 << 24;
constexpr int kMaximumForcedSteps = 64;
constexpr int kMaximumSelectCount = 128;
constexpr int kMaximumStateTokenBytes = 1 << 25;
constexpr int kMaximumObservationBytes = 1 << 30;
constexpr int kMaximumStateSlots = 1 << 16;

constexpr int kErrorInvalidInput = 1;
constexpr int kErrorForcedStepCap = 90;
constexpr int kErrorOutputCapacity = 91;
constexpr int kErrorUnsupportedChance = 92;
constexpr int kErrorEngineStepBudget = 93;
constexpr int kErrorArenaCapacity = 94;
constexpr int kErrorStaleHandle = 95;
constexpr int kErrorException = 99;

constexpr char kAbiDescriptor[] =
    "cg-planner-session/v5;row_order=request_aligned;"
    "header=<8i+raw_sha256+producer_sha256;"
    "metadata=<14i:error,rules_exact,endpoint,root_player,leaf_player,"
    "leaf_context,transition_steps,forced_steps,observation_offset,"
    "observation_size,result,leaf_select_type,session_generation,state_slot;"
    "observation=root_visible_select_logs_current_json_v1;"
    "handle=lane_local_generation_slot";

enum class RequestKind : int {
  kOpen = 0,
  kContinue = 1,
};

enum class Endpoint : int {
  kInvalid = 0,
  kTerminal = 1,
  kSameSeatMain = 2,
  kTurnHandoff = 3,
  kRootStrategicPrompt = 4,
  kChancePrompt = 5,
};

struct CellMetadata {
  int error = 0;
  bool rules_exact = false;
  Endpoint endpoint = Endpoint::kInvalid;
  int root_player = -1;
  int leaf_player = -1;
  int leaf_context = -1;
  int transition_steps = 0;
  int forced_steps = 0;
  int observation_offset = 0;
  int observation_size = 0;
  int result = -1;
  int leaf_select_type = -1;
  int session_generation = -1;
  int state_slot = -1;
};

struct SessionSlot {
  State state;
  std::mt19937 rng;

  SessionSlot(const State& source_state, const std::mt19937& source_rng)
      : state(source_state), rng(source_rng) {}
};

struct SessionLane {
  std::unique_ptr<ApiData, void (*)(ApiData*)> data;
  std::mutex mutex;
  int last_generation = 0;
  int active_generation = 0;
  int root_player = -1;
  bool manual_coin = false;
  int max_state_slots = 0;
  int live_states = 0;
  std::array<std::uint8_t, kFingerprintBytes> producer_fingerprint{};
  std::vector<std::unique_ptr<SessionSlot>> slots;

  SessionLane();

  void ResetActiveSession();
  int BeginSession(
      int requested_root_player,
      bool requested_manual_coin,
      int requested_max_state_slots,
      const unsigned char* requested_producer_fingerprint);
  int Allocate(const State& state, const std::mt19937& rng);
  SessionSlot* Get(int generation, int slot);
  bool Release(int generation, int slot);
};

struct TransitionCaps {
  int max_engine_steps;
  int max_forced_steps;
  int max_observation_bytes;
};

void SetLastError(const std::string& message);
const std::string& LastError();

bool CheckedSum(const int* values, int count, int maximum_item, int* total);
bool FillSearchConfig(
    const int* hidden_counts,
    const int* hidden_values,
    int world_index,
    int* value_offset,
    bool manual_coin,
    SearchStartConfig* config);
bool HiddenCountsMatchState(
    const int* hidden_counts,
    int world_index,
    const State& state,
    int root_player);
void RestoreDeterminizedHiddenZones(State* state);
std::vector<std::vector<int>> ReadActions(
    const int* action_counts,
    const int* action_values,
    int action_count);

int ExecuteTransition(
    SessionLane* lane,
    const SearchInfo& root,
    const std::mt19937& root_rng,
    const std::vector<int>& action,
    const TransitionCaps& caps,
    int* engine_steps,
    std::vector<unsigned char>* observations,
    CellMetadata* metadata);

std::array<std::uint8_t, kFingerprintBytes> OpenRequestFingerprint(
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
    bool manual_coin,
    int max_state_slots,
    const TransitionCaps& caps);

std::array<std::uint8_t, kFingerprintBytes> ContinueRequestFingerprint(
    int generation,
    const int* parent_slots,
    int parent_count,
    const int* action_counts,
    int action_count_count,
    const int* action_values,
    int action_value_count,
    const TransitionCaps& caps);

std::vector<unsigned char> BuildPayload(
    RequestKind kind,
    int generation,
    const std::array<std::uint8_t, kFingerprintBytes>& request_fingerprint,
    const unsigned char* producer_fingerprint,
    std::vector<CellMetadata> metadata,
    const std::vector<unsigned char>& observation_scratch,
    const std::vector<int>* row_order = nullptr);

}  // namespace planner_session

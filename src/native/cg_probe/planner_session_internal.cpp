// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Pokémon TCG.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only

#include "planner_session_internal.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

#include "sha256.h"

namespace planner_session {
namespace {

constexpr char kOpenFingerprintDomain[] =
    "ptcg-rl/native-planning-session-open/v1\0";
constexpr char kContinueFingerprintDomain[] =
    "ptcg-rl/native-planning-session-continue/v1\0";

thread_local std::string g_last_error;

void AppendInt(std::vector<unsigned char>* output, std::int32_t value) {
  const std::uint32_t raw = static_cast<std::uint32_t>(value);
  output->push_back(static_cast<unsigned char>(raw & 0xffU));
  output->push_back(static_cast<unsigned char>((raw >> 8U) & 0xffU));
  output->push_back(static_cast<unsigned char>((raw >> 16U) & 0xffU));
  output->push_back(static_cast<unsigned char>((raw >> 24U) & 0xffU));
}

void UpdateByte(Sha256* digest, std::uint8_t value) {
  digest->Update(std::span<const std::uint8_t>(&value, 1));
}

void UpdateLittleEndian32(Sha256* digest, std::uint32_t value) {
  const std::array<std::uint8_t, 4> bytes = {
      static_cast<std::uint8_t>(value & 0xffU),
      static_cast<std::uint8_t>((value >> 8U) & 0xffU),
      static_cast<std::uint8_t>((value >> 16U) & 0xffU),
      static_cast<std::uint8_t>((value >> 24U) & 0xffU),
  };
  digest->Update(bytes);
}

void UpdateIntegers(Sha256* digest, const int* values, int count) {
  UpdateLittleEndian32(digest, static_cast<std::uint32_t>(count));
  for (int index = 0; index < count; ++index) {
    UpdateLittleEndian32(
        digest, static_cast<std::uint32_t>(values[index]));
  }
}

bool CopyHiddenList(
    const int* hidden_counts,
    const int* hidden_values,
    int list_index,
    int* value_offset,
    std::vector<int>* destination) {
  const int count = hidden_counts[list_index];
  destination->resize(count);
  bool valid = true;
  for (int index = 0; index < count; ++index) {
    const int card_id = hidden_values[(*value_offset)++];
    if (!CardTable.contains(card_id)) {
      valid = false;
      continue;
    }
    (*destination)[index] = card_id;
  }
  return valid;
}

int ValidateAction(State* root, const std::vector<int>& action) {
  std::vector<int> sorted = action;
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
    return 6;
  }
  root->selected = action;
  const int error = root->checkPlayerSelect();
  root->selected.clear();
  return error;
}

SearchInfo SafeSearchStep(
    Search* search,
    long long search_id,
    const std::vector<int>& selected) {
  const int previous_last_id = search->lastSearchId();
  const SearchInfo result = search->step(search_id, selected);
  if (result.errorCode != 0 && search->lastSearchId() > previous_last_id) {
    search->clearSingle(search->lastSearchId());
  }
  return result;
}

bool ForcedAction(const State& state, std::vector<int>* selected) {
  const int option_count = static_cast<int>(state.options.size());
  const int min_count = std::min(option_count, std::max(0, state.selectMin));
  const int max_count =
      std::min(option_count, std::max(min_count, state.selectMax));
  if (max_count == 0) {
    selected->clear();
    return true;
  }
  if (option_count == 1 && min_count == 1 && max_count == 1) {
    *selected = {0};
    return true;
  }
  selected->clear();
  return false;
}

Endpoint EndpointFor(const State& state, int root_player) {
  if (state.isFinish()) {
    return Endpoint::kTerminal;
  }
  if (state.selectContext == SelectContext::CoinHead) {
    return Endpoint::kChancePrompt;
  }
  if (state.selectPlayer != root_player) {
    return Endpoint::kTurnHandoff;
  }
  if (state.selectType == SelectType::Main &&
      state.selectContext == SelectContext::Main) {
    return Endpoint::kSameSeatMain;
  }
  return Endpoint::kRootStrategicPrompt;
}

class SendDeckGuard {
 public:
  explicit SendDeckGuard(Game* game)
      : game_(game), original_(game->config.sendDeck) {
    game_->config.sendDeck = false;
  }

  SendDeckGuard(const SendDeckGuard&) = delete;
  SendDeckGuard& operator=(const SendDeckGuard&) = delete;

  ~SendDeckGuard() { game_->config.sendDeck = original_; }

 private:
  Game* game_;
  bool original_;
};

class RngGuard {
 public:
  RngGuard(Game* game, const std::mt19937& branch_rng)
      : game_(game), original_(game->rng), branch_rng_(branch_rng) {
    game_->rng = branch_rng_;
  }

  RngGuard(const RngGuard&) = delete;
  RngGuard& operator=(const RngGuard&) = delete;

  ~RngGuard() { game_->rng = original_; }

  bool consumed() const { return game_->rng != branch_rng_; }
  std::mt19937 current() const { return game_->rng; }

 private:
  Game* game_;
  std::mt19937 original_;
  std::mt19937 branch_rng_;
};

void RootObservationJson(
    const State& state,
    int root_player,
    int start_log_index,
    JsonBuilder* builder) {
  const SendDeckGuard send_deck_guard(state.game);
  builder->clear();
  builder->append('{');
  builder->appendKey("select");
  if (state.isFinish() || state.selectType == SelectType::None ||
      state.selectPlayer != root_player) {
    builder->appendNull();
  } else {
    SelectJson(state, *builder, false);
  }
  builder->appendCommaKey("logs");
  LogsJson(state, *builder, root_player, start_log_index, false);
  builder->appendCommaKey("current");
  Current(state, *builder, root_player, false);
  builder->append('}');
}

void AppendMetadata(
    std::vector<unsigned char>* output,
    const CellMetadata& metadata) {
  AppendInt(output, metadata.error);
  AppendInt(output, metadata.rules_exact ? 1 : 0);
  AppendInt(output, static_cast<int>(metadata.endpoint));
  AppendInt(output, metadata.root_player);
  AppendInt(output, metadata.leaf_player);
  AppendInt(output, metadata.leaf_context);
  AppendInt(output, metadata.transition_steps);
  AppendInt(output, metadata.forced_steps);
  AppendInt(output, metadata.observation_offset);
  AppendInt(output, metadata.observation_size);
  AppendInt(output, metadata.result);
  AppendInt(output, metadata.leaf_select_type);
  AppendInt(output, metadata.session_generation);
  AppendInt(output, metadata.state_slot);
}

}  // namespace

SessionLane::SessionLane()
    : data(ApiAgentStart(), ApiBattleFinish) {}

void SessionLane::ResetActiveSession() {
  data->search.clear();
  slots.clear();
  active_generation = 0;
  root_player = -1;
  manual_coin = false;
  max_state_slots = 0;
  live_states = 0;
  producer_fingerprint.fill(0);
}

int SessionLane::BeginSession(
    int requested_root_player,
    bool requested_manual_coin,
    int requested_max_state_slots,
    const unsigned char* requested_producer_fingerprint) {
  if (active_generation != 0 ||
      last_generation == std::numeric_limits<int>::max()) {
    return -1;
  }
  ++last_generation;
  active_generation = last_generation;
  root_player = requested_root_player;
  manual_coin = requested_manual_coin;
  max_state_slots = requested_max_state_slots;
  live_states = 0;
  slots.clear();
  // Reserve only the lightweight pointer arena.  Reserving a vector of
  // in-place State objects at the ABI maximum would eagerly allocate an
  // impractically large block before a single continuation is retained.
  slots.reserve(static_cast<std::size_t>(requested_max_state_slots));
  std::copy(
      requested_producer_fingerprint,
      requested_producer_fingerprint + kFingerprintBytes,
      producer_fingerprint.begin());
  return active_generation;
}

int SessionLane::Allocate(const State& state, const std::mt19937& rng) {
  // Never reuse a released slot in the same generation.  A stale integer
  // handle therefore cannot silently alias a different engine state.
  if (active_generation == 0 ||
      static_cast<int>(slots.size()) >= max_state_slots) {
    return -1;
  }
  const int slot = static_cast<int>(slots.size());
  slots.push_back(std::make_unique<SessionSlot>(state, rng));
  ++live_states;
  return slot;
}

SessionSlot* SessionLane::Get(int generation, int slot) {
  if (generation != active_generation || slot < 0 ||
      slot >= static_cast<int>(slots.size()) || slots[slot] == nullptr) {
    return nullptr;
  }
  return slots[slot].get();
}

bool SessionLane::Release(int generation, int slot) {
  SessionSlot* current = Get(generation, slot);
  if (current == nullptr) {
    return false;
  }
  static_cast<void>(current);
  slots[slot].reset();
  --live_states;
  return true;
}

void SetLastError(const std::string& message) {
  g_last_error = message;
}

const std::string& LastError() {
  return g_last_error;
}

bool CheckedSum(
    const int* values,
    int count,
    int maximum_item,
    int* total) {
  std::int64_t sum = 0;
  for (int index = 0; index < count; ++index) {
    if (values[index] < 0 || values[index] > maximum_item) {
      return false;
    }
    sum += values[index];
    if (sum > std::numeric_limits<int>::max()) {
      return false;
    }
  }
  *total = static_cast<int>(sum);
  return true;
}

bool FillSearchConfig(
    const int* hidden_counts,
    const int* hidden_values,
    int world_index,
    int* value_offset,
    bool manual_coin,
    SearchStartConfig* config) {
  const int base = world_index * kHiddenListCount;
  *config = {};
  config->manualCoin = manual_coin;
  bool valid = true;
  valid &= CopyHiddenList(
      hidden_counts, hidden_values, base + 0, value_offset,
      &config->myDeck);
  valid &= CopyHiddenList(
      hidden_counts, hidden_values, base + 1, value_offset,
      &config->myPrize);
  valid &= CopyHiddenList(
      hidden_counts, hidden_values, base + 2, value_offset,
      &config->enemyDeck);
  valid &= CopyHiddenList(
      hidden_counts, hidden_values, base + 3, value_offset,
      &config->enemyPrize);
  valid &= CopyHiddenList(
      hidden_counts, hidden_values, base + 4, value_offset,
      &config->enemyHand);
  valid &= CopyHiddenList(
      hidden_counts, hidden_values, base + 5, value_offset,
      &config->enemyActive);
  return valid;
}

bool HiddenCountsMatchState(
    const int* hidden_counts,
    int world_index,
    const State& state,
    int root_player) {
  const int opponent = 1 - root_player;
  const std::array<int, kHiddenListCount> expected = {
      state.selectDeck
          ? 0
          : static_cast<int>(state.players[root_player].deck.size()),
      static_cast<int>(state.players[root_player].prize.size()),
      static_cast<int>(state.players[opponent].deck.size()),
      static_cast<int>(state.players[opponent].prize.size()),
      static_cast<int>(state.players[opponent].hand.size()),
      IsActiveNull(state, opponent)
          ? static_cast<int>(state.players[opponent].active.size())
          : 0,
  };
  const int base = world_index * kHiddenListCount;
  for (int index = 0; index < kHiddenListCount; ++index) {
    if (hidden_counts[base + index] != expected[index]) {
      return false;
    }
  }
  return true;
}

void RestoreDeterminizedHiddenZones(State* state) {
  for (PlayerState& player : state->players) {
    for (CardRef prize : player.prize) {
      if (!prize.isNull()) {
        state->getCard(prize).reverse = true;
      }
    }
  }
}

std::vector<std::vector<int>> ReadActions(
    const int* action_counts,
    const int* action_values,
    int action_count) {
  std::vector<std::vector<int>> actions;
  actions.reserve(action_count);
  int offset = 0;
  for (int action_index = 0; action_index < action_count; ++action_index) {
    std::vector<int> action;
    action.reserve(action_counts[action_index]);
    for (int index = 0; index < action_counts[action_index]; ++index) {
      action.push_back(action_values[offset++]);
    }
    actions.push_back(std::move(action));
  }
  return actions;
}

int ExecuteTransition(
    SessionLane* lane,
    const SearchInfo& root,
    const std::mt19937& root_rng,
    const std::vector<int>& action,
    const TransitionCaps& caps,
    int* engine_steps,
    std::vector<unsigned char>* observations,
    CellMetadata* metadata) {
  metadata->root_player = lane->root_player;
  if (root.errorCode != 0 || root.state == nullptr) {
    metadata->error = root.errorCode == 0 ? kErrorInvalidInput : root.errorCode;
    return metadata->error;
  }
  const int action_error = ValidateAction(root.state, action);
  if (action_error != 0) {
    metadata->error = action_error;
    return action_error;
  }
  if (*engine_steps >= caps.max_engine_steps) {
    metadata->error = kErrorEngineStepBudget;
    return metadata->error;
  }

  RngGuard rng_guard(&lane->data->game, root_rng);
  const int root_log_index = static_cast<int>(root.state->logs.size());
  ++(*engine_steps);
  SearchInfo current = SafeSearchStep(
      &lane->data->search, root.searchId, action);
  if (current.errorCode != 0 || current.state == nullptr) {
    metadata->error = current.errorCode == 0
                          ? kErrorInvalidInput
                          : current.errorCode;
    return metadata->error;
  }
  metadata->transition_steps = 1;
  while (!current.state->isFinish()) {
    if (EndpointFor(*current.state, lane->root_player) !=
        Endpoint::kRootStrategicPrompt) {
      break;
    }
    std::vector<int> forced;
    if (!ForcedAction(*current.state, &forced)) {
      break;
    }
    if (metadata->forced_steps >= caps.max_forced_steps) {
      metadata->error = kErrorForcedStepCap;
      break;
    }
    if (*engine_steps >= caps.max_engine_steps) {
      metadata->error = kErrorEngineStepBudget;
      break;
    }
    ++(*engine_steps);
    const SearchInfo next = SafeSearchStep(
        &lane->data->search, current.searchId, forced);
    if (next.errorCode != 0 || next.state == nullptr) {
      metadata->error = next.errorCode == 0
                            ? kErrorInvalidInput
                            : next.errorCode;
      break;
    }
    lane->data->search.clearSingle(current.searchId);
    current = next;
    ++metadata->forced_steps;
    ++metadata->transition_steps;
  }

  if (metadata->error == 0 && rng_guard.consumed()) {
    metadata->error = kErrorUnsupportedChance;
  }
  if (metadata->error == 0) {
    metadata->rules_exact = true;
    metadata->endpoint = EndpointFor(*current.state, lane->root_player);
    if (current.state->isFinish()) {
      metadata->leaf_player = -1;
      metadata->leaf_context = -1;
      metadata->leaf_select_type = -1;
    } else {
      metadata->leaf_player = current.state->selectPlayer;
      metadata->leaf_context =
          static_cast<int>(current.state->selectContext) - 1;
      metadata->leaf_select_type =
          static_cast<int>(current.state->selectType) - 1;
    }
    metadata->result = current.state->apiResult();
    if (metadata->endpoint == Endpoint::kRootStrategicPrompt ||
        metadata->endpoint == Endpoint::kChancePrompt) {
      const int slot = lane->Allocate(*current.state, rng_guard.current());
      if (slot < 0) {
        metadata->error = kErrorArenaCapacity;
        metadata->rules_exact = false;
        metadata->endpoint = Endpoint::kInvalid;
      } else {
        metadata->session_generation = lane->active_generation;
        metadata->state_slot = slot;
      }
    }
  }
  if (metadata->error == 0) {
    RootObservationJson(
        *current.state,
        lane->root_player,
        root_log_index,
        &lane->data->jsonBuilder);
    const auto& json = lane->data->jsonBuilder.buf;
    if (json.size() >
        static_cast<std::size_t>(caps.max_observation_bytes) -
            observations->size()) {
      metadata->error = kErrorOutputCapacity;
      metadata->rules_exact = false;
      metadata->endpoint = Endpoint::kInvalid;
      if (metadata->state_slot >= 0) {
        lane->Release(
            metadata->session_generation, metadata->state_slot);
        metadata->session_generation = -1;
        metadata->state_slot = -1;
      }
    } else {
      metadata->observation_offset =
          static_cast<int>(observations->size());
      metadata->observation_size = static_cast<int>(json.size());
      const auto* begin =
          reinterpret_cast<const unsigned char*>(json.data());
      observations->insert(
          observations->end(), begin, begin + json.size());
    }
  }
  lane->data->search.clearSingle(current.searchId);
  return metadata->error;
}

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
    const TransitionCaps& caps) {
  Sha256 digest;
  digest.Update(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(kOpenFingerprintDomain),
      sizeof(kOpenFingerprintDomain) - 1));
  UpdateLittleEndian32(&digest, static_cast<std::uint32_t>(state_token_count));
  digest.Update(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(state_token),
      static_cast<std::size_t>(state_token_count)));
  for (int value : {
           world_count,
           candidate_count,
           root_player,
           max_state_slots,
           caps.max_engine_steps,
           caps.max_forced_steps,
           caps.max_observation_bytes,
       }) {
    UpdateLittleEndian32(&digest, static_cast<std::uint32_t>(value));
  }
  UpdateByte(&digest, manual_coin ? 1U : 0U);
  UpdateIntegers(&digest, hidden_counts, hidden_count_count);
  UpdateIntegers(&digest, hidden_values, hidden_value_count);
  UpdateIntegers(&digest, candidate_counts, candidate_count_count);
  UpdateIntegers(&digest, candidate_values, candidate_value_count);
  return digest.Finalize();
}

std::array<std::uint8_t, kFingerprintBytes> ContinueRequestFingerprint(
    int generation,
    const int* parent_slots,
    int parent_count,
    const int* action_counts,
    int action_count_count,
    const int* action_values,
    int action_value_count,
    const TransitionCaps& caps) {
  Sha256 digest;
  digest.Update(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(kContinueFingerprintDomain),
      sizeof(kContinueFingerprintDomain) - 1));
  for (int value : {
           generation,
           parent_count,
           caps.max_engine_steps,
           caps.max_forced_steps,
           caps.max_observation_bytes,
       }) {
    UpdateLittleEndian32(&digest, static_cast<std::uint32_t>(value));
  }
  UpdateIntegers(&digest, parent_slots, parent_count);
  UpdateIntegers(&digest, action_counts, action_count_count);
  UpdateIntegers(&digest, action_values, action_value_count);
  return digest.Finalize();
}

std::vector<unsigned char> BuildPayload(
    RequestKind kind,
    int generation,
    const std::array<std::uint8_t, kFingerprintBytes>& request_fingerprint,
    const unsigned char* producer_fingerprint,
    std::vector<CellMetadata> metadata,
    const std::vector<unsigned char>& observation_scratch,
    const std::vector<int>* row_order) {
  std::vector<CellMetadata> ordered;
  if (row_order == nullptr) {
    ordered = std::move(metadata);
  } else {
    if (row_order->size() != metadata.size()) {
      throw std::invalid_argument("row order does not cover session metadata");
    }
    ordered.reserve(metadata.size());
    for (int index : *row_order) {
      if (index < 0 || index >= static_cast<int>(metadata.size())) {
        throw std::invalid_argument("row order references absent metadata");
      }
      ordered.push_back(metadata[index]);
    }
  }

  std::vector<unsigned char> observations;
  observations.reserve(observation_scratch.size());
  for (CellMetadata& cell : ordered) {
    const int source_offset = cell.observation_offset;
    cell.observation_offset = static_cast<int>(observations.size());
    if (cell.observation_size <= 0) {
      continue;
    }
    if (source_offset < 0 ||
        source_offset + cell.observation_size >
            static_cast<int>(observation_scratch.size())) {
      throw std::invalid_argument("session observation slice is invalid");
    }
    const auto begin = observation_scratch.begin() + source_offset;
    observations.insert(
        observations.end(), begin, begin + cell.observation_size);
  }

  std::vector<unsigned char> payload;
  payload.reserve(
      8 * sizeof(std::int32_t) + 2 * kFingerprintBytes +
      ordered.size() * kMetadataWidth * sizeof(std::int32_t) +
      observations.size());
  AppendInt(&payload, kMagic);
  AppendInt(&payload, kPayloadVersion);
  AppendInt(&payload, static_cast<int>(kind));
  AppendInt(&payload, generation);
  AppendInt(&payload, static_cast<int>(ordered.size()));
  AppendInt(&payload, kMetadataWidth);
  AppendInt(&payload, static_cast<int>(observations.size()));
  AppendInt(&payload, 0);
  payload.insert(
      payload.end(), request_fingerprint.begin(), request_fingerprint.end());
  payload.insert(
      payload.end(), producer_fingerprint,
      producer_fingerprint + kFingerprintBytes);
  for (const CellMetadata& cell : ordered) {
    AppendMetadata(&payload, cell);
  }
  payload.insert(payload.end(), observations.begin(), observations.end());
  return payload;
}

}  // namespace planner_session

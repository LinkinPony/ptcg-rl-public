// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in data/ptcg_engine/ptcgProgram 22/LICENSES/.

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "All.h"
#include "request_fingerprint.h"
#include "state_token_validation.h"

#ifdef _MSC_VER
#define CG_PLANNER_API __declspec(dllexport)
#else
#define CG_PLANNER_API __attribute__((visibility("default")))
#endif

extern "C" void CgProbeInitialize();

namespace {

constexpr std::int32_t kPlannerMagic = 0x31434745;  // "EGC1", little-endian.
constexpr std::int32_t kPlannerVersion = 6;
constexpr int kRequestFingerprintBytes = 32;
constexpr int kHiddenListCount = 6;
constexpr int kMetadataWidth = 14;
constexpr int kMaximumCellsPerCall = 1 << 16;
constexpr int kMaximumEngineStepsPerCall = 1 << 24;
constexpr int kMaximumForcedSteps = 64;
constexpr int kMaximumSelectCount = 128;
constexpr int kMaximumStateTokenBytes = 1 << 25;
constexpr int kMaximumObservationBytes = 1 << 30;
constexpr int kErrorInvalidInput = 1;
constexpr int kErrorForcedStepCap = 90;
constexpr int kErrorOutputCapacity = 91;
constexpr int kErrorUnsupportedChance = 92;
constexpr int kErrorEngineStepBudget = 93;
constexpr int kErrorException = 99;
constexpr char kPlannerAbiDescriptor[] =
    "cg-planner/v6;cell_order=candidate_major;"
    "header=<6i+raw_sha256+producer_sha256;"
    "metadata=<14i:error,rules_exact,endpoint,root_player,leaf_player,"
    "leaf_context,transition_steps,forced_steps,observation_offset,"
    "observation_size,result,leaf_select_type,leaf_observation_offset,"
    "leaf_observation_size;"
    "rng=request_seeded_world_stream_common_across_candidates_v1;"
    "observation=root_visible_then_leaf_actor_visible_select_logs_current_json_v1";

enum class PlannerEndpoint : int {
  kInvalid = 0,
  kTerminal = 1,
  kSameSeatMain = 2,
  kTurnHandoff = 3,
  kRootStrategicPrompt = 4,
  kChancePrompt = 5,
};

thread_local std::string g_planner_last_error;

void SetPlannerError(const std::string& message) {
  g_planner_last_error = message;
}

void AppendInt(std::vector<unsigned char>& output, std::int32_t value) {
  const std::uint32_t raw = static_cast<std::uint32_t>(value);
  output.push_back(static_cast<unsigned char>(raw & 0xffU));
  output.push_back(static_cast<unsigned char>((raw >> 8U) & 0xffU));
  output.push_back(static_cast<unsigned char>((raw >> 16U) & 0xffU));
  output.push_back(static_cast<unsigned char>((raw >> 24U) & 0xffU));
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

void RestoreDeterminizedHiddenZones(State* state) {
  // Search::start materializes erased prize identities but leaves Card::reverse
  // at its zero-initialized value.  Prize cards are face-down in the source
  // state: keeping them face-up both leaks sampled identities through
  // Current/PlayerJson and changes engine rules such as Lucky Bonus.  Restore
  // the hidden-zone invariant before executing any candidate.
  for (PlayerState& player : state->players) {
    for (CardRef prize : player.prize) {
      if (!prize.isNull()) {
        state->getCard(prize).reverse = true;
      }
    }
  }
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

std::vector<std::vector<int>> ReadCandidates(
    const int* candidate_counts,
    const int* candidate_values,
    int candidate_count) {
  std::vector<std::vector<int>> candidates;
  candidates.reserve(candidate_count);
  int offset = 0;
  for (int candidate_index = 0; candidate_index < candidate_count;
       ++candidate_index) {
    std::vector<int> candidate;
    candidate.reserve(candidate_counts[candidate_index]);
    for (int index = 0; index < candidate_counts[candidate_index]; ++index) {
      candidate.push_back(candidate_values[offset++]);
    }
    candidates.push_back(std::move(candidate));
  }
  return candidates;
}

int ValidateCandidate(State* root, const std::vector<int>& candidate) {
  std::vector<int> sorted = candidate;
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
    return 6;
  }
  root->selected = candidate;
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

PlannerEndpoint EndpointFor(const State& state, int root_player) {
  if (state.isFinish()) {
    return PlannerEndpoint::kTerminal;
  }
  if (state.selectContext == SelectContext::CoinHead) {
    return PlannerEndpoint::kChancePrompt;
  }
  if (state.selectPlayer != root_player) {
    return PlannerEndpoint::kTurnHandoff;
  }
  if (state.selectType == SelectType::Main &&
      state.selectContext == SelectContext::Main) {
    return PlannerEndpoint::kSameSeatMain;
  }
  return PlannerEndpoint::kRootStrategicPrompt;
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

class RngBranchGuard {
 public:
  RngBranchGuard(Game* game, const std::mt19937* root_rng)
      : game_(game), root_rng_(root_rng) {
    game_->rng = *root_rng_;
  }

  RngBranchGuard(const RngBranchGuard&) = delete;
  RngBranchGuard& operator=(const RngBranchGuard&) = delete;

  ~RngBranchGuard() { game_->rng = *root_rng_; }

  bool consumed() const { return game_->rng != *root_rng_; }

 private:
  Game* game_;
  const std::mt19937* root_rng_;
};

std::mt19937 WorldRng(std::uint64_t stochastic_seed, int world_index) {
  std::seed_seq sequence = {
      static_cast<std::uint32_t>(stochastic_seed),
      static_cast<std::uint32_t>(stochastic_seed >> 32U),
      static_cast<std::uint32_t>(world_index),
      0x43524731U,
  };
  return std::mt19937(sequence);
}

void ActorObservationJson(
    const State& state,
    int observer_player,
    int start_log_index,
    JsonBuilder* builder) {
  // Search states retain the engine's visualization configuration.  The
  // planner contract must never inherit ``sendDeck`` because ordered deck
  // identities are not root-observable, even when a diagnostic caller enabled
  // them on the source game.
  const SendDeckGuard send_deck_guard(state.game);
  builder->clear();
  builder->append('{');
  builder->appendKey("select");
  if (state.isFinish() || state.selectType == SelectType::None ||
      state.selectPlayer != observer_player) {
    builder->appendNull();
  } else {
    SelectJson(state, *builder, false);
  }
  builder->appendCommaKey("logs");
  LogsJson(state, *builder, observer_player, start_log_index, false);
  builder->appendCommaKey("current");
  Current(state, *builder, observer_player, false);
  builder->append('}');
}

struct PlannerLane {
  std::unique_ptr<ApiData, void (*)(ApiData*)> data;
  std::mutex mutex;

  PlannerLane() : data(ApiAgentStart(), ApiBattleFinish) {}
};

struct CellMetadata {
  int error = 0;
  bool rules_exact = false;
  PlannerEndpoint endpoint = PlannerEndpoint::kInvalid;
  int root_player = -1;
  int leaf_player = -1;
  int leaf_context = -1;
  int transition_steps = 0;
  int forced_steps = 0;
  int observation_offset = 0;
  int observation_size = 0;
  int result = -1;
  int leaf_select_type = -1;
  int leaf_observation_offset = 0;
  int leaf_observation_size = 0;
};

void AppendMetadata(
    std::vector<unsigned char>* output,
    const CellMetadata& metadata) {
  AppendInt(*output, metadata.error);
  AppendInt(*output, metadata.rules_exact ? 1 : 0);
  AppendInt(*output, static_cast<int>(metadata.endpoint));
  AppendInt(*output, metadata.root_player);
  AppendInt(*output, metadata.leaf_player);
  AppendInt(*output, metadata.leaf_context);
  AppendInt(*output, metadata.transition_steps);
  AppendInt(*output, metadata.forced_steps);
  AppendInt(*output, metadata.observation_offset);
  AppendInt(*output, metadata.observation_size);
  AppendInt(*output, metadata.result);
  AppendInt(*output, metadata.leaf_select_type);
  AppendInt(*output, metadata.leaf_observation_offset);
  AppendInt(*output, metadata.leaf_observation_size);
}

}  // namespace

extern "C" {

struct CgPlannerResult {
  int error;
  const unsigned char* data;
  int size;
};

CG_PLANNER_API int CgPlannerPayloadVersion() {
  return kPlannerVersion;
}

CG_PLANNER_API const char* CgPlannerAbiDescriptor() {
  return kPlannerAbiDescriptor;
}

CG_PLANNER_API const char* CgPlannerLastError() {
  return g_planner_last_error.c_str();
}

CG_PLANNER_API void* CgPlannerCreateLane() {
  try {
    CgProbeInitialize();
    SetPlannerError("");
    return new PlannerLane();
  } catch (const std::exception& error) {
    SetPlannerError(error.what());
    return nullptr;
  } catch (...) {
    SetPlannerError("unknown planner lane creation failure");
    return nullptr;
  }
}

CG_PLANNER_API void CgPlannerDestroyLane(void* raw_lane) {
  delete static_cast<PlannerLane*>(raw_lane);
}

CG_PLANNER_API CgPlannerResult CgPlannerDecisionBatch(
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
    std::uint64_t stochastic_seed,
    int max_cells,
    int max_engine_steps,
    int max_forced_steps,
    int max_observation_bytes) {
  try {
    auto* lane = static_cast<PlannerLane*>(raw_lane);
    if (lane == nullptr || producer_contract_fingerprint == nullptr ||
        producer_contract_fingerprint_count != kRequestFingerprintBytes ||
        state_token == nullptr || state_token_count <= 0 ||
        state_token_count > kMaximumStateTokenBytes ||
        hidden_counts == nullptr ||
        (hidden_values == nullptr && hidden_value_count != 0) ||
        candidate_counts == nullptr ||
        (candidate_values == nullptr && candidate_value_count != 0) ||
        world_count <= 0 || world_count > kMaximumCellsPerCall ||
        candidate_count <= 0 || candidate_count > kMaximumCellsPerCall ||
        root_player < 0 || root_player > 1 ||
        hidden_count_count != world_count * kHiddenListCount ||
        candidate_count_count != candidate_count ||
        (manual_coin != 0 && manual_coin != 1) ||
        max_cells <= 0 || max_cells > kMaximumCellsPerCall ||
        max_engine_steps <= 0 ||
        max_engine_steps > kMaximumEngineStepsPerCall ||
        max_forced_steps < 0 || max_forced_steps > kMaximumForcedSteps ||
        max_observation_bytes <= 0 ||
        max_observation_bytes > kMaximumObservationBytes) {
      SetPlannerError("invalid CgPlannerDecisionBatch input");
      return {kErrorInvalidInput, nullptr, 0};
    }
    const std::int64_t transition_count =
        static_cast<std::int64_t>(world_count) * candidate_count;
    if (transition_count > max_cells) {
      SetPlannerError("candidate-by-world grid exceeds max_cells");
      return {kErrorInvalidInput, nullptr, 0};
    }
    if (transition_count > max_engine_steps) {
      SetPlannerError("candidate-by-world grid exceeds max_engine_steps");
      return {kErrorInvalidInput, nullptr, 0};
    }
    int expected_hidden_values = 0;
    int expected_candidate_values = 0;
    if (!CheckedSum(
            hidden_counts, hidden_count_count, DECK_SIZE,
            &expected_hidden_values) ||
        expected_hidden_values != hidden_value_count ||
        !CheckedSum(
            candidate_counts, candidate_count_count, kMaximumSelectCount,
            &expected_candidate_values) ||
        expected_candidate_values != candidate_value_count) {
      SetPlannerError("ragged input counts do not match supplied buffers");
      return {kErrorInvalidInput, nullptr, 0};
    }

    const std::array<std::uint8_t, kRequestFingerprintBytes>
        raw_request_fingerprint = ConsequenceRequestFingerprint(
            state_token, state_token_count, hidden_counts,
            hidden_count_count, hidden_values, hidden_value_count,
            world_count, candidate_counts, candidate_count_count,
            candidate_values, candidate_value_count, candidate_count,
            root_player, manual_coin != 0, stochastic_seed, max_cells,
            max_engine_steps, max_forced_steps, max_observation_bytes);

    std::scoped_lock<std::mutex> lock(lane->mutex);
    std::vector<std::uint8_t> decoded_state;
    std::string state_error;
    if (!DecodeValidatedStateToken(
            state_token, state_token_count, &decoded_state, &state_error)) {
      SetPlannerError(state_error);
      return {kErrorInvalidInput, nullptr, 0};
    }
    lane->data->reader.buf = std::move(decoded_state);
    lane->data->reader.pos = 0;
    lane->data->state.deserialize(lane->data->reader);
    if (!ValidateDeserializedPlannerRoot(lane->data->state, root_player)) {
      SetPlannerError("state token does not contain a valid live root prompt");
      return {kErrorInvalidInput, nullptr, 0};
    }
    const State base_state = lane->data->state;
    const std::vector<std::vector<int>> candidates = ReadCandidates(
        candidate_counts, candidate_values, candidate_count);
    std::vector<CellMetadata> metadata(
        static_cast<std::size_t>(transition_count));
    std::vector<unsigned char> observation_scratch;
    int hidden_offset = 0;
    int engine_steps = 0;

    for (int world_index = 0; world_index < world_count; ++world_index) {
      lane->data->game.rng = WorldRng(stochastic_seed, world_index);
      SearchStartConfig config;
      const bool valid_counts = HiddenCountsMatchState(
          hidden_counts, world_index, base_state, root_player);
      const bool valid_values = FillSearchConfig(
          hidden_counts, hidden_values, world_index, &hidden_offset,
          manual_coin != 0, &config);
      const bool valid_hidden = valid_counts && valid_values;
      lane->data->search.clear();
      const SearchInfo root = valid_hidden
                                  ? lane->data->search.start(config, base_state)
                                  : SearchInfo::error(kErrorInvalidInput);
      if (root.errorCode == 0) {
        RestoreDeterminizedHiddenZones(root.state);
      }
      const std::mt19937 root_rng = lane->data->game.rng;
      for (int candidate_index = 0; candidate_index < candidate_count;
           ++candidate_index) {
        const std::vector<int>& candidate = candidates[candidate_index];
        const std::size_t cell_index =
            static_cast<std::size_t>(candidate_index) * world_count +
            world_index;
        CellMetadata cell;
        cell.root_player = root_player;
        if (root.errorCode != 0) {
          cell.error = root.errorCode;
          metadata[cell_index] = cell;
          continue;
        }
        const int candidate_error = ValidateCandidate(root.state, candidate);
        if (candidate_error != 0) {
          cell.error = candidate_error;
          metadata[cell_index] = cell;
          continue;
        }
        const RngBranchGuard rng_guard(&lane->data->game, &root_rng);
        const int root_log_index = static_cast<int>(root.state->logs.size());
        if (engine_steps >= max_engine_steps) {
          lane->data->search.clear();
          SetPlannerError("planner request exhausted max_engine_steps");
          return {kErrorEngineStepBudget, nullptr, 0};
        }
        ++engine_steps;
        SearchInfo current = SafeSearchStep(
            &lane->data->search, root.searchId, candidate);
        bool owns_current = current.errorCode == 0;
        if (!owns_current) {
          cell.error = current.errorCode;
          metadata[cell_index] = cell;
          continue;
        }
        cell.transition_steps = 1;
        while (!current.state->isFinish()) {
          // Semantic boundaries take precedence over syntactic prompt
          // forcedness.  In particular, a zero-choice MAIN prompt or the
          // next player's singleton prompt already belongs to the next
          // decision/turn and must not be consumed by this root transition.
          // Only forced strategic prompts for the root player are completed.
          if (EndpointFor(*current.state, root_player) !=
              PlannerEndpoint::kRootStrategicPrompt) {
            break;
          }
          std::vector<int> forced;
          if (!ForcedAction(*current.state, &forced)) {
            break;
          }
          if (cell.forced_steps >= max_forced_steps) {
            cell.error = kErrorForcedStepCap;
            break;
          }
          if (engine_steps >= max_engine_steps) {
            lane->data->search.clear();
            SetPlannerError("planner request exhausted max_engine_steps");
            return {kErrorEngineStepBudget, nullptr, 0};
          }
          ++engine_steps;
          const SearchInfo next = SafeSearchStep(
              &lane->data->search, current.searchId, forced);
          if (next.errorCode != 0) {
            cell.error = next.errorCode;
            break;
          }
          lane->data->search.clearSingle(current.searchId);
          current = next;
          ++cell.forced_steps;
          ++cell.transition_steps;
        }
        // Manual-coin planner calls require an explicit chance endpoint and
        // therefore reject any opaque RNG consumption.  Reanalysis calls use
        // engine-sampled randomness (manual_coin=false): the branch guard
        // still restores the common root stream after every candidate, while
        // the realized legal stochastic outcome remains valid evidence.
        if (cell.error == 0 && manual_coin != 0 && rng_guard.consumed()) {
          cell.error = kErrorUnsupportedChance;
        }
        if (cell.error == 0) {
          cell.rules_exact = true;
          cell.endpoint = EndpointFor(*current.state, root_player);
          if (current.state->isFinish()) {
            cell.leaf_player = -1;
            cell.leaf_context = -1;
            cell.leaf_select_type = -1;
          } else {
            cell.leaf_player = current.state->selectPlayer;
            cell.leaf_context =
                static_cast<int>(current.state->selectContext) - 1;
            cell.leaf_select_type =
                static_cast<int>(current.state->selectType) - 1;
          }
          cell.result = current.state->apiResult();
          ActorObservationJson(
              *current.state, root_player, root_log_index,
              &lane->data->jsonBuilder);
          const auto& json = lane->data->jsonBuilder.buf;
          if (json.size() >
              static_cast<std::size_t>(max_observation_bytes) -
                  observation_scratch.size()) {
            lane->data->search.clearSingle(current.searchId);
            lane->data->search.clear();
            SetPlannerError("planner observation output exceeds capacity");
            return {kErrorOutputCapacity, nullptr, 0};
          }
          cell.observation_offset =
              static_cast<int>(observation_scratch.size());
          cell.observation_size = static_cast<int>(json.size());
          const auto* begin = reinterpret_cast<const unsigned char*>(json.data());
          observation_scratch.insert(
              observation_scratch.end(), begin, begin + json.size());
          if (!current.state->isFinish()) {
            ActorObservationJson(
                *current.state, cell.leaf_player, root_log_index,
                &lane->data->jsonBuilder);
            const auto& leaf_json = lane->data->jsonBuilder.buf;
            if (leaf_json.size() >
                static_cast<std::size_t>(max_observation_bytes) -
                    observation_scratch.size()) {
              lane->data->search.clearSingle(current.searchId);
              lane->data->search.clear();
              SetPlannerError("planner observation output exceeds capacity");
              return {kErrorOutputCapacity, nullptr, 0};
            }
            cell.leaf_observation_offset =
                static_cast<int>(observation_scratch.size());
            cell.leaf_observation_size = static_cast<int>(leaf_json.size());
            const auto* leaf_begin =
                reinterpret_cast<const unsigned char*>(leaf_json.data());
            observation_scratch.insert(
                observation_scratch.end(), leaf_begin,
                leaf_begin + leaf_json.size());
          }
        }
        lane->data->search.clearSingle(current.searchId);
        metadata[cell_index] = cell;
      }
      lane->data->search.clear();
    }

    // Execution remains world-major so all candidates reuse one root Search
    // prefix.  Serialize cells candidate-major, the stable Python/tensor ABI,
    // by rebasing each observation slice into the final contiguous blob.
    int candidate_major_observation_size = 0;
    for (const CellMetadata& cell : metadata) {
      candidate_major_observation_size +=
          cell.observation_size + cell.leaf_observation_size;
    }
    std::vector<unsigned char> observations;
    observations.reserve(
        static_cast<std::size_t>(candidate_major_observation_size));
    for (CellMetadata& cell : metadata) {
      const int source_offset = cell.observation_offset;
      cell.observation_offset = static_cast<int>(observations.size());
      if (cell.observation_size > 0) {
        const auto source_begin = observation_scratch.begin() + source_offset;
        observations.insert(
            observations.end(), source_begin,
            source_begin + cell.observation_size);
      }
    }
    // Keep all root-visible slices as one contiguous prefix so existing
    // root-observation consumers can retain a zero-copy view.  Leaf-actor
    // slices form the contiguous suffix in the same candidate-major order.
    for (CellMetadata& cell : metadata) {
      const int source_offset = cell.leaf_observation_offset;
      cell.leaf_observation_offset = static_cast<int>(observations.size());
      if (cell.leaf_observation_size > 0) {
        const auto source_begin = observation_scratch.begin() + source_offset;
        observations.insert(
            observations.end(), source_begin,
            source_begin + cell.leaf_observation_size);
      }
    }

    std::vector<unsigned char> payload;
    payload.reserve(
        6 * sizeof(std::int32_t) + 2 * kRequestFingerprintBytes +
        metadata.size() * kMetadataWidth * sizeof(std::int32_t) +
        observations.size());
    AppendInt(payload, kPlannerMagic);
    AppendInt(payload, kPlannerVersion);
    AppendInt(payload, world_count);
    AppendInt(payload, candidate_count);
    AppendInt(payload, kMetadataWidth);
    AppendInt(payload, static_cast<int>(observations.size()));
    payload.insert(
        payload.end(), raw_request_fingerprint.begin(),
        raw_request_fingerprint.end());
    payload.insert(
        payload.end(), producer_contract_fingerprint,
        producer_contract_fingerprint + producer_contract_fingerprint_count);
    for (const CellMetadata& cell : metadata) {
      AppendMetadata(&payload, cell);
    }
    payload.insert(payload.end(), observations.begin(), observations.end());

    auto* output = new unsigned char[payload.size()];
    std::copy(payload.begin(), payload.end(), output);
    SetPlannerError("");
    return {0, output, static_cast<int>(payload.size())};
  } catch (const std::exception& error) {
    SetPlannerError(error.what());
    return {kErrorException, nullptr, 0};
  } catch (...) {
    SetPlannerError("unknown planner decision batch failure");
    return {kErrorException, nullptr, 0};
  }
}

CG_PLANNER_API void CgPlannerFree(const unsigned char* data) {
  delete[] data;
}

}  // extern "C"

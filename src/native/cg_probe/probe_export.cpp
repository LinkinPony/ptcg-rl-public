// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in data/ptcg_engine/ptcgProgram 22/LICENSES/.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

#include "All.h"
#include "probe_features.h"

#ifdef _MSC_VER
#define CG_PROBE_API __declspec(dllexport)
#else
#define CG_PROBE_API __attribute__((visibility("default")))
#endif

namespace {

constexpr std::int32_t kMagic = 0x31504743;  // "CGP1", little-endian.
constexpr std::int32_t kVersion = 2;
constexpr int kHiddenListCount = 6;
// Keep this aligned with engine/probe_resolution.py.
constexpr int kMaxForcedProbeSteps = 16;
constexpr int kMaxMacroSteps = 32;
constexpr int kMaxDeterministicCompletionSteps = 32;
constexpr int kErrorInvalidInput = 1;
// The explicit macro reached its semantic boundary before all supplied steps
// were consumed.  Keep this outside the engine's Search error-code range so a
// caller can distinguish malformed macro attribution from an illegal action.
constexpr int kErrorMacroTrailingAction = 1001;
constexpr int kErrorException = 99;
constexpr int kFactVersion = 1;

using cg_probe_internal::DynamicEffectFeatures;
using cg_probe_internal::WireLog;

thread_local std::string g_last_error;
std::once_flag g_initialize_once;

void set_last_error(const std::string& message) {
  g_last_error = message;
}

void append_int(std::vector<unsigned char>& out, std::int32_t value) {
  std::uint32_t raw = static_cast<std::uint32_t>(value);
  out.push_back(static_cast<unsigned char>(raw & 0xffU));
  out.push_back(static_cast<unsigned char>((raw >> 8U) & 0xffU));
  out.push_back(static_cast<unsigned char>((raw >> 16U) & 0xffU));
  out.push_back(static_cast<unsigned char>((raw >> 24U) & 0xffU));
}

int param_at(const Log& log, int index) {
  if (index < 0 || index >= log.param.size()) {
    return 0;
  }
  return log.param[index];
}

void append_log_record(
    std::vector<unsigned char>& out,
    LogType type,
    const std::array<int, 7>& params) {
  append_int(out, static_cast<std::int32_t>(type));
  for (int value : params) {
    append_int(out, static_cast<std::int32_t>(value));
  }
}

std::array<int, 7> params(std::initializer_list<int> values) {
  std::array<int, 7> result = {};
  int index = 0;
  for (int value : values) {
    if (index >= static_cast<int>(result.size())) {
      break;
    }
    result[index] = value;
    ++index;
  }
  return result;
}

WireLog visible_log(const Log& log, int player_index) {
  switch (log.logType) {
    case LogType::Draw:
      if (param_at(log, 0) == player_index || player_index == 2) {
        return {log.logType, params({param_at(log, 0), param_at(log, 1), param_at(log, 2)})};
      }
      return {LogType::DrawReverse, params({param_at(log, 0)})};
    case LogType::MoveCard: {
      int open_type = param_at(log, 5);
      bool visible = open_type == 0 ||
                     (open_type == 1 && param_at(log, 0) == player_index) ||
                     (open_type == 3 && player_index == 0) ||
                     (open_type == 4 && player_index == 1) ||
                     player_index == 2;
      if (visible) {
        return {
            log.logType,
            params({
                param_at(log, 0),
                param_at(log, 1),
                param_at(log, 2),
                param_at(log, 3),
                param_at(log, 4),
            }),
        };
      }
      return {
          LogType::MoveCardReverse,
          params({param_at(log, 0), param_at(log, 3), param_at(log, 4)}),
      };
    }
    default:
      break;
  }
  std::array<int, 7> raw = {};
  for (int index = 0; index < 7; ++index) {
    raw[index] = param_at(log, index);
  }
  return {log.logType, raw};
}

struct PokemonRecord {
  int player_index;
  int area;
  int area_index;
  int visible;
  int card_id;
  int serial;
  int hp;
  int max_hp;
  int energy_count;
};

void add_pokemon_record(
    const State& state,
    std::vector<PokemonRecord>& records,
    CardRef ref,
    int player_index,
    AreaType area,
    int area_index) {
  PokemonRecord record = {};
  record.player_index = player_index;
  record.area = static_cast<int>(area);
  record.area_index = area_index;
  if (!ref.isNull()) {
    const Card& card = state.getCard(ref);
    if (!card.reverse) {
      std::vector<CardRef> energy_cards;
      state.getEnergyCards(ref, energy_cards);
      const CardMaster& master = card.getMaster();
      record.visible = 1;
      record.card_id = master.cardId;
      record.serial = ref.cardIndex;
      record.hp = state.getHp(card);
      record.max_hp = state.getMaxHp(card);
      record.energy_count = static_cast<int>(energy_cards.size());
    }
  }
  records.push_back(record);
}

void append_state_block(std::vector<unsigned char>& out, const State* state) {
  if (state == nullptr) {
    append_int(out, 0);
    append_int(out, -1);
    append_int(out, 0);
    append_int(out, 0);
    append_int(out, 0);
    return;
  }
  std::vector<PokemonRecord> records;
  for (int player_index = 0; player_index < 2; ++player_index) {
    const PlayerState& player = state->players[player_index];
    for (int index = 0; index < player.active.size(); ++index) {
      add_pokemon_record(
          *state,
          records,
          player.active[index],
          player_index,
          AreaType::Active,
          index);
    }
    for (int index = 0; index < player.bench.size(); ++index) {
      add_pokemon_record(
          *state,
          records,
          player.bench[index],
          player_index,
          AreaType::Bench,
          index);
    }
  }

  append_int(out, state->selectPlayer);
  append_int(out, state->apiResult());
  append_int(out, state->players[0].prize.size());
  append_int(out, state->players[1].prize.size());
  append_int(out, static_cast<std::int32_t>(records.size()));
  for (const PokemonRecord& record : records) {
    append_int(out, record.player_index);
    append_int(out, record.area);
    append_int(out, record.area_index);
    append_int(out, record.visible);
    append_int(out, record.card_id);
    append_int(out, record.serial);
    append_int(out, record.hp);
    append_int(out, record.max_hp);
    append_int(out, record.energy_count);
  }
}

void collect_visible_logs(State* state, std::vector<WireLog>& records) {
  if (state == nullptr) {
    return;
  }
  int player_index = state->selectPlayer;
  int start_index = state->nextLogStart();
  for (int index = start_index; index < static_cast<int>(state->logs.size()); ++index) {
    const Log& log = state->logs[index];
    if (log.logType > LogType::Result) {
      continue;
    }
    records.push_back(visible_log(log, player_index));
  }
}

void collect_visible_logs_from_cursor(
    const State* state,
    int player_index,
    int& log_cursor,
    std::vector<WireLog>& records) {
  if (state == nullptr) {
    return;
  }
  int log_count = static_cast<int>(state->logs.size());
  int start_index = std::clamp(log_cursor, 0, log_count);
  for (int index = start_index; index < log_count; ++index) {
    const Log& log = state->logs[index];
    if (log.logType > LogType::Result) {
      continue;
    }
    records.push_back(visible_log(log, player_index));
  }
  log_cursor = log_count;
}

void append_logs_block(
    std::vector<unsigned char>& out,
    const std::vector<WireLog>& records) {
  append_int(out, static_cast<std::int32_t>(records.size()));
  for (const WireLog& record : records) {
    append_log_record(out, record.type, record.params);
  }
}

void append_transition(
    std::vector<unsigned char>& out,
    int error,
    bool resolved,
    int forced_steps,
    const State* before,
    const State* after,
    const std::vector<WireLog>& logs) {
  append_int(out, error);
  append_int(out, resolved ? 1 : 0);
  append_int(out, forced_steps);
  append_state_block(out, before);
  append_state_block(out, after);
  append_logs_block(out, logs);
}

bool is_resolved_probe_successor(const State& state) {
  return state.isFinish() ||
         (state.selectType == SelectType::Main &&
          state.selectContext == SelectContext::Main);
}

bool is_resolved_macro_successor(const State& state, int root_player) {
  return state.isFinish() || state.selectPlayer != root_player ||
         (state.selectType == SelectType::Main &&
          state.selectContext == SelectContext::Main);
}

bool forced_probe_action(const State& state, std::vector<int>& selected) {
  int option_count = static_cast<int>(state.options.size());
  int min_count = std::min(option_count, std::max(0, state.selectMin));
  int max_count = std::min(option_count, std::max(min_count, state.selectMax));
  if (option_count == 1 && min_count == 1 && max_count == 1) {
    selected = {0};
    return true;
  }
  if (option_count > 0 && min_count == option_count && max_count == option_count &&
      state.selectContext != SelectContext::SkillOrder) {
    selected.resize(option_count);
    for (int index = 0; index < option_count; ++index) {
      selected[index] = index;
    }
    return true;
  }
  selected.clear();
  return false;
}

bool copy_hidden_list(
    const int* hidden_counts,
    const int* hidden_values,
    int list_index,
    int& value_offset,
    std::vector<int>& destination) {
  int count = hidden_counts[list_index];
  if (count < 0) {
    return false;
  }
  destination.resize(count);
  bool valid = true;
  for (int index = 0; index < count; ++index) {
    int card_id = hidden_values[value_offset++];
    if (!CardTable.contains(card_id)) {
      valid = false;
      continue;
    }
    destination[index] = card_id;
  }
  return valid;
}

bool fill_config(
    const int* hidden_counts,
    const int* hidden_values,
    int world_index,
    int& value_offset,
    bool manual_coin,
    SearchStartConfig& config) {
  int base = world_index * kHiddenListCount;
  config = {};
  config.manualCoin = manual_coin;
  bool valid = true;
  valid &= copy_hidden_list(
      hidden_counts, hidden_values, base + 0, value_offset, config.myDeck);
  valid &= copy_hidden_list(
      hidden_counts, hidden_values, base + 1, value_offset, config.myPrize);
  valid &= copy_hidden_list(
      hidden_counts, hidden_values, base + 2, value_offset, config.enemyDeck);
  valid &= copy_hidden_list(
      hidden_counts, hidden_values, base + 3, value_offset, config.enemyPrize);
  valid &= copy_hidden_list(
      hidden_counts, hidden_values, base + 4, value_offset, config.enemyHand);
  valid &= copy_hidden_list(
      hidden_counts, hidden_values, base + 5, value_offset, config.enemyActive);
  return valid;
}

void restore_determinized_hidden_zones(State& state) {
  // Search::start recreates erased prizes as face-up cards. Restore the source
  // state's face-down invariant so sampled identities stay private and prize
  // abilities retain their engine semantics.
  for (PlayerState& player : state.players) {
    for (CardRef prize : player.prize) {
      if (!prize.isNull()) {
        state.getCard(prize).reverse = true;
      }
    }
  }
}

std::vector<std::vector<int>> read_candidates(
    const int* candidate_counts,
    const int* candidate_values,
    int candidate_count) {
  std::vector<std::vector<int>> candidates;
  candidates.reserve(candidate_count);
  int offset = 0;
  for (int candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
    int count = candidate_counts[candidate_index];
    if (count < 0) {
      throw std::runtime_error("negative candidate select count");
    }
    std::vector<int> selected;
    selected.reserve(count);
    for (int index = 0; index < count; ++index) {
      selected.push_back(candidate_values[offset++]);
    }
    candidates.push_back(std::move(selected));
  }
  return candidates;
}

using MacroCandidate = std::vector<std::vector<int>>;

std::vector<MacroCandidate> read_macro_candidates(
    const int* macro_step_counts,
    const int* step_select_counts,
    const int* step_select_values,
    int macro_count) {
  std::vector<MacroCandidate> macros;
  macros.reserve(macro_count);
  int step_offset = 0;
  int value_offset = 0;
  for (int macro_index = 0; macro_index < macro_count; ++macro_index) {
    int step_count = macro_step_counts[macro_index];
    if (step_count <= 0 || step_count > kMaxMacroSteps) {
      throw std::runtime_error("invalid macro step count");
    }
    MacroCandidate macro;
    macro.reserve(step_count);
    for (int step_index = 0; step_index < step_count; ++step_index) {
      int select_count = step_select_counts[step_offset++];
      if (select_count < 0) {
        throw std::runtime_error("negative macro select count");
      }
      std::vector<int> selected;
      selected.reserve(select_count);
      for (int select_index = 0; select_index < select_count; ++select_index) {
        selected.push_back(step_select_values[value_offset++]);
      }
      macro.push_back(std::move(selected));
    }
    macros.push_back(std::move(macro));
  }
  return macros;
}

std::vector<int> deterministic_completion_action(const State& state) {
  int option_count = static_cast<int>(state.options.size());
  int min_count = std::min(option_count, std::max(0, state.selectMin));
  std::vector<int> selected;
  selected.reserve(min_count);
  for (int index = 0; index < min_count; ++index) {
    selected.push_back(index);
  }
  return selected;
}

void run_fact_root(
    ApiData* data,
    const char* state_token,
    int state_token_count,
    const int* hidden_counts,
    const int* hidden_values,
    int world_count,
    const int* candidate_values,
    int candidate_count,
    bool manual_coin,
    bool require_identical_worlds,
    float* output_features,
    unsigned char* output_masks,
    int& output_error,
    int& output_unresolved) {
  SetBattleData(data, state_token, state_token_count);
  const State base_state = data->state;
  std::vector<DynamicEffectFeatures> first_features(candidate_count);
  std::vector<int> resolved_counts(candidate_count, 0);
  std::vector<bool> feature_mismatch(candidate_count, false);
  int hidden_value_offset = 0;
  output_error = 0;
  output_unresolved = 0;

  for (int world_index = 0; world_index < world_count; ++world_index) {
    SearchStartConfig config;
    const bool valid_hidden = fill_config(
        hidden_counts,
        hidden_values,
        world_index,
        hidden_value_offset,
        manual_coin,
        config);
    if (!valid_hidden) {
      output_error = kErrorInvalidInput;
      return;
    }

    data->search.clear();
    SearchInfo root = data->search.start(config, base_state);
    if (root.errorCode != 0) {
      output_error = root.errorCode;
      data->search.clear();
      return;
    }
    restore_determinized_hidden_zones(*root.state);
    for (int candidate_index = 0; candidate_index < candidate_count;
         ++candidate_index) {
      const std::vector<int> candidate = {candidate_values[candidate_index]};
      std::vector<WireLog> logs;
      SearchInfo successor = data->search.step(root.searchId, candidate);
      if (successor.errorCode != 0) {
        output_error = successor.errorCode;
        data->search.clear();
        return;
      }
      collect_visible_logs(successor.state, logs);

      int forced_steps = 0;
      int transition_error = 0;
      bool resolved = false;
      while (true) {
        if (is_resolved_probe_successor(*successor.state)) {
          resolved = true;
          break;
        }
        std::vector<int> forced;
        if (!forced_probe_action(*successor.state, forced) ||
            forced_steps >= kMaxForcedProbeSteps) {
          break;
        }
        SearchInfo next = data->search.step(successor.searchId, forced);
        if (next.errorCode != 0) {
          transition_error = next.errorCode;
          break;
        }
        collect_visible_logs(next.state, logs);
        data->search.clearSingle(successor.searchId);
        successor = next;
        ++forced_steps;
      }
      if (transition_error != 0) {
        output_error = transition_error;
        data->search.clearSingle(successor.searchId);
        data->search.clear();
        return;
      }
      if (!resolved) {
        ++output_unresolved;
      } else {
        const DynamicEffectFeatures features =
            cg_probe_internal::BuildDynamicEffectFeatures(
                *root.state, *successor.state, logs);
        if (resolved_counts[candidate_index] == 0) {
          first_features[candidate_index] = features;
        } else if (features != first_features[candidate_index]) {
          feature_mismatch[candidate_index] = true;
        }
        ++resolved_counts[candidate_index];
      }
      data->search.clearSingle(successor.searchId);
    }
    data->search.clear();
  }

  for (int candidate_index = 0; candidate_index < candidate_count;
       ++candidate_index) {
    const bool usable = resolved_counts[candidate_index] == world_count &&
                        (!require_identical_worlds ||
                         !feature_mismatch[candidate_index]);
    output_masks[candidate_index] = usable ? 1 : 0;
    if (!usable) {
      continue;
    }
    std::copy(
        first_features[candidate_index].begin(),
        first_features[candidate_index].end(),
        output_features +
            candidate_index * cg_probe_internal::kDynamicEffectFeatureWidth);
  }
}

}  // namespace

extern "C" {

struct CgProbeResult {
  int error;
  const unsigned char* data;
  int size;
};

CG_PROBE_API void CgProbeInitialize() {
  std::call_once(g_initialize_once, InitializeAll);
}

CG_PROBE_API const char* CgProbeLastError() {
  return g_last_error.c_str();
}

CG_PROBE_API int CgProbePayloadVersion() {
  return kVersion;
}

CG_PROBE_API int CgProbeFactVersion() {
  return kFactVersion;
}

CG_PROBE_API int CgProbeFactWidth() {
  return cg_probe_internal::kDynamicEffectFeatureWidth;
}

CG_PROBE_API CgProbeResult CgProbeBatch(
    const char* state_token,
    int state_token_count,
    const int* hidden_counts,
    const int* hidden_values,
    int world_count,
    const int* candidate_counts,
    const int* candidate_values,
    int candidate_count,
    int manual_coin) {
  try {
    CgProbeInitialize();
    if (state_token == nullptr || state_token_count <= 0 || hidden_counts == nullptr ||
        hidden_values == nullptr || candidate_counts == nullptr ||
        candidate_values == nullptr || world_count <= 0 || candidate_count <= 0) {
      set_last_error("invalid CgProbeBatch input");
      return {kErrorInvalidInput, nullptr, 0};
    }

    std::unique_ptr<ApiData, void (*)(ApiData*)> data(ApiAgentStart(), ApiBattleFinish);
    SetBattleData(data.get(), state_token, state_token_count);
    State base_state = data->state;
    std::vector<std::vector<int>> candidates =
        read_candidates(candidate_counts, candidate_values, candidate_count);

    std::vector<unsigned char> payload;
    append_int(payload, kMagic);
    append_int(payload, kVersion);
    append_int(payload, world_count);
    append_int(payload, candidate_count);

    int hidden_value_offset = 0;
    for (int world_index = 0; world_index < world_count; ++world_index) {
      SearchStartConfig config;
      bool valid_hidden = fill_config(
          hidden_counts,
          hidden_values,
          world_index,
          hidden_value_offset,
          manual_coin != 0,
          config);
      if (!valid_hidden) {
        for (int candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
          append_transition(
              payload,
              kErrorInvalidInput,
              false,
              0,
              nullptr,
              nullptr,
              {});
        }
        continue;
      }

      data->search.clear();
      SearchInfo root = data->search.start(config, base_state);
      if (root.errorCode != 0) {
        for (int candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
          append_transition(
              payload,
              root.errorCode,
              false,
              0,
              nullptr,
              nullptr,
              {});
        }
        continue;
      }
      restore_determinized_hidden_zones(*root.state);
      for (const std::vector<int>& candidate : candidates) {
        std::vector<WireLog> logs;
        SearchInfo successor = data->search.step(root.searchId, candidate);
        if (successor.errorCode != 0) {
          append_transition(
              payload,
              successor.errorCode,
              false,
              0,
              root.state,
              nullptr,
              logs);
          continue;
        }
        collect_visible_logs(successor.state, logs);

        int forced_steps = 0;
        int transition_error = 0;
        bool resolved = false;
        while (true) {
          if (is_resolved_probe_successor(*successor.state)) {
            resolved = true;
            break;
          }
          std::vector<int> forced;
          if (!forced_probe_action(*successor.state, forced) ||
              forced_steps >= kMaxForcedProbeSteps) {
            break;
          }
          SearchInfo next = data->search.step(successor.searchId, forced);
          if (next.errorCode != 0) {
            transition_error = next.errorCode;
            break;
          }
          collect_visible_logs(next.state, logs);
          data->search.clearSingle(successor.searchId);
          successor = next;
          ++forced_steps;
        }
        append_transition(
            payload,
            transition_error,
            resolved,
            forced_steps,
            root.state,
            successor.state,
            logs);
        data->search.clearSingle(successor.searchId);
      }
      data->search.clear();
    }

    unsigned char* data_buffer = new unsigned char[payload.size()];
    std::copy(payload.begin(), payload.end(), data_buffer);
    set_last_error("");
    return {0, data_buffer, static_cast<int>(payload.size())};
  } catch (const std::exception& error) {
    set_last_error(error.what());
    return {kErrorException, nullptr, 0};
  } catch (...) {
    set_last_error("unknown native probe failure");
    return {kErrorException, nullptr, 0};
  }
}

// Execute fully specified action sequences directly through the native Search
// implementation.  This benchmark-oriented entry point deliberately accepts
// complete macros instead of inventing card semantics.  When requested, any
// prompt remaining after the explicit sequence is completed by selecting the
// minimum legal prefix until MAIN/terminal or the bounded completion limit.
CG_PROBE_API CgProbeResult CgProbeMacroBatch(
    const char* state_token,
    int state_token_count,
    const int* hidden_counts,
    const int* hidden_values,
    int world_count,
    const int* macro_step_counts,
    const int* step_select_counts,
    const int* step_select_values,
    int macro_count,
    int manual_coin,
    int complete_to_boundary) {
  try {
    CgProbeInitialize();
    if (state_token == nullptr || state_token_count <= 0 ||
        hidden_counts == nullptr || hidden_values == nullptr ||
        macro_step_counts == nullptr || step_select_counts == nullptr ||
        step_select_values == nullptr || world_count <= 0 || macro_count <= 0) {
      set_last_error("invalid CgProbeMacroBatch input");
      return {kErrorInvalidInput, nullptr, 0};
    }

    std::unique_ptr<ApiData, void (*)(ApiData*)> data(ApiAgentStart(), ApiBattleFinish);
    SetBattleData(data.get(), state_token, state_token_count);
    State base_state = data->state;
    std::vector<MacroCandidate> macros = read_macro_candidates(
        macro_step_counts,
        step_select_counts,
        step_select_values,
        macro_count);

    std::vector<unsigned char> payload;
    append_int(payload, kMagic);
    append_int(payload, kVersion);
    append_int(payload, world_count);
    append_int(payload, macro_count);

    int hidden_value_offset = 0;
    for (int world_index = 0; world_index < world_count; ++world_index) {
      SearchStartConfig config;
      bool valid_hidden = fill_config(
          hidden_counts,
          hidden_values,
          world_index,
          hidden_value_offset,
          manual_coin != 0,
          config);
      if (!valid_hidden) {
        for (int macro_index = 0; macro_index < macro_count; ++macro_index) {
          append_transition(
              payload,
              kErrorInvalidInput,
              false,
              0,
              nullptr,
              nullptr,
              {});
        }
        continue;
      }

      data->search.clear();
      SearchInfo root = data->search.start(config, base_state);
      if (root.errorCode != 0) {
        for (int macro_index = 0; macro_index < macro_count; ++macro_index) {
          append_transition(
              payload,
              root.errorCode,
              false,
              0,
              nullptr,
              nullptr,
              {});
        }
        continue;
      }
      restore_determinized_hidden_zones(*root.state);

      for (const MacroCandidate& macro : macros) {
        std::vector<WireLog> logs;
        SearchInfo current = root;
        bool owns_current = false;
        int transition_error = 0;
        int log_cursor = static_cast<int>(root.state->logs.size());
        int root_player = root.state->selectPlayer;
        for (const std::vector<int>& selected : macro) {
          if (owns_current &&
              is_resolved_macro_successor(*current.state, root_player)) {
            transition_error = kErrorMacroTrailingAction;
            break;
          }
          SearchInfo next = data->search.step(current.searchId, selected);
          if (next.errorCode != 0) {
            transition_error = next.errorCode;
            break;
          }
          collect_visible_logs_from_cursor(
              next.state, root_player, log_cursor, logs);
          if (owns_current) {
            data->search.clearSingle(current.searchId);
          }
          current = next;
          owns_current = true;
        }

        int completion_steps = 0;
        while (transition_error == 0 && owns_current &&
               !is_resolved_macro_successor(*current.state, root_player) &&
               complete_to_boundary != 0 &&
               completion_steps < kMaxDeterministicCompletionSteps) {
          std::vector<int> selected = deterministic_completion_action(*current.state);
          SearchInfo next = data->search.step(current.searchId, selected);
          if (next.errorCode != 0) {
            transition_error = next.errorCode;
            break;
          }
          collect_visible_logs_from_cursor(
              next.state, root_player, log_cursor, logs);
          data->search.clearSingle(current.searchId);
          current = next;
          ++completion_steps;
        }

        // A malformed macro with trailing actions still reached a meaningful
        // boundary.  Preserve that endpoint metadata while returning the
        // explicit trailing-action error, and never execute the extra action.
        bool resolved = owns_current &&
                        is_resolved_macro_successor(*current.state, root_player);
        append_transition(
            payload,
            transition_error,
            resolved,
            completion_steps,
            root.state,
            owns_current ? current.state : nullptr,
            logs);
        if (owns_current) {
          data->search.clearSingle(current.searchId);
        }
      }
      data->search.clear();
    }

    unsigned char* data_buffer = new unsigned char[payload.size()];
    std::copy(payload.begin(), payload.end(), data_buffer);
    set_last_error("");
    return {0, data_buffer, static_cast<int>(payload.size())};
  } catch (const std::exception& error) {
    set_last_error(error.what());
    return {kErrorException, nullptr, 0};
  } catch (...) {
    set_last_error("unknown native macro probe failure");
    return {kErrorException, nullptr, 0};
  }
}

// Execute a ragged collection of privacy-erased training roots entirely in
// native code and emit only the 33-wide model facts. Candidate selections are
// intentionally singleton option indices: that is the exact ATTACK/ABILITY
// surface used by native rollout collection. Python remains responsible for
// belief sampling and for placing the returned rows into model tensors.
CG_PROBE_API int CgProbeFactBatch(
    const char* state_tokens,
    const int* state_token_offsets,
    int root_count,
    const int* hidden_counts,
    const int* hidden_values,
    const int* hidden_value_offsets,
    int world_count,
    const int* candidate_offsets,
    const int* candidate_values,
    int manual_coin,
    int require_identical_worlds,
    float* output_features,
    unsigned char* output_masks,
    int* output_root_errors,
    int* output_root_unresolved) {
  try {
    CgProbeInitialize();
    if (state_tokens == nullptr || state_token_offsets == nullptr ||
        hidden_counts == nullptr || hidden_values == nullptr ||
        hidden_value_offsets == nullptr || candidate_offsets == nullptr ||
        candidate_values == nullptr || output_features == nullptr ||
        output_masks == nullptr || output_root_errors == nullptr ||
        output_root_unresolved == nullptr || root_count <= 0 ||
        world_count <= 0) {
      set_last_error("invalid CgProbeFactBatch input");
      return kErrorInvalidInput;
    }
    if (state_token_offsets[0] != 0 || hidden_value_offsets[0] != 0 ||
        candidate_offsets[0] != 0) {
      set_last_error("CgProbeFactBatch offsets must start at zero");
      return kErrorInvalidInput;
    }

    std::unique_ptr<ApiData, void (*)(ApiData*)> data(
        ApiAgentStart(), ApiBattleFinish);
    for (int root_index = 0; root_index < root_count; ++root_index) {
      const int state_start = state_token_offsets[root_index];
      const int state_stop = state_token_offsets[root_index + 1];
      const int hidden_start = hidden_value_offsets[root_index];
      const int hidden_stop = hidden_value_offsets[root_index + 1];
      const int candidate_start = candidate_offsets[root_index];
      const int candidate_stop = candidate_offsets[root_index + 1];
      if (state_start < 0 || state_stop <= state_start ||
          hidden_start < 0 || hidden_stop < hidden_start ||
          candidate_start < 0 || candidate_stop <= candidate_start) {
        set_last_error("CgProbeFactBatch offsets are invalid");
        return kErrorInvalidInput;
      }
      const int expected_hidden_values = std::accumulate(
          hidden_counts + root_index * world_count * kHiddenListCount,
          hidden_counts +
              (root_index + 1) * world_count * kHiddenListCount,
          0);
      if (expected_hidden_values < 0 ||
          hidden_stop - hidden_start != expected_hidden_values) {
        set_last_error("CgProbeFactBatch hidden values are misaligned");
        return kErrorInvalidInput;
      }
      const int candidate_count = candidate_stop - candidate_start;
      std::fill(
          output_features +
              candidate_start *
                  cg_probe_internal::kDynamicEffectFeatureWidth,
          output_features +
              candidate_stop *
                  cg_probe_internal::kDynamicEffectFeatureWidth,
          0.0F);
      std::fill(
          output_masks + candidate_start,
          output_masks + candidate_stop,
          static_cast<unsigned char>(0));
      run_fact_root(
          data.get(),
          state_tokens + state_start,
          state_stop - state_start,
          hidden_counts + root_index * world_count * kHiddenListCount,
          hidden_values + hidden_start,
          world_count,
          candidate_values + candidate_start,
          candidate_count,
          manual_coin != 0,
          require_identical_worlds != 0,
          output_features +
              candidate_start *
                  cg_probe_internal::kDynamicEffectFeatureWidth,
          output_masks + candidate_start,
          output_root_errors[root_index],
          output_root_unresolved[root_index]);
    }
    set_last_error("");
    return 0;
  } catch (const std::exception& error) {
    set_last_error(error.what());
    return kErrorException;
  } catch (...) {
    set_last_error("unknown native fact failure");
    return kErrorException;
  }
}

CG_PROBE_API void CgProbeFree(const unsigned char* data) {
  delete[] data;
}

}  // extern "C"

// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®,
// and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only

#include "rollout_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "parallel_executor.h"

namespace cg_train_rollout {
namespace {

constexpr std::array<std::uint32_t, 24> kExpectedLogParamCounts = {
    1, 2, 1, 1, 3, 1, 5, 3, 5, 5, 3, 5,
    5, 5, 7, 4, 5, 4, 4, 4, 4, 4, 2, 2,
};

constexpr std::int32_t kLogTurnStart = 2;
constexpr std::int32_t kLogTurnEnd = 3;
constexpr std::int32_t kLogDraw = 4;
constexpr std::int32_t kLogDrawReverse = 5;
constexpr std::int32_t kLogMoveCard = 6;
constexpr std::int32_t kLogMoveCardReverse = 7;
constexpr std::int32_t kLogSwitch = 8;
constexpr std::int32_t kLogChange = 9;
constexpr std::int32_t kLogPlay = 10;
constexpr std::int32_t kLogAttach = 11;
constexpr std::int32_t kLogEvolve = 12;
constexpr std::int32_t kLogDevolve = 13;
constexpr std::int32_t kLogMoveAttached = 14;
constexpr std::int32_t kLogAttack = 15;
constexpr std::int32_t kLogHpChange = 16;
constexpr std::int32_t kLogPoisoned = 17;
constexpr std::int32_t kLogBurned = 18;
constexpr std::int32_t kLogAsleep = 19;
constexpr std::int32_t kLogParalyzed = 20;
constexpr std::int32_t kLogConfused = 21;
constexpr std::int32_t kLogResult = 23;

constexpr std::int32_t kAreaVirtual = 0;
constexpr std::int32_t kAreaDeck = 1;
constexpr std::int32_t kAreaHand = 2;
constexpr std::int32_t kLowDeckReboundMaximum = 5;

bool IsSuccess(std::int32_t status) {
  return status == kReadyStatus || status == kFinishedStatus;
}

template <typename T>
void RequirePointer(const T* pointer, const char* name) {
  if (pointer == nullptr) {
    throw std::invalid_argument(std::string("null rollout source pointer: ") +
                                name);
  }
}

std::size_t CheckedProduct(std::size_t left, std::size_t right,
                           const char* name) {
  constexpr std::size_t kMaximum =
      static_cast<std::size_t>(
          std::numeric_limits<std::ptrdiff_t>::max());
  if (right != 0 && left > kMaximum / right) {
    throw std::invalid_argument(std::string("rollout ") + name +
                                " size exceeds the addressable range");
  }
  return left * right;
}

std::size_t CheckedStride(std::uint32_t card_vocab_size) {
  constexpr std::size_t kMaximum =
      static_cast<std::size_t>(
          std::numeric_limits<std::ptrdiff_t>::max());
  const std::size_t vocab = card_vocab_size;
  if (vocab >= kMaximum) {
    throw std::invalid_argument(
        "rollout catalog vocabulary exceeds the addressable range");
  }
  return vocab + 1;
}

std::int32_t Param(const CgTrainOutput& source, std::size_t column,
                   std::uint32_t row) {
  switch (column) {
    case 0:
      return source.log_p0[row];
    case 1:
      return source.log_p1[row];
    case 2:
      return source.log_p2[row];
    case 3:
      return source.log_p3[row];
    case 4:
      return source.log_p4[row];
    case 5:
      return source.log_p5[row];
    case 6:
      return source.log_p6[row];
    default:
      throw std::logic_error("public log parameter column is out of range");
  }
}

std::int32_t LogPlayer(const CgTrainOutput& source, std::uint32_t index) {
  return source.log_type[index] == kLogResult ? -1 : source.log_p0[index];
}

std::int32_t DeckDelta(const CgTrainOutput& source, std::uint32_t index) {
  const std::int32_t type = source.log_type[index];
  if (type == kLogDraw || type == kLogDrawReverse) {
    return -1;
  }
  if (type == kLogMoveCard) {
    return (source.log_p4[index] == kAreaDeck ? 1 : 0) -
           (source.log_p3[index] == kAreaDeck ? 1 : 0);
  }
  if (type == kLogMoveCardReverse) {
    return (source.log_p2[index] == kAreaDeck ? 1 : 0) -
           (source.log_p1[index] == kAreaDeck ? 1 : 0);
  }
  return 0;
}

void Increment(std::map<std::int32_t, std::int32_t>* values,
               std::int32_t key, std::int32_t amount = 1) {
  (*values)[key] += amount;
}

CountRow PositiveCounts(
    const std::map<std::int32_t, std::int32_t>& values) {
  CountRow result;
  result.reserve(values.size());
  for (const auto& [card_id, count] : values) {
    if (card_id > 0 && count > 0) {
      result.emplace_back(card_id, count);
    }
  }
  return result;
}

void ValidateSource(const CgTrainOutput& source, std::uint32_t batch_count) {
  if (source.struct_size != sizeof(CgTrainOutput)) {
    throw std::invalid_argument("rollout source struct size is incompatible");
  }
  if (batch_count == 0 || batch_count > source.slot_capacity) {
    throw std::invalid_argument("rollout source row count is invalid");
  }
  RequirePointer(source.status, "status");
  RequirePointer(source.error, "error");
  RequirePointer(source.select_player, "select_player");
  RequirePointer(source.select_type, "select_type");
  RequirePointer(source.select_context, "select_context");
  RequirePointer(source.select_min, "select_min");
  RequirePointer(source.select_max, "select_max");
  RequirePointer(source.result, "result");
  RequirePointer(source.turn, "turn");
  RequirePointer(source.option_offsets, "option_offsets");
  RequirePointer(source.option_type, "option_type");
  RequirePointer(source.option_p0, "option_p0");
  RequirePointer(source.option_p1, "option_p1");
  RequirePointer(source.option_p2, "option_p2");
  RequirePointer(source.option_p3, "option_p3");
  RequirePointer(source.option_p4, "option_p4");
  RequirePointer(source.turn_action_count, "turn_action_count");
  RequirePointer(source.first_player, "first_player");
  RequirePointer(source.turn_flags, "turn_flags");
  RequirePointer(source.remain_damage_counter, "remain_damage_counter");
  RequirePointer(source.remain_energy_cost, "remain_energy_cost");
  RequirePointer(source.player0_deck_count, "player0_deck_count");
  RequirePointer(source.player1_deck_count, "player1_deck_count");
  RequirePointer(source.player0_hand_count, "player0_hand_count");
  RequirePointer(source.player1_hand_count, "player1_hand_count");
  RequirePointer(source.player0_prize_count, "player0_prize_count");
  RequirePointer(source.player1_prize_count, "player1_prize_count");
  RequirePointer(source.player0_bench_max, "player0_bench_max");
  RequirePointer(source.player1_bench_max, "player1_bench_max");
  RequirePointer(source.player0_status_flags, "player0_status_flags");
  RequirePointer(source.player1_status_flags, "player1_status_flags");
  RequirePointer(source.context_card_row, "context_card_row");
  RequirePointer(source.effect_card_row, "effect_card_row");
  RequirePointer(source.visible_card_offsets, "visible_card_offsets");
  RequirePointer(source.attachment_offsets, "attachment_offsets");
  RequirePointer(source.log_offsets, "log_offsets");
  if (source.option_offsets[0] != 0 || source.visible_card_offsets[0] != 0 ||
      source.attachment_offsets[0] != 0 || source.log_offsets[0] != 0) {
    throw std::invalid_argument("rollout source CSR must begin at zero");
  }
  for (std::uint32_t row = 0; row < batch_count; ++row) {
    if (source.option_offsets[row + 1] < source.option_offsets[row] ||
        source.visible_card_offsets[row + 1] <
            source.visible_card_offsets[row] ||
        source.attachment_offsets[row + 1] <
            source.attachment_offsets[row] ||
        source.log_offsets[row + 1] < source.log_offsets[row]) {
      throw std::invalid_argument("rollout source CSR is not monotonic");
    }
  }
  const std::uint32_t option_count = source.option_offsets[batch_count];
  const std::uint32_t visible_count = source.visible_card_offsets[batch_count];
  const std::uint32_t attachment_count =
      source.attachment_offsets[batch_count];
  const std::uint32_t log_count = source.log_offsets[batch_count];
  if (option_count > source.option_capacity ||
      visible_count > source.visible_card_capacity ||
      attachment_count > source.attachment_capacity ||
      log_count > source.log_capacity) {
    throw std::invalid_argument("rollout source CSR exceeds declared capacity");
  }
  if (visible_count > 0) {
    RequirePointer(source.visible_card_owner, "visible_card_owner");
    RequirePointer(source.visible_card_area, "visible_card_area");
    RequirePointer(source.visible_card_area_index, "visible_card_area_index");
    RequirePointer(source.visible_card_id, "visible_card_id");
    RequirePointer(source.visible_card_serial, "visible_card_serial");
    RequirePointer(source.visible_card_hp, "visible_card_hp");
    RequirePointer(source.visible_card_max_hp, "visible_card_max_hp");
    RequirePointer(source.visible_card_appear_this_turn,
                   "visible_card_appear_this_turn");
  }
  if (attachment_count > 0) {
    RequirePointer(source.attachment_parent, "attachment_parent");
    RequirePointer(source.attachment_kind, "attachment_kind");
    RequirePointer(source.attachment_card_id, "attachment_card_id");
    RequirePointer(source.attachment_card_serial, "attachment_card_serial");
    RequirePointer(source.attachment_energy_type, "attachment_energy_type");
    RequirePointer(source.attachment_energy_units, "attachment_energy_units");
  }
  if (log_count > 0) {
    RequirePointer(source.log_type, "log_type");
    RequirePointer(source.log_param_count, "log_param_count");
    RequirePointer(source.log_p0, "log_p0");
    RequirePointer(source.log_p1, "log_p1");
    RequirePointer(source.log_p2, "log_p2");
    RequirePointer(source.log_p3, "log_p3");
    RequirePointer(source.log_p4, "log_p4");
    RequirePointer(source.log_p5, "log_p5");
    RequirePointer(source.log_p6, "log_p6");
  }
}

CountRow DeckCounts(const std::int32_t* deck, std::uint32_t card_vocab_size) {
  std::map<std::int32_t, std::int32_t> counts;
  for (std::size_t index = 0; index < kDeckSize; ++index) {
    if (deck[index] <= 0 ||
        static_cast<std::uint32_t>(deck[index]) > card_vocab_size) {
      throw std::invalid_argument(
          "rollout reset deck contains a card outside the catalog vocabulary");
    }
    ++counts[deck[index]];
  }
  return PositiveCounts(counts);
}

void RecordIdentity(std::int32_t card_id, std::int32_t serial,
                    PerspectiveHistory* history) {
  if (card_id <= 0) {
    return;
  }
  const auto [unused, inserted] =
      history->revealed_by_serial.emplace(serial, card_id);
  if (inserted) {
    Increment(&history->revealed_counts, card_id);
  }
}

std::vector<std::pair<std::int32_t, std::int32_t>> RevealedPairs(
    const CgTrainOutput& source, std::uint32_t index) {
  const std::int32_t type = source.log_type[index];
  if (type == kLogDraw || type == kLogMoveCard || type == kLogPlay ||
      type == kLogAttack || type == kLogHpChange) {
    return {{source.log_p1[index], source.log_p2[index]}};
  }
  if (type == kLogSwitch || type == kLogChange || type == kLogAttach ||
      type == kLogEvolve || type == kLogDevolve) {
    return {{source.log_p1[index], source.log_p2[index]},
            {source.log_p3[index], source.log_p4[index]}};
  }
  if (type == kLogMoveAttached) {
    return {{source.log_p1[index], source.log_p2[index]},
            {source.log_p3[index], source.log_p4[index]},
            {source.log_p5[index], source.log_p6[index]}};
  }
  if (type == kLogPoisoned || type == kLogBurned ||
      type == kLogAsleep || type == kLogParalyzed ||
      type == kLogConfused) {
    return {{source.log_p2[index], source.log_p3[index]}};
  }
  return {};
}

void UpdateHistory(const CgTrainOutput& source, std::uint32_t index,
                   std::int32_t player, const CatalogData& catalog,
                   PerspectiveHistory* history) {
  const std::int32_t type = source.log_type[index];
  if (type == kLogAttack) {
    ++history->history[player][0];
    history->turn_attacked[player] = true;
    const std::int32_t serial = source.log_p2[index];
    const std::int32_t attack_id = source.log_p3[index];
    if (serial > 0 && attack_id > 0) {
      history->last_attack_by_serial[serial] = attack_id;
    }
  } else if (type == kLogPlay) {
    if (catalog.supporter_card_ids.contains(source.log_p1[index])) {
      ++history->history[player][1];
    }
  } else if (type == kLogAttach) {
    ++history->history[player][2];
  } else if (type == kLogSwitch) {
    ++history->history[player][3];
  }
}

void UpdateDeckFlow(const CgTrainOutput& source, std::uint32_t index,
                    std::int32_t player, std::int32_t deck_count_before,
                    std::int32_t deck_delta,
                    PerspectiveHistory* history) {
  const std::int32_t type = source.log_type[index];
  if (type == kLogTurnStart) {
    history->current_draw_counts[player] = 0;
    history->current_return_counts[player] = 0;
    history->current_deck_deltas[player] = 0;
    history->turn_open[player] = true;
    history->turn_attacked[player] = false;
    history->turn_rebounded[player] = false;
    return;
  }
  if (type == kLogTurnEnd) {
    if (history->turn_open[player]) {
      history->recent_draw_counts[player] =
          history->current_draw_counts[player];
      history->recent_return_counts[player] =
          history->current_return_counts[player];
      history->recent_deck_deltas[player] =
          history->current_deck_deltas[player];
      if (history->turn_attacked[player]) {
        history->no_attack_turns[player] = 0;
      } else {
        ++history->no_attack_turns[player];
      }
      history->turn_open[player] = false;
    }
    return;
  }
  if (type == kLogDraw || type == kLogDrawReverse) {
    ++history->draw_counts[player];
    if (history->turn_open[player]) {
      ++history->current_draw_counts[player];
      --history->current_deck_deltas[player];
    }
    return;
  }
  if (type != kLogMoveCard && type != kLogMoveCardReverse) {
    return;
  }
  const std::int32_t from_area =
      type == kLogMoveCard ? source.log_p3[index] : source.log_p1[index];
  const std::int32_t to_area =
      type == kLogMoveCard ? source.log_p4[index] : source.log_p2[index];
  if (history->turn_open[player]) {
    if (from_area == kAreaDeck) {
      --history->current_deck_deltas[player];
    }
    if (to_area == kAreaDeck) {
      ++history->current_deck_deltas[player];
    }
  }
  if (to_area != kAreaDeck) {
    return;
  }
  ++history->return_counts[player];
  if (history->turn_open[player]) {
    ++history->current_return_counts[player];
    if (deck_count_before <= kLowDeckReboundMaximum && deck_delta > 0 &&
        !history->turn_rebounded[player]) {
      ++history->rebound_counts[player];
      history->turn_rebounded[player] = true;
    }
  }
}

}  // namespace

std::string& LastError() {
  thread_local std::string error;
  return error;
}

std::int32_t Fail(std::int32_t code, std::string message) {
  LastError() = std::move(message);
  return code;
}

RolloutEncoderCore::RolloutEncoderCore(
    std::uint32_t slot_capacity, const CgTrainRolloutCatalog& catalog)
    : slot_capacity_(slot_capacity), slots_(slot_capacity) {
  if (slot_capacity == 0) {
    throw std::invalid_argument("rollout slot capacity must be positive");
  }
  if (catalog.struct_size != sizeof(CgTrainRolloutCatalog) ||
      catalog.deck_count == 0 || catalog.card_vocab_size == 0 ||
      catalog.posterior_cache_capacity == 0 ||
      catalog.deck_count >
          static_cast<std::uint32_t>(
              std::numeric_limits<std::int32_t>::max()) ||
      catalog.card_vocab_size >
          static_cast<std::uint32_t>(
              std::numeric_limits<std::int32_t>::max()) ||
      catalog.supporter_count > catalog.card_vocab_size) {
    throw std::invalid_argument("rollout catalog descriptor is invalid");
  }
  RequirePointer(catalog.entry_counts, "catalog.entry_counts");
  RequirePointer(catalog.exact_log_priors, "catalog.exact_log_priors");
  RequirePointer(catalog.log_combinations, "catalog.log_combinations");
  RequirePointer(catalog.log_factorials, "catalog.log_factorials");
  RequirePointer(catalog.unknown_card_probabilities,
                 "catalog.unknown_card_probabilities");
  RequirePointer(catalog.unknown_log_card_probabilities,
                 "catalog.unknown_log_card_probabilities");
  if (catalog.supporter_count > 0) {
    RequirePointer(catalog.supporter_card_ids, "catalog.supporter_card_ids");
  }
  catalog_.deck_count = catalog.deck_count;
  catalog_.card_vocab_size = catalog.card_vocab_size;
  catalog_.posterior_cache_capacity = catalog.posterior_cache_capacity;
  const std::size_t stride = CheckedStride(catalog.card_vocab_size);
  const std::size_t entry_values =
      CheckedProduct(catalog.deck_count, stride, "catalog entry");
  catalog_.entry_counts.assign(catalog.entry_counts,
                               catalog.entry_counts + entry_values);
  catalog_.exact_log_priors.assign(
      catalog.exact_log_priors,
      catalog.exact_log_priors + catalog.deck_count);
  std::copy_n(catalog.log_combinations, catalog_.log_combinations.size(),
              catalog_.log_combinations.begin());
  std::copy_n(catalog.log_factorials, catalog_.log_factorials.size(),
              catalog_.log_factorials.begin());
  catalog_.unknown_card_probabilities.assign(
      catalog.unknown_card_probabilities,
      catalog.unknown_card_probabilities + catalog.card_vocab_size);
  catalog_.unknown_log_card_probabilities.assign(
      catalog.unknown_log_card_probabilities,
      catalog.unknown_log_card_probabilities + catalog.card_vocab_size);
  catalog_.unknown_log_prior = catalog.unknown_log_prior;
  for (std::uint32_t index = 0; index < catalog.supporter_count; ++index) {
    const std::int32_t card_id = catalog.supporter_card_ids[index];
    if (card_id <= 0 ||
        static_cast<std::uint32_t>(card_id) > catalog.card_vocab_size) {
      throw std::invalid_argument(
          "rollout supporter card ID is outside the catalog vocabulary");
    }
    catalog_.supporter_card_ids.insert(card_id);
  }
  if (!std::isfinite(catalog_.unknown_log_prior) ||
      std::any_of(catalog_.exact_log_priors.begin(),
                  catalog_.exact_log_priors.end(),
                  [](double value) { return !std::isfinite(value); }) ||
      std::any_of(catalog_.unknown_card_probabilities.begin(),
                  catalog_.unknown_card_probabilities.end(),
                  [](double value) {
                    return !std::isfinite(value) || value <= 0.0;
                  }) ||
      std::any_of(catalog_.unknown_log_card_probabilities.begin(),
                  catalog_.unknown_log_card_probabilities.end(),
                  [](double value) { return !std::isfinite(value); }) ||
      std::any_of(catalog_.log_factorials.begin(),
                  catalog_.log_factorials.end(),
                  [](double value) { return !std::isfinite(value); })) {
    throw std::invalid_argument("rollout catalog numeric columns are invalid");
  }
  for (std::size_t deck = 0; deck < catalog_.deck_count; ++deck) {
    const std::int16_t* counts =
        catalog_.entry_counts.data() + deck * stride;
    if (counts[0] != 0) {
      throw std::invalid_argument(
          "rollout catalog sentinel count must be zero");
    }
    std::int32_t total = 0;
    for (std::size_t card_id = 1; card_id < stride; ++card_id) {
      if (counts[card_id] < 0 ||
          counts[card_id] > static_cast<std::int32_t>(kDeckSize)) {
        throw std::invalid_argument(
            "rollout catalog card count is outside deck bounds");
      }
      total += counts[card_id];
      if (total > static_cast<std::int32_t>(kDeckSize)) {
        throw std::invalid_argument(
            "rollout catalog exact deck exceeds 60 cards");
      }
    }
    if (total != static_cast<std::int32_t>(kDeckSize)) {
      throw std::invalid_argument(
          "rollout catalog exact deck does not contain 60 cards");
    }
  }
  for (std::size_t available = 0; available <= kDeckSize; ++available) {
    for (std::size_t observed = 0; observed <= available; ++observed) {
      if (!std::isfinite(
              catalog_.log_combinations[available * 61 + observed])) {
        throw std::invalid_argument(
            "rollout catalog combination table is invalid");
      }
    }
  }
  double unknown_probability_sum = 0.0;
  for (std::size_t index = 0;
       index < catalog_.unknown_card_probabilities.size(); ++index) {
    const double probability = catalog_.unknown_card_probabilities[index];
    unknown_probability_sum += probability;
    if (std::abs(std::log(probability) -
                 catalog_.unknown_log_card_probabilities[index]) > 1.0e-12) {
      throw std::invalid_argument(
          "rollout catalog unknown probability columns disagree");
    }
  }
  if (std::abs(unknown_probability_sum - 1.0) > 1.0e-9) {
    throw std::invalid_argument(
        "rollout catalog unknown card probabilities do not sum to one");
  }
}

void RolloutEncoderCore::ConsumeReset(
    std::uint32_t batch_count, const std::uint32_t* slots,
    const std::int32_t* decks, const CgTrainOutput& source) {
  Consume(true, batch_count, slots, decks, source);
}

void RolloutEncoderCore::ConsumeStep(
    std::uint32_t batch_count, const std::uint32_t* slots,
    const CgTrainOutput& source) {
  Consume(false, batch_count, slots, nullptr, source);
}

void RolloutEncoderCore::Consume(
    bool reset, std::uint32_t batch_count, const std::uint32_t* slot_values,
    const std::int32_t* decks, const CgTrainOutput& source) {
  ValidateSource(source, batch_count);
  RequirePointer(slot_values, "slots");
  if (reset) {
    RequirePointer(decks, "reset_decks");
  }
  std::set<std::uint32_t> unique_slots;
  for (std::uint32_t row = 0; row < batch_count; ++row) {
    const std::uint32_t slot = slot_values[row];
    if (slot >= slot_capacity_ || !unique_slots.insert(slot).second) {
      throw std::invalid_argument(
          "rollout consume slots must be unique and in range");
    }
  }
  std::vector<std::pair<std::uint32_t, SlotState>> staged(batch_count);
  cg_train_internal::ParallelExecutor& executor =
      cg_train_internal::ParallelExecutor::Global();
  executor.ParallelFor(
      batch_count, executor.worker_count(), [&](std::size_t row) {
        const std::uint32_t slot = slot_values[row];
        RawRow raw = CopyRow(source, row);
        SlotState candidate = slots_[slot];
        if (reset && IsSuccess(raw.status)) {
          candidate = SlotState{};
          for (std::int32_t perspective = 0; perspective < 2;
               ++perspective) {
            PerspectiveHistory history;
            history.initialized = true;
            history.perspective = perspective;
            const std::size_t deck_base =
                (static_cast<std::size_t>(row) * 2 + perspective) *
                kDeckSize;
            history.own_deck_counts =
                DeckCounts(decks + deck_base, catalog_.card_vocab_size);
            candidate.histories[perspective] = std::move(history);
          }
        }
        if (IsSuccess(raw.status)) {
          if (raw.error != kNoError || raw.perspective < 0 ||
              raw.perspective > 1) {
            throw std::invalid_argument(
                "successful rollout row has invalid status metadata");
          }
          PerspectiveHistory& history =
              candidate.histories[raw.perspective];
          if (!history.initialized) {
            throw std::invalid_argument(
                "rollout step references a perspective without a reset");
          }
          const VisibleEvidence visible = InspectVisible(raw);
          ApplyVisible(visible, &history);
          ApplyLogs(source, row, &history);
        } else if (source.log_offsets[row + 1] !=
                   source.log_offsets[row]) {
          throw std::invalid_argument(
              "failed rollout row exposed public log state");
        }
        candidate.current = std::move(raw);
        if (IsSuccess(candidate.current.status)) {
          const PreparedRow prepared =
              PrepareRow(candidate, candidate.current.perspective);
          ValidatePreparedRowForModel(prepared);
        }
        staged[row] = std::pair{slot, std::move(candidate)};
      });
  InvalidateModelPlan();
  for (auto& [slot, value] : staged) {
    slots_[slot] = std::move(value);
  }
}

RawRow RolloutEncoderCore::CopyRow(
    const CgTrainOutput& source, std::uint32_t row) const {
  RawRow result;
  result.initialized = true;
  result.status = source.status[row];
  result.error = source.error[row];
  result.perspective = source.select_player[row];
  result.select_type = source.select_type[row];
  result.select_context = source.select_context[row];
  result.select_min = source.select_min[row];
  result.select_max = source.select_max[row];
  result.result = source.result[row];
  result.turn = source.turn[row];
  result.turn_action_count = source.turn_action_count[row];
  result.first_player = source.first_player[row];
  result.turn_flags = source.turn_flags[row];
  result.remain_damage_counter = source.remain_damage_counter[row];
  result.remain_energy_cost = source.remain_energy_cost[row];
  result.deck_counts = {source.player0_deck_count[row],
                        source.player1_deck_count[row]};
  result.hand_counts = {source.player0_hand_count[row],
                        source.player1_hand_count[row]};
  result.prize_counts = {source.player0_prize_count[row],
                         source.player1_prize_count[row]};
  result.bench_max = {source.player0_bench_max[row],
                      source.player1_bench_max[row]};
  result.status_flags = {source.player0_status_flags[row],
                         source.player1_status_flags[row]};

  const std::uint32_t option_start = source.option_offsets[row];
  const std::uint32_t option_stop = source.option_offsets[row + 1];
  result.options.reserve(option_stop - option_start);
  for (std::uint32_t index = option_start; index < option_stop; ++index) {
    result.options.push_back(
        Option{.type = source.option_type[index],
               .params = {source.option_p0[index], source.option_p1[index],
                          source.option_p2[index], source.option_p3[index],
                          source.option_p4[index]}});
  }

  const std::uint32_t visible_start = source.visible_card_offsets[row];
  const std::uint32_t visible_stop = source.visible_card_offsets[row + 1];
  result.visible_cards.reserve(visible_stop - visible_start);
  for (std::uint32_t index = visible_start; index < visible_stop; ++index) {
    result.visible_cards.push_back(PublicCard{
        .owner = source.visible_card_owner[index],
        .area = source.visible_card_area[index],
        .area_index = source.visible_card_area_index[index],
        .card_id = source.visible_card_id[index],
        .serial = source.visible_card_serial[index],
        .hp = source.visible_card_hp[index],
        .max_hp = source.visible_card_max_hp[index],
        .appear_this_turn = source.visible_card_appear_this_turn[index],
    });
  }
  const auto local_reference = [&](std::uint32_t absolute,
                                   const char* name) -> std::uint32_t {
    if (absolute == kMissingRow) {
      return kMissingRow;
    }
    if (absolute < visible_start || absolute >= visible_stop) {
      throw std::invalid_argument(std::string("rollout ") + name +
                                  " crosses its source row");
    }
    return absolute - visible_start;
  };
  result.context_card_row =
      local_reference(source.context_card_row[row], "context card");
  result.effect_card_row =
      local_reference(source.effect_card_row[row], "effect card");

  const std::uint32_t attachment_start = source.attachment_offsets[row];
  const std::uint32_t attachment_stop = source.attachment_offsets[row + 1];
  result.attachments.reserve(attachment_stop - attachment_start);
  for (std::uint32_t index = attachment_start; index < attachment_stop;
       ++index) {
    const std::uint32_t parent = source.attachment_parent[index];
    if (parent < visible_start || parent >= visible_stop) {
      throw std::invalid_argument(
          "rollout attachment parent crosses its source row");
    }
    result.attachments.push_back(Attachment{
        .parent = parent - visible_start,
        .kind = source.attachment_kind[index],
        .card_id = source.attachment_card_id[index],
        .serial = source.attachment_card_serial[index],
        .energy_type = source.attachment_energy_type[index],
        .energy_units = source.attachment_energy_units[index],
    });
  }
  return result;
}

VisibleEvidence RolloutEncoderCore::InspectVisible(
    const RawRow& row) const {
  VisibleEvidence result;
  const std::int32_t opponent = 1 - row.perspective;
  for (const PublicCard& card : row.visible_cards) {
    if (card.card_id <= 0 || card.area == kAreaVirtual) {
      continue;
    }
    if (card.owner == row.perspective) {
      Increment(&result.own_counts, card.card_id);
    } else if (card.owner == opponent) {
      if (card.area == kAreaHand) {
        throw std::invalid_argument(
            "rollout public state exposed opponent hand identity");
      }
      if (card.area == kAreaDeck) {
        throw std::invalid_argument(
            "rollout public state exposed opponent deck identity");
      }
      Increment(&result.opponent_counts, card.card_id);
      if (card.serial > 0) {
        result.opponent_by_serial.emplace(card.serial, card.card_id);
      }
    }
  }
  for (const Attachment& attachment : row.attachments) {
    if (attachment.parent >= row.visible_cards.size()) {
      throw std::invalid_argument(
          "rollout attachment parent is outside visible cards");
    }
    if (attachment.card_id <= 0) {
      continue;
    }
    const std::int32_t owner = row.visible_cards[attachment.parent].owner;
    if (owner == row.perspective) {
      Increment(&result.own_counts, attachment.card_id);
    } else if (owner == opponent) {
      Increment(&result.opponent_counts, attachment.card_id);
      if (attachment.serial > 0) {
        result.opponent_by_serial.emplace(attachment.serial,
                                          attachment.card_id);
      }
    }
  }
  return result;
}

void RolloutEncoderCore::ApplyVisible(
    const VisibleEvidence& visible, PerspectiveHistory* history) const {
  for (const auto& [serial, card_id] : visible.opponent_by_serial) {
    const auto [unused, inserted] =
        history->revealed_by_serial.emplace(serial, card_id);
    if (inserted) {
      Increment(&history->revealed_counts, card_id);
    }
  }
}

void RolloutEncoderCore::ApplyLogs(
    const CgTrainOutput& source, std::uint32_t row,
    PerspectiveHistory* history) const {
  const std::uint32_t start = source.log_offsets[row];
  const std::uint32_t stop = source.log_offsets[row + 1];
  std::array<std::int32_t, 2> net_deltas{};
  for (std::uint32_t index = start; index < stop; ++index) {
    const std::int32_t type = source.log_type[index];
    if (type < 0 ||
        static_cast<std::size_t>(type) >= kExpectedLogParamCounts.size()) {
      throw std::invalid_argument("rollout public log type is unsupported");
    }
    const std::uint32_t expected = kExpectedLogParamCounts[type];
    if (source.log_param_count[index] != expected) {
      throw std::invalid_argument(
          "rollout public log parameter count is not canonical");
    }
    for (std::size_t column = expected; column < kLogParamWidth; ++column) {
      if (Param(source, column, index) != 0) {
        throw std::invalid_argument(
            "rollout public log exposed a nonzero unused parameter");
      }
    }
    const std::int32_t player = LogPlayer(source, index);
    if (player == 0 || player == 1) {
      net_deltas[player] += DeckDelta(source, index);
    }
  }
  std::array<std::int32_t, 2> running_counts = {
      std::max(0, source.player0_deck_count[row] - net_deltas[0]),
      std::max(0, source.player1_deck_count[row] - net_deltas[1]),
  };
  const std::int32_t opponent = 1 - history->perspective;
  for (std::uint32_t index = start; index < stop; ++index) {
    const std::int32_t player = LogPlayer(source, index);
    if (player == 0 || player == 1) {
      const std::int32_t delta = DeckDelta(source, index);
      UpdateDeckFlow(source, index, player, running_counts[player], delta,
                     history);
      running_counts[player] =
          std::max(0, running_counts[player] + delta);
      UpdateHistory(source, index, player, catalog_, history);
    }
    if (player == opponent && source.log_type[index] != kLogDrawReverse &&
        source.log_type[index] != kLogMoveCardReverse) {
      for (const auto& [card_id, serial] : RevealedPairs(source, index)) {
        RecordIdentity(card_id, serial, history);
      }
    }
  }
  history->observed_deck_counts = {
      std::max(0, source.player0_deck_count[row]),
      std::max(0, source.player1_deck_count[row]),
  };
  history->has_observed_deck_counts = {true, true};
}

void RolloutEncoderCore::ClearSlots(
    std::span<const std::uint32_t> slot_values) {
  std::set<std::uint32_t> unique;
  for (std::uint32_t slot : slot_values) {
    if (slot >= slot_capacity_ || !unique.insert(slot).second) {
      throw std::invalid_argument(
          "rollout clear slots must be unique and in range");
    }
  }
  InvalidateModelPlan();
  for (std::uint32_t slot : slot_values) {
    slots_[slot] = SlotState{};
  }
}

void RolloutEncoderCore::ValidateSelection(
    std::span<const std::uint32_t> slot_values,
    std::span<const std::int32_t> perspectives) const {
  if (slot_values.empty() || slot_values.size() != perspectives.size()) {
    throw std::invalid_argument(
        "rollout row selection must contain aligned non-empty vectors");
  }
  std::set<std::uint32_t> unique;
  for (std::size_t row = 0; row < slot_values.size(); ++row) {
    const std::uint32_t slot = slot_values[row];
    const std::int32_t perspective = perspectives[row];
    if (slot >= slot_capacity_ || !unique.insert(slot).second) {
      throw std::invalid_argument(
          "rollout selected slots must be unique and in range");
    }
    if (perspective < 0 || perspective > 1) {
      throw std::invalid_argument(
          "rollout selected perspective is invalid");
    }
    const SlotState& state = slots_[slot];
    if (!state.current.initialized ||
        state.current.perspective != perspective ||
        state.current.status != kReadyStatus ||
        state.current.error != kNoError ||
        !state.histories[perspective].initialized) {
      throw std::invalid_argument(
          "rollout selected slot is not a ready matching perspective");
    }
    if (state.current.options.empty()) {
      throw std::invalid_argument(
          "rollout selected slot has no legal options");
    }
  }
}

std::vector<PreparedRow> RolloutEncoderCore::PrepareRows(
    std::span<const std::uint32_t> slot_values,
    std::span<const std::int32_t> perspectives) {
  ValidateSelection(slot_values, perspectives);
  std::vector<PreparedRow> rows(slot_values.size());
  cg_train_internal::ParallelExecutor& executor =
      cg_train_internal::ParallelExecutor::Global();
  executor.ParallelFor(
      rows.size(), executor.worker_count(), [&](std::size_t row_index) {
        rows[row_index] = PrepareRow(
            slots_[slot_values[row_index]], perspectives[row_index]);
      });
  return rows;
}

void RolloutEncoderCore::InvalidateModelPlan() {
  prepared_plan_.reset();
}

PreparedRow RolloutEncoderCore::PrepareRow(
    const SlotState& slot, std::int32_t perspective) {
  const RawRow& raw = slot.current;
  const PerspectiveHistory& history = slot.histories[perspective];
  const VisibleEvidence visible = InspectVisible(raw);
  PreparedRow prepared;
  prepared.raw = &raw;
  prepared.history = &history;

  prepared.own_unseen.reserve(history.own_deck_counts.size());
  for (const auto& [card_id, count] : history.own_deck_counts) {
    const auto found = visible.own_counts.find(card_id);
    const std::int32_t visible_count =
        found == visible.own_counts.end() ? 0 : found->second;
    if (count > visible_count) {
      prepared.own_unseen.emplace_back(card_id, count - visible_count);
    }
  }

  std::set<std::int32_t> opponent_ids;
  for (const auto& [card_id, unused] : history.revealed_counts) {
    opponent_ids.insert(card_id);
  }
  for (const auto& [card_id, unused] : visible.opponent_counts) {
    opponent_ids.insert(card_id);
  }
  for (std::int32_t card_id : opponent_ids) {
    const auto tracked_it = history.revealed_counts.find(card_id);
    const auto visible_it = visible.opponent_counts.find(card_id);
    const std::int32_t tracked =
        tracked_it == history.revealed_counts.end() ? 0 : tracked_it->second;
    const std::int32_t current =
        visible_it == visible.opponent_counts.end() ? 0 : visible_it->second;
    if (tracked > current) {
      prepared.opponent_revealed.emplace_back(card_id, tracked - current);
    }
    const std::int32_t known = std::max(tracked, current);
    if (known > 0) {
      prepared.known.emplace_back(card_id, known);
    }
  }
  prepared.posterior = PosteriorFor(prepared.known);
  const std::uint64_t token_count =
      2 + raw.visible_cards.size() + prepared.own_unseen.size() +
      prepared.opponent_revealed.size();
  if (token_count > std::numeric_limits<std::uint16_t>::max()) {
    throw std::overflow_error(
        "rollout state token count exceeds uint16 pointer capacity");
  }
  prepared.state_token_count = static_cast<std::uint32_t>(token_count);
  prepared.attachment_count =
      static_cast<std::uint32_t>(raw.attachments.size());
  prepared.option_count = static_cast<std::uint32_t>(raw.options.size());
  prepared.deck_count =
      static_cast<std::uint32_t>(history.own_deck_counts.size());
  return prepared;
}

std::shared_ptr<const Posterior> RolloutEncoderCore::PosteriorFor(
    const CountRow& known) {
  const std::scoped_lock<std::mutex> lock(posterior_mutex_);
  auto found = posterior_cache_.find(known);
  if (found != posterior_cache_.end()) {
    posterior_lru_.splice(posterior_lru_.end(), posterior_lru_,
                          found->second.second);
    return found->second.first;
  }
  auto posterior = std::make_shared<const Posterior>(
      ComputePosterior(known));
  posterior_lru_.push_back(known);
  auto iterator = std::prev(posterior_lru_.end());
  posterior_cache_.emplace(known,
                           PosteriorCacheEntry{posterior, iterator});
  if (posterior_cache_.size() > catalog_.posterior_cache_capacity) {
    const CountRow evicted = posterior_lru_.front();
    posterior_lru_.pop_front();
    posterior_cache_.erase(evicted);
  }
  return posterior;
}

Posterior RolloutEncoderCore::ComputePosterior(
    const CountRow& known) const {
  const std::size_t deck_count = catalog_.deck_count;
  const std::size_t vocab = catalog_.card_vocab_size;
  const std::size_t stride = vocab + 1;
  std::int32_t known_total = 0;
  for (const auto& [card_id, count] : known) {
    if (card_id <= 0 || static_cast<std::size_t>(card_id) > vocab ||
        count <= 0) {
      throw std::invalid_argument(
          "rollout public evidence is outside the catalog vocabulary");
    }
    if (count > static_cast<std::int32_t>(kDeckSize) - known_total) {
      throw std::invalid_argument(
          "rollout public evidence exceeds an exact deck");
    }
    known_total += count;
  }

  std::vector<double> log_scores(deck_count);
  std::vector<bool> compatible(deck_count, true);
  std::size_t compatible_count = 0;
  for (std::size_t deck = 0; deck < deck_count; ++deck) {
    double score = catalog_.exact_log_priors[deck];
    for (const auto& [card_id, observed] : known) {
      const std::int32_t available =
          catalog_.entry_counts[deck * stride + card_id];
      if (available < observed) {
        compatible[deck] = false;
        break;
      }
      score +=
          catalog_.log_combinations[available * 61 + observed];
    }
    score -= catalog_.log_combinations[kDeckSize * 61 + known_total];
    log_scores[deck] =
        compatible[deck] ? score
                         : -std::numeric_limits<double>::infinity();
    if (compatible[deck]) {
      ++compatible_count;
    }
  }

  std::vector<double> probabilities(deck_count, 0.0);
  double unknown_probability = 1.0;
  if (compatible_count > 0) {
    double unknown_score = catalog_.unknown_log_prior +
                           catalog_.log_factorials[known_total];
    for (const auto& [card_id, observed] : known) {
      unknown_score -= catalog_.log_factorials[observed];
      unknown_score +=
          static_cast<double>(observed) *
          catalog_.unknown_log_card_probabilities[card_id - 1];
    }
    double maximum = unknown_score;
    for (std::size_t deck = 0; deck < deck_count; ++deck) {
      if (compatible[deck]) {
        maximum = std::max(maximum, log_scores[deck]);
      }
    }
    double total = std::exp(unknown_score - maximum);
    for (std::size_t deck = 0; deck < deck_count; ++deck) {
      if (compatible[deck]) {
        probabilities[deck] = std::exp(log_scores[deck] - maximum);
        total += probabilities[deck];
      }
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
      throw std::runtime_error(
          "rollout public catalog posterior normalization failed");
    }
    for (double& probability : probabilities) {
      probability /= total;
    }
    unknown_probability = std::exp(unknown_score - maximum) / total;
  }

  std::vector<double> expected(vocab + 1, 0.0);
  double exact_mass = 0.0;
  for (std::size_t deck = 0; deck < deck_count; ++deck) {
    const double probability = probabilities[deck];
    if (probability == 0.0) {
      continue;
    }
    exact_mass += probability;
    const std::int16_t* counts =
        catalog_.entry_counts.data() + deck * stride;
    for (std::size_t card_id = 1; card_id <= vocab; ++card_id) {
      expected[card_id] +=
          probability * static_cast<double>(counts[card_id]);
    }
  }
  for (const auto& [card_id, observed] : known) {
    expected[card_id] -= exact_mass * static_cast<double>(observed);
  }
  const double remaining =
      static_cast<double>(kDeckSize - known_total);
  for (std::size_t card_id = 1; card_id <= vocab; ++card_id) {
    expected[card_id] +=
        unknown_probability * remaining *
        catalog_.unknown_card_probabilities[card_id - 1];
  }

  double entropy = 0.0;
  for (double probability : probabilities) {
    if (probability > 0.0) {
      entropy -= probability * std::log(probability);
    }
  }
  if (unknown_probability > 0.0) {
    entropy -= unknown_probability * std::log(unknown_probability);
  }
  Posterior result;
  result.expected_counts.resize(vocab);
  result.expected_valid.resize(vocab);
  for (std::size_t card_id = 1; card_id <= vocab; ++card_id) {
    result.expected_valid[card_id - 1] =
        static_cast<std::uint8_t>(expected[card_id] > 0.0);
    result.expected_counts[card_id - 1] =
        static_cast<float>(std::max(0.0, expected[card_id]));
  }
  result.unknown_probability =
      static_cast<float>(unknown_probability);
  result.entropy = static_cast<float>(entropy);
  result.compatible_count =
      static_cast<std::int32_t>(compatible_count);
  result.known_total = known_total;
  return result;
}

std::uint32_t RolloutEncoderCore::PlanKnown(
    std::span<const std::uint32_t> slot_values,
    std::span<const std::int32_t> perspectives) {
  const std::vector<PreparedRow> rows =
      PrepareRows(slot_values, perspectives);
  std::uint64_t count = 0;
  for (const PreparedRow& row : rows) {
    count += row.known.size();
  }
  if (count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error(
        "rollout known evidence exceeds uint32 capacity");
  }
  return static_cast<std::uint32_t>(count);
}

void RolloutEncoderCore::WriteKnown(
    std::span<const std::uint32_t> slot_values,
    std::span<const std::int32_t> perspectives,
    std::uint32_t value_capacity, std::uint32_t* offsets,
    std::int32_t* card_ids, std::int32_t* counts) {
  RequirePointer(offsets, "known.offsets");
  const std::vector<PreparedRow> rows =
      PrepareRows(slot_values, perspectives);
  std::uint64_t required = 0;
  for (const PreparedRow& row : rows) {
    required += row.known.size();
  }
  if (required > value_capacity) {
    throw std::length_error(
        "rollout known output capacity is insufficient");
  }
  if (required > 0) {
    RequirePointer(card_ids, "known.card_ids");
    RequirePointer(counts, "known.counts");
  }
  std::uint32_t offset = 0;
  offsets[0] = 0;
  for (std::size_t row_index = 0; row_index < rows.size(); ++row_index) {
    for (const auto& [card_id, count] : rows[row_index].known) {
      card_ids[offset] = card_id;
      counts[offset] = count;
      ++offset;
    }
    offsets[row_index + 1] = offset;
  }
}

}  // namespace cg_train_rollout

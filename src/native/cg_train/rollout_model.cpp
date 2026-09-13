// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®,
// and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only

#include "rollout_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "parallel_executor.h"

namespace cg_train_rollout {
namespace {

constexpr std::int32_t kAreaVirtual = 0;
constexpr std::int32_t kAreaHand = 2;
constexpr std::int32_t kAreaActive = 4;
constexpr std::int32_t kAreaBench = 5;
constexpr std::int32_t kAreaStadium = 7;
constexpr std::int64_t kAreaOov = 13;
constexpr std::int64_t kOwnerUnknown = 0;
constexpr std::int64_t kOwnerSelf = 1;
constexpr std::int64_t kOwnerOpponent = 2;
constexpr std::int64_t kOwnerShared = 3;
constexpr std::int64_t kTokenKindOov = 15;
constexpr std::int64_t kEntitySlotOov = 17;

constexpr std::int32_t kOptionNumber = 0;
constexpr std::int32_t kOptionCard = 3;
constexpr std::int32_t kOptionToolCard = 4;
constexpr std::int32_t kOptionEnergyCard = 5;
constexpr std::int32_t kOptionEnergy = 6;
constexpr std::int32_t kOptionPlay = 7;
constexpr std::int32_t kOptionAttach = 8;
constexpr std::int32_t kOptionEvolve = 9;
constexpr std::int32_t kOptionAbility = 10;
constexpr std::int32_t kOptionDiscard = 11;
constexpr std::int32_t kOptionAttack = 13;
constexpr std::int32_t kOptionSkill = 15;
constexpr std::int32_t kOptionSpecialCondition = 16;

constexpr std::int32_t kAttachmentEnergy = 1;
constexpr std::int32_t kAttachmentTool = 2;
constexpr std::int32_t kAttachmentPreEvolution = 3;

using AreaKey =
    std::tuple<std::int32_t, std::int32_t, std::int32_t>;
using TokenIdentity = std::pair<std::int64_t, std::int32_t>;
using AttachmentKey =
    std::tuple<std::int64_t, std::int32_t, std::int32_t>;

template <typename T>
void RequirePointer(const T* pointer, const char* name) {
  if (pointer == nullptr) {
    throw std::invalid_argument(std::string("null rollout output pointer: ") +
                                name);
  }
}

float Clip(double value, double low = 0.0, double high = 1.0) {
  return static_cast<float>(std::clamp(value, low, high));
}

float Saturating(std::int32_t value, double scale) {
  const double normalized = std::max(0, value);
  return static_cast<float>(normalized / (normalized + scale));
}

float SignedSaturating(std::int32_t value, double scale) {
  const double magnitude = std::abs(static_cast<double>(value));
  if (magnitude == 0.0) {
    return 0.0F;
  }
  return static_cast<float>(
      std::copysign(magnitude / (magnitude + scale), value));
}

std::int32_t ZeroBasedCode(std::int32_t value) {
  return value > 0 ? value - 1 : 0;
}

std::size_t CheckedElements(std::size_t left, std::size_t right,
                            const char* name) {
  constexpr std::size_t kMaximum =
      static_cast<std::size_t>(
          std::numeric_limits<std::ptrdiff_t>::max());
  if (right != 0 && left > kMaximum / right) {
    throw std::length_error(std::string("rollout ") + name +
                            " tensor size overflows");
  }
  return left * right;
}

std::int64_t TokenKindForArea(std::int32_t area) {
  switch (area) {
    case 1:
      return 9;  // deck
    case 2:
      return 4;  // hand
    case 3:
      return 5;  // discard
    case 4:
      return 2;  // active
    case 5:
      return 3;  // bench
    case 6:
      return 6;  // prize
    case 7:
      return 7;  // stadium
    case 12:
      return 8;  // looking
    default:
      return kTokenKindOov;
  }
}

std::int32_t CanonicalOwner(std::int32_t area,
                            std::int32_t owner) {
  return area == kAreaStadium ? -1 : owner;
}

std::uint32_t PositiveBeliefWidth(const Posterior& posterior) {
  const std::size_t positive = static_cast<std::size_t>(std::count_if(
      posterior.expected_valid.begin(), posterior.expected_valid.end(),
      [](std::uint8_t value) { return value != 0; }));
  return static_cast<std::uint32_t>(std::max<std::size_t>(1, positive));
}

std::array<std::int32_t, kHistorySize> HistoryCounts(
    const PerspectiveHistory& history) {
  const std::int32_t own = history.perspective;
  const std::int32_t opponent = 1 - own;
  return {
      history.history[own][0],
      history.history[own][1],
      history.history[own][2],
      history.history[own][3],
      history.history[opponent][0],
      history.history[opponent][1],
      history.history[opponent][2],
      history.history[opponent][3],
  };
}

std::array<std::int32_t, kDeckFlowSize> DeckFlowCounts(
    const PerspectiveHistory& history) {
  std::array<std::int32_t, kDeckFlowSize> result{};
  std::size_t offset = 0;
  for (std::int32_t player :
       {history.perspective, 1 - history.perspective}) {
    const bool open = history.turn_open[player];
    result[offset++] = history.draw_counts[player];
    result[offset++] = history.return_counts[player];
    result[offset++] = open ? history.current_draw_counts[player]
                            : history.recent_draw_counts[player];
    result[offset++] = open ? history.current_return_counts[player]
                            : history.recent_return_counts[player];
    result[offset++] = open ? history.current_deck_deltas[player]
                            : history.recent_deck_deltas[player];
    result[offset++] = history.rebound_counts[player];
    result[offset++] = history.no_attack_turns[player];
  }
  return result;
}

void ValidateOutput(const CgTrainRolloutShape& shape,
                    std::uint32_t row_offset,
                    std::uint32_t belief_row_offset,
                    const CgTrainRolloutOutput& output) {
  if (output.struct_size != sizeof(CgTrainRolloutOutput)) {
    throw std::invalid_argument("rollout output struct size is incompatible");
  }
  if (row_offset > output.row_capacity ||
      shape.row_count > output.row_capacity - row_offset ||
      belief_row_offset > output.belief_row_capacity ||
      shape.belief_row_count >
          output.belief_row_capacity - belief_row_offset ||
      output.state_token_width < shape.state_token_width ||
      output.state_attachment_width < shape.state_attachment_width ||
      output.option_width < shape.option_width ||
      output.deck_width < shape.deck_width ||
      output.belief_width < shape.belief_width) {
    throw std::length_error("rollout model output capacity is insufficient");
  }
  const std::size_t state_elements =
      CheckedElements(output.row_capacity, output.state_token_width,
                      "state");
  CheckedElements(state_elements, kTokenScalarSize, "state scalar");
  CheckedElements(output.row_capacity, output.state_attachment_width,
                  "state attachment");
  const std::size_t option_elements =
      CheckedElements(output.row_capacity, output.option_width, "option");
  CheckedElements(option_elements, kMaximumEntitySlots, "option entity");
  CheckedElements(option_elements, kOptionScalarSize, "option scalar");
  CheckedElements(option_elements, kDynamicEffectSize,
                  "option dynamic effect");
  CheckedElements(output.row_capacity, output.deck_width, "deck");
  CheckedElements(output.belief_row_capacity, output.belief_width,
                  "belief");
  RequirePointer(output.state_card_ids, "state_card_ids");
  RequirePointer(output.state_areas, "state_areas");
  RequirePointer(output.state_owner_roles, "state_owner_roles");
  RequirePointer(output.state_token_kinds, "state_token_kinds");
  RequirePointer(output.state_scalars, "state_scalars");
  RequirePointer(output.state_last_attack_ids, "state_last_attack_ids");
  RequirePointer(output.state_padding_mask, "state_padding_mask");
  RequirePointer(output.state_attachment_card_ids,
                 "state_attachment_card_ids");
  RequirePointer(output.state_attachment_parent_indices,
                 "state_attachment_parent_indices");
  RequirePointer(output.state_attachment_kinds,
                 "state_attachment_kinds");
  RequirePointer(output.state_entity_slots, "state_entity_slots");
  RequirePointer(output.state_sequence_lengths, "state_sequence_lengths");
  RequirePointer(output.option_types, "option_types");
  RequirePointer(output.option_contexts, "option_contexts");
  RequirePointer(output.option_entity_slots, "option_entity_slots");
  RequirePointer(output.option_entity_slot_mask,
                 "option_entity_slot_mask");
  RequirePointer(output.option_attack_ids, "option_attack_ids");
  RequirePointer(output.option_card_ids, "option_card_ids");
  RequirePointer(output.option_scalars, "option_scalars");
  RequirePointer(output.option_dynamic_effect_features,
                 "option_dynamic_effect_features");
  RequirePointer(output.option_dynamic_effect_masks,
                 "option_dynamic_effect_masks");
  RequirePointer(output.option_valid, "option_valid");
  RequirePointer(output.option_min_counts, "option_min_counts");
  RequirePointer(output.option_max_counts, "option_max_counts");
  RequirePointer(output.option_lengths, "option_lengths");
  RequirePointer(output.option_maximum_counts, "option_maximum_counts");
  RequirePointer(output.deck_card_ids, "deck_card_ids");
  RequirePointer(output.deck_counts, "deck_counts");
  RequirePointer(output.deck_valid, "deck_valid");
  RequirePointer(output.belief_card_ids, "belief_card_ids");
  RequirePointer(output.belief_expected_counts,
                 "belief_expected_counts");
  RequirePointer(output.belief_valid, "belief_valid");
  RequirePointer(output.belief_scalars, "belief_scalars");
  RequirePointer(output.belief_row_indices, "belief_row_indices");
}

}  // namespace

void RolloutEncoderCore::ValidatePreparedRowForModel(
    const PreparedRow& prepared) const {
  if (prepared.raw == nullptr || prepared.history == nullptr ||
      prepared.posterior == nullptr) {
    throw std::invalid_argument("rollout prepared row is incomplete");
  }
  const RawRow& raw = *prepared.raw;
  if (raw.status == kReadyStatus && raw.options.empty()) {
    throw std::invalid_argument("rollout ready row has no legal options");
  }
  for (std::size_t visible_index = 0;
       visible_index < raw.visible_cards.size(); ++visible_index) {
    const PublicCard& card = raw.visible_cards[visible_index];
    if (card.area < 0 || card.area >= 16 || card.area_index < 0 ||
        card.area_index >= 256 || card.card_id < 0 ||
        static_cast<std::uint32_t>(card.card_id) >
            catalog_.card_vocab_size) {
      throw std::invalid_argument(
          "rollout public card exceeds the model schema");
    }
    const bool context = raw.context_card_row == visible_index;
    const bool effect = raw.effect_card_row == visible_index;
    if ((context &&
         (card.area != kAreaVirtual || card.area_index != 0)) ||
        (effect &&
         (card.area != kAreaVirtual || card.area_index != 1)) ||
        (card.area == kAreaVirtual && !context && !effect)) {
      throw std::invalid_argument(
          "rollout virtual card metadata is inconsistent");
    }
  }
  std::pair<std::uint32_t, std::int32_t> previous_group{
      kMissingRow, std::numeric_limits<std::int32_t>::min()};
  std::int32_t attachment_index = 0;
  for (const Attachment& attachment : raw.attachments) {
    if (attachment.parent >= raw.visible_cards.size() ||
        attachment.card_id <= 0 ||
        static_cast<std::uint32_t>(attachment.card_id) >
            catalog_.card_vocab_size ||
        attachment.card_id > std::numeric_limits<std::uint16_t>::max() ||
        attachment.kind < kAttachmentEnergy ||
        attachment.kind > kAttachmentPreEvolution) {
      throw std::invalid_argument(
          "rollout attachment identity exceeds the model schema");
    }
    const std::uint32_t parent_token = attachment.parent + 2;
    if (parent_token > std::numeric_limits<std::uint16_t>::max()) {
      throw std::invalid_argument(
          "rollout attachment parent exceeds the model schema");
    }
    const auto group = std::pair{attachment.parent, attachment.kind};
    if (group != previous_group) {
      previous_group = group;
      attachment_index = 0;
    } else {
      ++attachment_index;
    }
    if (attachment_index >= 256) {
      throw std::invalid_argument(
          "rollout attachment index exceeds the pointer schema");
    }
    if (attachment.kind == kAttachmentEnergy &&
        (attachment.energy_type < 0 || attachment.energy_type >= 12 ||
         attachment.energy_units < 0)) {
      throw std::invalid_argument(
          "rollout effective energy metadata is invalid");
    }
  }
  if (prepared.posterior->expected_counts.size() !=
          catalog_.card_vocab_size ||
      prepared.posterior->expected_valid.size() !=
          catalog_.card_vocab_size) {
    throw std::logic_error(
        "rollout posterior width differs from the catalog");
  }
}

PreparedPlan RolloutEncoderCore::PrepareModelPlan(
    std::span<const std::uint32_t> slot_values,
    std::span<const std::int32_t> perspectives) {
  PreparedPlan plan;
  plan.slots.assign(slot_values.begin(), slot_values.end());
  plan.perspectives.assign(perspectives.begin(), perspectives.end());
  plan.rows = PrepareRows(slot_values, perspectives);
  cg_train_internal::ParallelExecutor& executor =
      cg_train_internal::ParallelExecutor::Global();
  executor.ParallelFor(
      plan.rows.size(), executor.worker_count(),
      [&](std::size_t row_index) {
        ValidatePreparedRowForModel(plan.rows[row_index]);
      });

  CgTrainRolloutShape& shape = plan.shape;
  shape.struct_size = sizeof(CgTrainRolloutShape);
  shape.row_count = static_cast<std::uint32_t>(plan.rows.size());
  shape.state_attachment_width = 1;
  shape.belief_width = 1;
  std::map<CountRow, std::uint32_t> belief_rows;
  plan.row_beliefs.reserve(plan.rows.size());
  for (std::size_t row_index = 0; row_index < plan.rows.size();
       ++row_index) {
    const PreparedRow& row = plan.rows[row_index];
    shape.state_token_width =
        std::max(shape.state_token_width, row.state_token_count);
    shape.state_attachment_width =
        std::max(shape.state_attachment_width, row.attachment_count);
    shape.option_width = std::max(shape.option_width, row.option_count);
    shape.deck_width = std::max(shape.deck_width, row.deck_count);
    shape.belief_width =
        std::max(shape.belief_width, PositiveBeliefWidth(*row.posterior));
    const auto [iterator, inserted] = belief_rows.try_emplace(
        row.known, static_cast<std::uint32_t>(belief_rows.size()));
    if (inserted) {
      plan.unique_belief_rows.push_back(row_index);
    }
    plan.row_beliefs.push_back(iterator->second);
  }
  shape.belief_row_count =
      static_cast<std::uint32_t>(belief_rows.size());
  return plan;
}

const PreparedPlan& RolloutEncoderCore::ModelPlan(
    std::span<const std::uint32_t> slot_values,
    std::span<const std::int32_t> perspectives) {
  const bool matches =
      prepared_plan_.has_value() &&
      prepared_plan_->slots.size() == slot_values.size() &&
      prepared_plan_->perspectives.size() == perspectives.size() &&
      std::equal(
          slot_values.begin(), slot_values.end(),
          prepared_plan_->slots.begin()) &&
      std::equal(
          perspectives.begin(), perspectives.end(),
          prepared_plan_->perspectives.begin());
  if (!matches) {
    prepared_plan_ = PrepareModelPlan(slot_values, perspectives);
  }
  return *prepared_plan_;
}

CgTrainRolloutShape RolloutEncoderCore::PlanRows(
    std::span<const std::uint32_t> slot_values,
    std::span<const std::int32_t> perspectives) {
  prepared_plan_ = PrepareModelPlan(slot_values, perspectives);
  return prepared_plan_->shape;
}

void RolloutEncoderCore::EncodeRows(
    std::span<const std::uint32_t> slot_values,
    std::span<const std::int32_t> perspectives,
    std::uint32_t row_offset, std::uint32_t belief_row_offset,
    CgTrainRolloutOutput* output) {
  RequirePointer(output, "output");
  const PreparedPlan& plan = ModelPlan(slot_values, perspectives);
  const std::vector<PreparedRow>& rows = plan.rows;
  const std::vector<std::uint32_t>& row_beliefs = plan.row_beliefs;
  const std::vector<std::size_t>& unique_belief_rows =
      plan.unique_belief_rows;
  const CgTrainRolloutShape& shape = plan.shape;
  ValidateOutput(shape, row_offset, belief_row_offset, *output);

  const std::size_t token_width = output->state_token_width;
  const std::size_t attachment_width =
      output->state_attachment_width;
  const std::size_t option_width = output->option_width;
  const std::size_t deck_width = output->deck_width;
  const std::size_t belief_width = output->belief_width;
  cg_train_internal::ParallelExecutor& executor =
      cg_train_internal::ParallelExecutor::Global();

  executor.ParallelFor(
      rows.size(), executor.worker_count(), [&](std::size_t local_row) {
    const std::size_t destination_row = row_offset + local_row;
    const std::size_t state_base = destination_row * token_width;
    const std::size_t state_scalar_base =
        state_base * kTokenScalarSize;
    std::fill_n(output->state_card_ids + state_base, token_width, 0);
    std::fill_n(output->state_areas + state_base, token_width, 0);
    std::fill_n(output->state_owner_roles + state_base, token_width,
                kOwnerUnknown);
    std::fill_n(output->state_token_kinds + state_base, token_width,
                kTokenKindOov);
    std::fill_n(output->state_scalars + state_scalar_base,
                token_width * kTokenScalarSize, 0.0F);
    std::fill_n(output->state_last_attack_ids + state_base,
                token_width, 0);
    std::fill_n(output->state_padding_mask + state_base, token_width,
                static_cast<std::uint8_t>(1));
    std::fill_n(output->state_entity_slots + state_base, token_width,
                static_cast<std::uint8_t>(0));
    const std::size_t attachment_base =
        destination_row * attachment_width;
    std::fill_n(output->state_attachment_card_ids + attachment_base,
                attachment_width, static_cast<std::uint16_t>(0));
    std::fill_n(
        output->state_attachment_parent_indices + attachment_base,
        attachment_width, static_cast<std::uint16_t>(0));
    std::fill_n(output->state_attachment_kinds + attachment_base,
                attachment_width, static_cast<std::uint8_t>(0));

    const std::size_t option_base = destination_row * option_width;
    std::fill_n(output->option_types + option_base, option_width, 0);
    std::fill_n(output->option_contexts + option_base, option_width, 0);
    std::fill_n(output->option_entity_slots +
                    option_base * kMaximumEntitySlots,
                option_width * kMaximumEntitySlots, 0);
    std::fill_n(output->option_entity_slot_mask +
                    option_base * kMaximumEntitySlots,
                option_width * kMaximumEntitySlots,
                static_cast<std::uint8_t>(0));
    std::fill_n(output->option_attack_ids + option_base, option_width,
                0);
    std::fill_n(output->option_card_ids + option_base, option_width, 0);
    std::fill_n(output->option_scalars +
                    option_base * kOptionScalarSize,
                option_width * kOptionScalarSize, 0.0F);
    std::fill_n(output->option_dynamic_effect_features +
                    option_base * kDynamicEffectSize,
                option_width * kDynamicEffectSize, 0.0F);
    std::fill_n(output->option_dynamic_effect_masks + option_base,
                option_width, static_cast<std::uint8_t>(0));
    std::fill_n(output->option_valid + option_base, option_width,
                static_cast<std::uint8_t>(0));

    const std::size_t deck_base = destination_row * deck_width;
    std::fill_n(output->deck_card_ids + deck_base, deck_width, 0);
    std::fill_n(output->deck_counts + deck_base, deck_width, 0.0F);
    std::fill_n(output->deck_valid + deck_base, deck_width,
                static_cast<std::uint8_t>(0));
      });
  executor.ParallelFor(
      unique_belief_rows.size(), executor.worker_count(),
      [&](std::size_t local) {
    const std::size_t destination = belief_row_offset + local;
    const std::size_t belief_base = destination * belief_width;
    std::fill_n(output->belief_card_ids + belief_base, belief_width, 0);
    std::fill_n(output->belief_expected_counts + belief_base,
                belief_width, 0.0F);
    std::fill_n(output->belief_valid + belief_base, belief_width,
                static_cast<std::uint8_t>(0));
    std::fill_n(output->belief_scalars +
                    destination * kBeliefScalarSize,
                kBeliefScalarSize, 0.0F);
      });

  executor.ParallelFor(
      rows.size(), executor.worker_count(), [&](std::size_t local_row) {
    const PreparedRow& prepared = rows[local_row];
    const RawRow& raw = *prepared.raw;
    const PerspectiveHistory& history = *prepared.history;
    const std::size_t destination_row = row_offset + local_row;
    const std::size_t state_base = destination_row * token_width;
    const auto scalar = [&](std::size_t token,
                            std::size_t feature) -> float& {
      return output->state_scalars[
          (state_base + token) * kTokenScalarSize + feature];
    };

    output->state_padding_mask[state_base] = 0;
    output->state_padding_mask[state_base + 1] = 0;
    output->state_owner_roles[state_base] = kOwnerShared;
    output->state_owner_roles[state_base + 1] = kOwnerShared;
    output->state_token_kinds[state_base] = 0;
    output->state_token_kinds[state_base + 1] = 1;
    output->state_sequence_lengths[destination_row] =
        prepared.state_token_count;

    const auto flow = DeckFlowCounts(history);
    for (std::size_t player_offset : {std::size_t{0}, std::size_t{7}}) {
      scalar(0, player_offset) =
          Saturating(flow[player_offset], 40.0);
      scalar(0, player_offset + 1) =
          Saturating(flow[player_offset + 1], 20.0);
      scalar(0, player_offset + 2) =
          Saturating(flow[player_offset + 2], 4.0);
      scalar(0, player_offset + 3) =
          Saturating(flow[player_offset + 3], 4.0);
      scalar(0, player_offset + 4) =
          SignedSaturating(flow[player_offset + 4], 4.0);
      scalar(0, player_offset + 5) =
          Saturating(flow[player_offset + 5], 10.0);
      scalar(0, player_offset + 6) =
          Saturating(flow[player_offset + 6], 10.0);
    }
    scalar(0, 14) = Saturating(raw.turn, 100.0);
    const auto history_counts = HistoryCounts(history);
    for (std::size_t index = 0; index < history_counts.size(); ++index) {
      scalar(0, 15 + index) =
          Saturating(history_counts[index], 20.0);
    }
    const std::int32_t perspective = raw.perspective;
    const std::int32_t opponent = 1 - perspective;
    scalar(0, 24) = Clip(static_cast<double>(raw.turn) / 100.0);
    scalar(0, 25) =
        Clip(static_cast<double>(raw.turn_action_count) / 20.0);
    scalar(0, 26) = raw.first_player == perspective ? 1.0F : 0.0F;
    for (std::size_t bit = 0; bit < 4; ++bit) {
      scalar(0, 27 + bit) =
          static_cast<float>((raw.turn_flags >> bit) & 1U);
    }
    scalar(0, 31) = static_cast<float>(raw.result) / 2.0F;
    scalar(0, 32) =
        Clip(static_cast<double>(raw.prize_counts[perspective]) / 6.0);
    scalar(0, 33) =
        Clip(static_cast<double>(raw.prize_counts[opponent]) / 6.0);
    scalar(0, 34) =
        Clip(static_cast<double>(raw.deck_counts[perspective]) / 60.0);
    scalar(0, 35) =
        Clip(static_cast<double>(raw.deck_counts[opponent]) / 60.0);
    scalar(0, 36) =
        Clip(static_cast<double>(raw.hand_counts[perspective]) / 20.0);
    scalar(0, 37) =
        Clip(static_cast<double>(raw.hand_counts[opponent]) / 20.0);
    scalar(0, 38) =
        Clip(static_cast<double>(raw.select_min) / 8.0);
    scalar(0, 39) =
        Clip(static_cast<double>(raw.select_max) / 8.0);
    scalar(0, 40) =
        Clip(static_cast<double>(raw.remain_damage_counter) / 50.0);
    scalar(0, 41) =
        Clip(static_cast<double>(raw.remain_energy_cost) / 12.0);
    scalar(0, 42) =
        Clip(static_cast<double>(ZeroBasedCode(raw.select_type)) / 64.0);
    scalar(0, 43) =
        Clip(static_cast<double>(ZeroBasedCode(raw.select_context)) / 64.0);
    for (std::size_t index = 0; index < history_counts.size(); ++index) {
      scalar(0, 44 + index) =
          Clip(static_cast<double>(history_counts[index]) / 40.0);
    }
    scalar(0, 53) = 0.0F;
    scalar(0, 54) = 1.0F;
    scalar(0, 55) =
        static_cast<float>(raw.bench_max[perspective]) / 8.0F;
    scalar(0, 56) = 1.0F;
    scalar(0, 57) =
        static_cast<float>(raw.bench_max[opponent]) / 8.0F;
    scalar(0, 58) = 1.0F;

    std::map<AreaKey, std::int64_t> area_tokens;
    std::map<std::int32_t, TokenIdentity> any_serial_tokens;
    std::map<std::int32_t, TokenIdentity> own_serial_tokens;
    for (std::size_t visible_index = 0;
         visible_index < raw.visible_cards.size(); ++visible_index) {
      const PublicCard& card = raw.visible_cards[visible_index];
      if (card.area < 0 || card.area >= 16 || card.area_index < 0 ||
          card.area_index >= 256) {
        throw std::invalid_argument(
            "rollout public card pointer exceeds model schema");
      }
      const std::size_t token = visible_index + 2;
      output->state_padding_mask[state_base + token] = 0;
      output->state_card_ids[state_base + token] =
          std::max(0, card.card_id);
      output->state_areas[state_base + token] =
          card.area >= 0 && card.area < kAreaOov ? card.area : kAreaOov;
      if (card.owner == perspective) {
        output->state_owner_roles[state_base + token] = kOwnerSelf;
        scalar(token, 3) = 1.0F;
      } else if (card.owner == 0 || card.owner == 1) {
        output->state_owner_roles[state_base + token] = kOwnerOpponent;
        scalar(token, 3) = -1.0F;
      } else if (card.owner < 0) {
        output->state_owner_roles[state_base + token] = kOwnerShared;
      } else {
        output->state_owner_roles[state_base + token] = kOwnerUnknown;
      }
      std::int64_t token_kind = TokenKindForArea(card.area);
      if (raw.context_card_row == visible_index) {
        if (card.area != kAreaVirtual || card.area_index != 0) {
          throw std::invalid_argument(
              "rollout context card metadata is inconsistent");
        }
        token_kind = 10;
      }
      if (raw.effect_card_row == visible_index) {
        if (card.area != kAreaVirtual || card.area_index != 1) {
          throw std::invalid_argument(
              "rollout effect card metadata is inconsistent");
        }
        token_kind = 11;
      }
      if (card.area == kAreaVirtual && token_kind == kTokenKindOov) {
        throw std::invalid_argument(
            "rollout virtual card is not context or effect");
      }
      output->state_token_kinds[state_base + token] = token_kind;

      const bool pokemon =
          (card.area == kAreaActive || card.area == kAreaBench) &&
          card.card_id > 0;
      if (pokemon) {
        const double hp = std::max(0, card.hp);
        const double max_hp = std::max(1, card.max_hp);
        scalar(token, 0) = Clip(hp / max_hp);
        scalar(token, 1) = Clip((max_hp - hp) / max_hp);
        scalar(token, 2) = Clip(max_hp / 400.0);
        scalar(token, 22) =
            card.appear_this_turn != 0 ? 1.0F : 0.0F;
        output->state_entity_slots[state_base + token] =
            static_cast<std::uint8_t>(std::min<std::int64_t>(
                kEntitySlotOov, std::max(0, card.area_index + 1)));
        if (card.area == kAreaActive &&
            (card.owner == 0 || card.owner == 1)) {
          const std::uint32_t flags = raw.status_flags[card.owner];
          for (std::size_t bit = 0; bit < 5; ++bit) {
            scalar(token, 17 + bit) =
                static_cast<float>((flags >> bit) & 1U);
          }
        }
      }
      const AreaKey area_key(
          card.area, CanonicalOwner(card.area, card.owner),
          card.area_index);
      area_tokens.try_emplace(area_key,
                              static_cast<std::int64_t>(token));
      if (card.serial > 0) {
        any_serial_tokens.try_emplace(
            card.serial,
            TokenIdentity{static_cast<std::int64_t>(token), card.card_id});
        if (card.owner == perspective) {
          own_serial_tokens.try_emplace(
              card.serial,
              TokenIdentity{static_cast<std::int64_t>(token),
                            card.card_id});
        }
      }
      if (pokemon && card.serial > 0) {
        const auto attack =
            history.last_attack_by_serial.find(card.serial);
        if (attack != history.last_attack_by_serial.end()) {
          output->state_last_attack_ids[state_base + token] =
              std::max(0, attack->second);
        }
      }
    }

    std::size_t context_token = 2 + raw.visible_cards.size();
    for (const auto& [card_id, count] : prepared.own_unseen) {
      output->state_padding_mask[state_base + context_token] = 0;
      output->state_card_ids[state_base + context_token] = card_id;
      output->state_owner_roles[state_base + context_token] = kOwnerSelf;
      output->state_token_kinds[state_base + context_token] = 12;
      scalar(context_token, 3) = 1.0F;
      scalar(context_token, 52) =
          Clip(static_cast<double>(count) / 60.0);
      ++context_token;
    }
    for (const auto& [card_id, count] :
         prepared.opponent_revealed) {
      output->state_padding_mask[state_base + context_token] = 0;
      output->state_card_ids[state_base + context_token] = card_id;
      output->state_owner_roles[state_base + context_token] =
          kOwnerOpponent;
      output->state_token_kinds[state_base + context_token] = 13;
      scalar(context_token, 3) = -1.0F;
      scalar(context_token, 52) =
          Clip(static_cast<double>(count) / 60.0);
      ++context_token;
    }

    std::map<AttachmentKey, TokenIdentity> attachment_tokens;
    std::pair<std::uint32_t, std::int32_t> previous_group{
        kMissingRow, std::numeric_limits<std::int32_t>::min()};
    std::int32_t attachment_index = 0;
    for (std::size_t local = 0; local < raw.attachments.size(); ++local) {
      const Attachment& attachment = raw.attachments[local];
      if (attachment.parent >= raw.visible_cards.size() ||
          attachment.card_id <= 0 ||
          attachment.card_id >
              std::numeric_limits<std::uint16_t>::max() ||
          attachment.kind < kAttachmentEnergy ||
          attachment.kind > kAttachmentPreEvolution) {
        throw std::invalid_argument(
            "rollout attachment identity exceeds model schema");
      }
      const std::uint32_t parent_token = attachment.parent + 2;
      if (parent_token > std::numeric_limits<std::uint16_t>::max()) {
        throw std::invalid_argument(
            "rollout attachment parent exceeds model schema");
      }
      output->state_attachment_card_ids[
          destination_row * attachment_width + local] =
          static_cast<std::uint16_t>(attachment.card_id);
      output->state_attachment_parent_indices[
          destination_row * attachment_width + local] =
          static_cast<std::uint16_t>(parent_token);
      output->state_attachment_kinds[
          destination_row * attachment_width + local] =
          static_cast<std::uint8_t>(attachment.kind);

      const auto group =
          std::pair{attachment.parent, attachment.kind};
      if (group != previous_group) {
        previous_group = group;
        attachment_index = 0;
      } else {
        ++attachment_index;
      }
      if (attachment_index >= 256) {
        throw std::invalid_argument(
            "rollout attachment index exceeds pointer schema");
      }
      attachment_tokens.try_emplace(
          AttachmentKey{parent_token, attachment.kind,
                        attachment_index},
          TokenIdentity{attachment.card_id, attachment.serial});

      if (attachment.kind == kAttachmentEnergy) {
        if (attachment.energy_type < 0 ||
            attachment.energy_type >= 12 ||
            attachment.energy_units < 0) {
          throw std::invalid_argument(
              "rollout effective energy metadata is invalid");
        }
        scalar(parent_token, 4 + attachment.energy_type) +=
            static_cast<float>(attachment.energy_units) / 8.0F;
      } else if (attachment.kind == kAttachmentTool) {
        scalar(parent_token, 16) += 0.25F;
      } else {
        scalar(parent_token, 23) += 1.0F / 3.0F;
      }
    }
    for (std::size_t token = 2;
         token < 2 + raw.visible_cards.size(); ++token) {
      for (std::size_t feature = 4; feature < 17; ++feature) {
        scalar(token, feature) = Clip(scalar(token, feature));
      }
      scalar(token, 23) = Clip(scalar(token, 23));
    }

    const std::size_t option_base = destination_row * option_width;
    const auto resolve_area =
        [&](std::int32_t area, std::int32_t owner,
            std::int32_t index) -> std::int64_t {
      if (owner < 0) {
        owner = perspective;
      }
      if (area < 0 || area >= 16 || index < 0 || index >= 256 ||
          (!((owner == 0 || owner == 1)) &&
           area != kAreaStadium)) {
        return -1;
      }
      const auto found = area_tokens.find(
          AreaKey{area, CanonicalOwner(area, owner), index});
      return found == area_tokens.end() ? -1 : found->second;
    };
    for (std::size_t local = 0; local < raw.options.size(); ++local) {
      const Option& option = raw.options[local];
      const std::size_t target = option_base + local;
      output->option_types[target] = option.type;
      output->option_contexts[target] =
          ZeroBasedCode(raw.select_context);
      output->option_valid[target] = 1;
      float* option_scalars =
          output->option_scalars + target * kOptionScalarSize;
      if (option.type == kOptionNumber) {
        option_scalars[0] =
            static_cast<float>(option.params[0]) / 10.0F;
      }
      if (option.type == kOptionEnergy) {
        option_scalars[1] =
            static_cast<float>(option.params[4]) / 10.0F;
      }
      if (option.type == kOptionEnergy ||
          option.type == kOptionEnergyCard) {
        option_scalars[2] =
            static_cast<float>(option.params[3]) / 16.0F;
        option_scalars[5] = 1.0F;
      }
      if (option.type == kOptionToolCard) {
        option_scalars[3] =
            static_cast<float>(option.params[3]) / 8.0F;
        option_scalars[6] = 1.0F;
      }
      if (option.type == kOptionSpecialCondition) {
        option_scalars[4] =
            static_cast<float>(option.params[0]) / 4.0F;
      }
      if (option.type == kOptionAttack) {
        output->option_attack_ids[target] =
            std::max(0, option.params[0]);
      }
      if (option.type == kOptionSkill) {
        output->option_card_ids[target] =
            std::max(0, option.params[0]);
      }

      std::int64_t primary = -1;
      std::int64_t secondary = -1;
      if (option.type == kOptionCard ||
          option.type == kOptionToolCard ||
          option.type == kOptionEnergyCard ||
          option.type == kOptionEnergy) {
        primary = resolve_area(option.params[0], option.params[2],
                               option.params[1]);
      } else if (option.type == kOptionPlay) {
        primary = resolve_area(kAreaHand, perspective, option.params[0]);
      } else if (option.type == kOptionAttach ||
                 option.type == kOptionEvolve) {
        primary = resolve_area(option.params[0], perspective,
                               option.params[1]);
        secondary = resolve_area(option.params[2], perspective,
                                 option.params[3]);
      } else if (option.type == kOptionAbility ||
                 option.type == kOptionDiscard) {
        primary = resolve_area(option.params[0], perspective,
                               option.params[1]);
      } else if (option.type == kOptionAttack) {
        primary = resolve_area(kAreaActive, perspective, 0);
      } else if (option.type == kOptionSkill) {
        if (option.params[0] == 0) {
          primary = 1;
        } else if (option.params[0] > 0 && option.params[1] > 0) {
          const auto own = own_serial_tokens.find(option.params[1]);
          if (own != own_serial_tokens.end()) {
            if (own->second.second == 0 ||
                own->second.second == option.params[0]) {
              primary = own->second.first;
            }
          } else {
            const auto any =
                any_serial_tokens.find(option.params[1]);
            if (any != any_serial_tokens.end() &&
                (any->second.second == 0 ||
                 any->second.second == option.params[0])) {
              primary = any->second.first;
            }
          }
        }
      }

      std::int32_t attachment_kind = 0;
      if (option.type == kOptionToolCard) {
        attachment_kind = kAttachmentTool;
      } else if (option.type == kOptionEnergyCard ||
                 option.type == kOptionEnergy) {
        attachment_kind = kAttachmentEnergy;
      }
      if (attachment_kind != 0 && primary >= 0 &&
          option.params[3] >= 0 && option.params[3] < 256) {
        const auto attachment = attachment_tokens.find(
            AttachmentKey{primary, attachment_kind, option.params[3]});
        if (attachment != attachment_tokens.end()) {
          output->option_card_ids[target] =
              attachment->second.first;
          option_scalars[7] =
              static_cast<float>(attachment->second.second) / 128.0F;
          option_scalars[8] = 1.0F;
        }
      }

      std::size_t entity_slot = target * kMaximumEntitySlots;
      if (primary >= 0) {
        output->option_entity_slots[entity_slot] = primary;
        output->option_entity_slot_mask[entity_slot] = 1;
        if (secondary >= 0) {
          output->option_entity_slots[entity_slot + 1] = secondary;
          output->option_entity_slot_mask[entity_slot + 1] = 1;
        }
      } else if (secondary >= 0) {
        output->option_entity_slots[entity_slot] = secondary;
        output->option_entity_slot_mask[entity_slot] = 1;
      }
    }
    const std::int64_t option_length =
        static_cast<std::int64_t>(raw.options.size());
    const std::int64_t minimum =
        std::min(option_length,
                 static_cast<std::int64_t>(std::max(0, raw.select_min)));
    const std::int64_t maximum = std::min(
        option_length,
        std::max(minimum,
                 static_cast<std::int64_t>(raw.select_max)));
    output->option_min_counts[destination_row] = minimum;
    output->option_max_counts[destination_row] = maximum;
    output->option_lengths[destination_row] =
        static_cast<std::uint32_t>(option_length);
    output->option_maximum_counts[destination_row] =
        static_cast<std::uint32_t>(maximum);

    const std::size_t deck_base = destination_row * deck_width;
    for (std::size_t local = 0;
         local < history.own_deck_counts.size(); ++local) {
      output->deck_card_ids[deck_base + local] =
          history.own_deck_counts[local].first;
      output->deck_counts[deck_base + local] =
          static_cast<float>(history.own_deck_counts[local].second);
      output->deck_valid[deck_base + local] = 1;
    }

    const std::uint32_t local_belief = row_beliefs[local_row];
    output->belief_row_indices[destination_row] =
        static_cast<std::int64_t>(belief_row_offset + local_belief);
      });

  executor.ParallelFor(
      unique_belief_rows.size(), executor.worker_count(),
      [&](std::size_t local) {
    const PreparedRow& prepared = rows[unique_belief_rows[local]];
    const Posterior& posterior = *prepared.posterior;
    const std::size_t destination = belief_row_offset + local;
    const std::size_t base = destination * belief_width;
    std::size_t local_card = 0;
    for (std::size_t card = 0;
         card < posterior.expected_counts.size(); ++card) {
      if (posterior.expected_valid[card] == 0) {
        continue;
      }
      output->belief_card_ids[base + local_card] =
          static_cast<std::int64_t>(card + 1);
      output->belief_expected_counts[base + local_card] =
          posterior.expected_counts[card];
      output->belief_valid[base + local_card] = 1;
      ++local_card;
    }
    float* scalars =
        output->belief_scalars + destination * kBeliefScalarSize;
    scalars[0] = posterior.entropy;
    scalars[1] = static_cast<float>(posterior.compatible_count);
    scalars[2] = static_cast<float>(posterior.known_total);
    scalars[3] = posterior.unknown_probability;
      });
}

}  // namespace cg_train_rollout

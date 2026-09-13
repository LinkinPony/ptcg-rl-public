// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®,
// and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use
// only; the full license is in data/ptcg_engine/ptcgProgram 22/LICENSES/.

#include "cg_train.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "All.h"
#include "parallel_executor.h"
#include "public_log.h"

namespace {

using cg_train_internal::ProjectPublicLog;
using cg_train_internal::PublicLog;
using cg_train_internal::ParallelExecutor;
using cg_train_internal::ResolveParallelWorkerCount;

constexpr std::uint32_t kAbiMagic = UINT32_C(0x31544743);  // "CGT1".
constexpr std::uint32_t kAbiVersion = 4;
constexpr std::uint32_t kMaxLaneCapacity = UINT32_C(1048576);
static_assert(
    offsetof(CgTrainOutput, selection_advance_count) +
            sizeof(CgTrainOutput::selection_advance_count) ==
        sizeof(CgTrainOutput),
    "ABI v4 transition counter must remain the final output field");

thread_local std::string g_last_error;
std::once_flag g_initialize_once;

struct Slot {
  std::unique_ptr<BattleData> data;
};

struct Row {
  BattleData* data = nullptr;
  std::int32_t status = CG_TRAIN_SLOT_UNINITIALIZED;
  std::int32_t error = CG_TRAIN_SLOT_ERROR_NONE;
  std::uint32_t selection_advance_count = 0;
};

struct PendingSlot {
  std::uint32_t slot = 0;
  std::unique_ptr<BattleData> data;
  Row row;
  bool commit = false;
};

struct PublicCounts {
  std::uint64_t visible_cards = 0;
  std::uint64_t attachments = 0;
};

struct RowOutputCounts {
  std::uint64_t options = 0;
  std::uint64_t visible_cards = 0;
  std::uint64_t attachments = 0;
  std::uint64_t logs = 0;
};

struct OutputPlan {
  std::vector<std::uint32_t> option_offsets;
  std::vector<std::uint32_t> visible_card_offsets;
  std::vector<std::uint32_t> attachment_offsets;
  std::vector<std::uint32_t> log_offsets;
};

constexpr CgTrainAbiDescriptor kAbiDescriptor = {
    .magic = kAbiMagic,
    .abi_version = kAbiVersion,
    .descriptor_size = sizeof(CgTrainAbiDescriptor),
    .output_size = sizeof(CgTrainOutput),
    .deck_size = DECK_SIZE,
    .players_per_game = 2,
    .option_param_count = 5,
    .option_type_count =
        static_cast<std::uint32_t>(SelectOptionType::SpecialCondition) + 1,
    .max_lane_capacity = kMaxLaneCapacity,
    .reserved = 0,
    .features = CG_TRAIN_FEATURE_EXPLICIT_SEEDS |
                CG_TRAIN_FEATURE_CSR_ACTIONS |
                CG_TRAIN_FEATURE_CALLER_OWNED_SOA |
                CG_TRAIN_FEATURE_STABLE_SLOT_ADDRESS |
                CG_TRAIN_FEATURE_FORCED_CHAIN |
                CG_TRAIN_FEATURE_NO_JSON |
                CG_TRAIN_FEATURE_ATOMIC_CAPACITY_FAILURE |
                CG_TRAIN_FEATURE_PUBLIC_CURRENT_STATE |
                CG_TRAIN_FEATURE_VISIBLE_CARD_CSR |
                CG_TRAIN_FEATURE_ATTACHMENT_CSR |
                CG_TRAIN_FEATURE_PUBLIC_LOG_DELTA |
                CG_TRAIN_FEATURE_SELECTION_ADVANCE_COUNT |
                CG_TRAIN_FEATURE_PARALLEL_ROWS |
                CG_TRAIN_FEATURE_PUBLIC_STATE_TOKEN_EXPORT |
                CG_TRAIN_FEATURE_PUBLIC_OBSERVATION_EXPORT |
                CG_TRAIN_FEATURE_SLOT_RELEASE,
};

void SetLastError(const std::string& message) { g_last_error = message; }

std::int32_t Fail(std::int32_t code, const std::string& message) {
  SetLastError(message);
  return code;
}

void InitializeEngine() {
  std::call_once(g_initialize_once, [] { InitializeAll(); });
}

void RebindState(BattleData* data) {
  data->state.game = &data->game;
  data->game.pushResponseFunc = {};
}

std::int32_t ReadyStatus(const BattleData& data) {
  return data.state.isFinish() ? CG_TRAIN_SLOT_FINISHED
                               : CG_TRAIN_SLOT_READY;
}

std::uint32_t AdvanceForcedChain(BattleData* data) {
  std::uint32_t advances = 0;
  while (!data->state.isFinish() && data->state.selectMax == 0) {
    data->state.selected.clear();
    data->next();
    if (advances == std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("forced selection advance count overflow");
    }
    ++advances;
  }
  return advances;
}

std::int32_t ValidateDeck(const std::int32_t* cards) {
  std::unordered_map<std::u8string, int> name_counts;
  bool has_basic = false;
  bool has_ace_spec = false;
  for (int index = 0; index < DECK_SIZE; ++index) {
    const CardId card_id = cards[index];
    const auto card_it = CardTable.find(card_id);
    if (card_it == CardTable.end()) {
      return CG_TRAIN_SLOT_ERROR_UNKNOWN_CARD;
    }
    const CardMaster& master = card_it->second;
    if (master.aceSpec) {
      if (has_ace_spec) {
        return CG_TRAIN_SLOT_ERROR_TOO_MANY_ACE_SPEC;
      }
      has_ace_spec = true;
    }
    if (master.cardType == CardType::Pokemon &&
        master.evolutionType == EvolutionType::Basic) {
      has_basic = true;
    }
    int& count = name_counts[master.name];
    ++count;
    if (count > DECK_SAME_CARD_MAX &&
        master.cardType != CardType::BasicEnergy) {
      return CG_TRAIN_SLOT_ERROR_TOO_MANY_COPIES;
    }
  }
  if (!has_basic) {
    return CG_TRAIN_SLOT_ERROR_NO_BASIC_POKEMON;
  }
  return CG_TRAIN_SLOT_ERROR_NONE;
}

std::int32_t ValidateDeckPair(const std::int32_t* decks) {
  for (int player = 0; player < 2; ++player) {
    const std::int32_t error =
        ValidateDeck(decks + player * static_cast<std::ptrdiff_t>(DECK_SIZE));
    if (error != CG_TRAIN_SLOT_ERROR_NONE) {
      return error;
    }
  }
  return CG_TRAIN_SLOT_ERROR_NONE;
}

struct NewBattleResult {
  std::unique_ptr<BattleData> data;
  std::uint32_t selection_advance_count = 0;
};

NewBattleResult NewBattle(
    const std::int32_t* decks, std::uint32_t seed) {
  GameConfig config = {};
  config.seed = seed;
  config.recordLog = true;
  config.manualCoin = false;
  config.sendDeck = false;
  config.deviceRand = false;
  for (int player = 0; player < 2; ++player) {
    for (int card = 0; card < DECK_SIZE; ++card) {
      const std::ptrdiff_t index =
          player * static_cast<std::ptrdiff_t>(DECK_SIZE) + card;
      config.decks[player].cards[card] = decks[index];
    }
  }

  auto data = std::make_unique<BattleData>();
  data->init(config, false);

  // Game::init treats zero as "choose a random seed". Reset the configured
  // seed and RNG before any game action so every uint32 seed, including zero,
  // remains an explicit deterministic input.
  data->game.config.seed = seed;
  data->game.rng = std::mt19937(seed);
  data->start();
  data->next();
  const std::uint32_t selection_advance_count =
      AdvanceForcedChain(data.get());
  return {
      .data = std::move(data),
      .selection_advance_count = selection_advance_count,
  };
}

std::unique_ptr<BattleData> CloneBattle(const BattleData& source) {
  auto clone = std::make_unique<BattleData>(source);
  RebindState(clone.get());
  return clone;
}

bool HasUniqueValidSlots(
    const CgTrainLane* lane, std::uint32_t batch_count,
    const std::uint32_t* slots, std::string* error);

std::uint32_t TurnFlags(const State& state) {
  std::uint32_t flags = 0;
  if (state.supporterPlayed) {
    flags |= CG_TRAIN_TURN_SUPPORTER_PLAYED;
  }
  if (state.stadiumPlayed) {
    flags |= CG_TRAIN_TURN_STADIUM_PLAYED;
  }
  if (state.energyPlayed) {
    flags |= CG_TRAIN_TURN_ENERGY_ATTACHED;
  }
  if (state.retreated) {
    flags |= CG_TRAIN_TURN_RETREATED;
  }
  return flags;
}

std::uint32_t PlayerStatusFlags(const PlayerState& player) {
  std::uint32_t flags = 0;
  if (player.isPoisoned()) {
    flags |= CG_TRAIN_PLAYER_POISONED;
  }
  if (player.burned) {
    flags |= CG_TRAIN_PLAYER_BURNED;
  }
  if (player.badStatus == BadStatusType::Asleep) {
    flags |= CG_TRAIN_PLAYER_ASLEEP;
  }
  if (player.badStatus == BadStatusType::Paralyzed) {
    flags |= CG_TRAIN_PLAYER_PARALYZED;
  }
  if (player.badStatus == BadStatusType::Confused) {
    flags |= CG_TRAIN_PLAYER_CONFUSED;
  }
  return flags;
}

std::int32_t LookingMode(const State& state, int perspective) {
  if (state.looking.size() == 0) {
    return CG_TRAIN_LOOKING_NULL;
  }
  const bool unauthorized =
      state.lookingPlayer != perspective && state.lookingPlayer != 2 &&
      perspective != 2;
  if (!unauthorized) {
    return CG_TRAIN_LOOKING_VISIBLE;
  }
  if (state.lookingPlayer >= 3 &&
      state.lookingPlayer == perspective + 3) {
    return CG_TRAIN_LOOKING_REDACTED;
  }
  return CG_TRAIN_LOOKING_NULL;
}

class PublicStateSink {
 public:
  PublicStateSink(
      const State& state, CgTrainOutput* output,
      std::uint64_t visible_card_offset, std::uint64_t attachment_offset)
      : state_(state),
        output_(output),
        visible_card_offset_(visible_card_offset),
        attachment_offset_(attachment_offset) {}

  std::uint64_t AppendHiddenCard(
      int owner, int area, int area_index) {
    const std::uint64_t row = visible_card_offset_++;
    if (output_ != nullptr) {
      WriteCard(
          row, owner, area, area_index, 0, 0, 0, 0, false);
    }
    return row;
  }

  std::uint64_t AppendCard(
      CardRef ref, int area, int area_index, bool pokemon,
      bool include_attachments) {
    const Card& card = state_.getCard(ref);
    const CardMaster& master = card.getMaster();
    const std::uint64_t row = visible_card_offset_++;
    if (output_ != nullptr) {
      WriteCard(
          row, card.playerIndex, area, area_index, master.cardId,
          ref.cardIndex, pokemon ? state_.getHp(card) : 0,
          pokemon ? state_.getMaxHp(card) : 0,
          pokemon && card.appear);
    }
    if (include_attachments) {
      AppendAttachments(ref, card, row);
    }
    return row;
  }

  std::uint64_t visible_card_offset() const {
    return visible_card_offset_;
  }

  std::uint64_t attachment_offset() const { return attachment_offset_; }

 private:
  void WriteCard(
      std::uint64_t row, int owner, int area, int area_index, int card_id,
      int serial, int hp, int max_hp, bool appear_this_turn) {
    const std::uint32_t index = static_cast<std::uint32_t>(row);
    output_->visible_card_owner[index] = owner;
    output_->visible_card_area[index] = area;
    output_->visible_card_area_index[index] = area_index;
    output_->visible_card_id[index] = card_id;
    output_->visible_card_serial[index] = serial;
    output_->visible_card_hp[index] = hp;
    output_->visible_card_max_hp[index] = max_hp;
    output_->visible_card_appear_this_turn[index] =
        appear_this_turn ? 1 : 0;
  }

  void AppendAttachment(
      std::uint64_t parent, CardRef ref, int kind, int energy_type,
      int energy_units) {
    const std::uint64_t row = attachment_offset_++;
    if (output_ == nullptr) {
      return;
    }
    const std::uint32_t index = static_cast<std::uint32_t>(row);
    output_->attachment_parent[index] =
        static_cast<std::uint32_t>(parent);
    output_->attachment_kind[index] = kind;
    output_->attachment_card_id[index] =
        state_.getCard(ref).getMaster().cardId;
    output_->attachment_card_serial[index] = ref.cardIndex;
    output_->attachment_energy_type[index] = energy_type;
    output_->attachment_energy_units[index] = energy_units;
  }

  void AppendAttachments(
      CardRef pokemon_ref, const Card& pokemon, std::uint64_t parent) {
    const PlayerState& player = state_.players.at(pokemon.playerIndex);
    for (CardRef ref : player.energy) {
      const Card& attached = state_.getCard(ref);
      if (attached.attachMoveCounter != pokemon.moveCounter) {
        continue;
      }
      const EnergyInfo info = state_.getEnergyInfo(attached, pokemon_ref);
      AppendAttachment(
          parent, ref, CG_TRAIN_ATTACHMENT_ENERGY,
          EnergyTypeIndex(info.type), info.count);
    }
    for (CardRef ref : player.tool) {
      const Card& attached = state_.getCard(ref);
      if (attached.attachMoveCounter == pokemon.moveCounter) {
        AppendAttachment(
            parent, ref, CG_TRAIN_ATTACHMENT_TOOL, -1, 0);
      }
    }
    for (CardRef ref : player.preEvolution) {
      const Card& attached = state_.getCard(ref);
      if (attached.attachMoveCounter == pokemon.moveCounter) {
        AppendAttachment(
            parent, ref, CG_TRAIN_ATTACHMENT_PRE_EVOLUTION, -1, 0);
      }
    }
  }

  const State& state_;
  CgTrainOutput* output_;
  std::uint64_t visible_card_offset_;
  std::uint64_t attachment_offset_;
};

void TraversePublicState(
    const State& state, std::size_t output_row, CgTrainOutput* output,
    PublicStateSink* sink) {
  const int perspective = state.selectPlayer;
  const int player_order[2] = {
      perspective == 1 ? 1 : 0,
      perspective == 1 ? 0 : 1,
  };

  for (int order_index = 0; order_index < 2; ++order_index) {
    const int player_index = player_order[order_index];
    const PlayerState& player = state.players.at(player_index);

    for (int index : range(player.active)) {
      const CardRef ref = player.active[index];
      const Card& card = state.getCard(ref);
      if (card.reverse && perspective != 2) {
        sink->AppendHiddenCard(
            player_index, CG_TRAIN_CARD_AREA_ACTIVE, index);
      } else {
        sink->AppendCard(
            ref, CG_TRAIN_CARD_AREA_ACTIVE, index, true, true);
      }
    }
    for (int index : range(player.bench)) {
      const CardRef ref = player.bench[index];
      const Card& card = state.getCard(ref);
      if (card.reverse && perspective != 2) {
        sink->AppendHiddenCard(
            player_index, CG_TRAIN_CARD_AREA_BENCH, index);
      } else {
        sink->AppendCard(
            ref, CG_TRAIN_CARD_AREA_BENCH, index, true, true);
      }
    }
    if (player_index == perspective || perspective == 2) {
      for (int index : range(player.hand)) {
        sink->AppendCard(
            player.hand[index], CG_TRAIN_CARD_AREA_HAND, index, false,
            false);
      }
    }
    for (int index : range(player.trash)) {
      sink->AppendCard(
          player.trash[index], CG_TRAIN_CARD_AREA_DISCARD, index, false,
          false);
    }
    for (int index : range(player.prize)) {
      const CardRef ref = player.prize[index];
      const Card& card = state.getCard(ref);
      if (card.reverse && perspective != 2) {
        sink->AppendHiddenCard(
            player_index, CG_TRAIN_CARD_AREA_PRIZE, index);
      } else {
        sink->AppendCard(
            ref, CG_TRAIN_CARD_AREA_PRIZE, index, false, false);
      }
    }
  }

  for (int index : range(state.stadium)) {
    sink->AppendCard(
        state.stadium[index], CG_TRAIN_CARD_AREA_STADIUM, index, false,
        false);
  }

  const std::int32_t looking_mode = LookingMode(state, perspective);
  if (output != nullptr) {
    output->looking_mode[output_row] = looking_mode;
  }
  if (looking_mode == CG_TRAIN_LOOKING_VISIBLE) {
    for (int index : range(state.looking)) {
      sink->AppendCard(
          state.looking[index], CG_TRAIN_CARD_AREA_LOOKING, index, false,
          false);
    }
  } else if (looking_mode == CG_TRAIN_LOOKING_REDACTED) {
    for (int index : range(state.looking)) {
      sink->AppendHiddenCard(
          perspective, CG_TRAIN_CARD_AREA_LOOKING, index);
    }
  }

  if (output != nullptr) {
    output->select_deck_visible[output_row] = state.selectDeck ? 1 : 0;
  }
  if (state.selectDeck) {
    const PlayerState& player = state.players.at(state.selectPlayer);
    for (int index : range(player.deck)) {
      sink->AppendCard(
          player.deck[index], CG_TRAIN_CARD_AREA_DECK, index, false,
          false);
    }
  }

  if (!state.contextCard.isNull()) {
    const std::uint64_t row = sink->AppendCard(
        state.contextCard, CG_TRAIN_CARD_AREA_VIRTUAL, 0, false, false);
    if (output != nullptr) {
      output->context_card_row[output_row] =
          static_cast<std::uint32_t>(row);
    }
  }
  if (state.onEffect()) {
    const std::uint64_t row = sink->AppendCard(
        state.getEffectCard().card, CG_TRAIN_CARD_AREA_VIRTUAL, 1, false,
        false);
    if (output != nullptr) {
      output->effect_card_row[output_row] =
          static_cast<std::uint32_t>(row);
    }
  }
}

PublicCounts CountPublicState(const State& state) {
  PublicStateSink sink(state, nullptr, 0, 0);
  TraversePublicState(state, 0, nullptr, &sink);
  return {
      .visible_cards = sink.visible_card_offset(),
      .attachments = sink.attachment_offset(),
  };
}

bool IsSuccessfulOutput(const Row& row) {
  return row.data != nullptr &&
         row.error == CG_TRAIN_SLOT_ERROR_NONE &&
         (row.status == CG_TRAIN_SLOT_READY ||
          row.status == CG_TRAIN_SLOT_FINISHED);
}

std::size_t PublicLogStart(const State& state, int perspective) {
  if (perspective < 0 || perspective >= 2) {
    throw std::runtime_error(
        "successful state has no acting player for public logs");
  }
  const int cursor = state.logIndex.at(perspective);
  if (cursor < 0 ||
      static_cast<std::size_t>(cursor) > state.logs.size()) {
    throw std::runtime_error("engine public-log cursor is out of range");
  }
  return static_cast<std::size_t>(cursor);
}

std::uint64_t CountPublicLogs(const Row& row) {
  if (!IsSuccessfulOutput(row)) {
    return 0;
  }
  const State& state = row.data->state;
  const std::size_t start = PublicLogStart(state, state.selectPlayer);
  std::uint64_t count = 0;
  for (std::size_t index = start; index < state.logs.size(); ++index) {
    if (state.logs[index].logType <= LogType::Result) {
      ++count;
    }
  }
  return count;
}

void WritePublicLog(
    const PublicLog& log, std::uint32_t index, CgTrainOutput* output) {
  output->log_type[index] = log.type;
  output->log_param_count[index] = log.param_count;
  output->log_p0[index] = log.params[0];
  output->log_p1[index] = log.params[1];
  output->log_p2[index] = log.params[2];
  output->log_p3[index] = log.params[3];
  output->log_p4[index] = log.params[4];
  output->log_p5[index] = log.params[5];
  output->log_p6[index] = log.params[6];
}

bool HasOutputPointers(const CgTrainOutput& output) {
  return output.status != nullptr && output.error != nullptr &&
         output.select_player != nullptr && output.select_type != nullptr &&
         output.select_context != nullptr && output.select_min != nullptr &&
         output.select_max != nullptr && output.result != nullptr &&
         output.turn != nullptr && output.option_offsets != nullptr &&
         output.option_type != nullptr && output.option_p0 != nullptr &&
         output.option_p1 != nullptr && output.option_p2 != nullptr &&
         output.option_p3 != nullptr && output.option_p4 != nullptr &&
         output.turn_action_count != nullptr &&
         output.first_player != nullptr && output.turn_flags != nullptr &&
         output.remain_damage_counter != nullptr &&
         output.remain_energy_cost != nullptr &&
         output.player0_deck_count != nullptr &&
         output.player1_deck_count != nullptr &&
         output.player0_hand_count != nullptr &&
         output.player1_hand_count != nullptr &&
         output.player0_prize_count != nullptr &&
         output.player1_prize_count != nullptr &&
         output.player0_bench_max != nullptr &&
         output.player1_bench_max != nullptr &&
         output.player0_status_flags != nullptr &&
         output.player1_status_flags != nullptr &&
         output.looking_mode != nullptr &&
         output.select_deck_visible != nullptr &&
         output.context_card_row != nullptr &&
         output.effect_card_row != nullptr &&
         output.visible_card_offsets != nullptr &&
         output.visible_card_owner != nullptr &&
         output.visible_card_area != nullptr &&
         output.visible_card_area_index != nullptr &&
         output.visible_card_id != nullptr &&
         output.visible_card_serial != nullptr &&
         output.visible_card_hp != nullptr &&
         output.visible_card_max_hp != nullptr &&
         output.visible_card_appear_this_turn != nullptr &&
         output.attachment_offsets != nullptr &&
         output.attachment_parent != nullptr &&
         output.attachment_kind != nullptr &&
         output.attachment_card_id != nullptr &&
         output.attachment_card_serial != nullptr &&
         output.attachment_energy_type != nullptr &&
         output.attachment_energy_units != nullptr &&
         output.log_offsets != nullptr && output.log_type != nullptr &&
         output.log_param_count != nullptr && output.log_p0 != nullptr &&
         output.log_p1 != nullptr && output.log_p2 != nullptr &&
         output.log_p3 != nullptr && output.log_p4 != nullptr &&
         output.log_p5 != nullptr && output.log_p6 != nullptr &&
         output.selection_advance_count != nullptr;
}

std::int32_t ValidateOutput(
    const CgTrainOutput* output, std::uint32_t batch_count,
    const std::vector<Row>& rows, ParallelExecutor* executor,
    std::uint32_t worker_count, OutputPlan* plan) {
  if (output == nullptr || output->struct_size != sizeof(CgTrainOutput)) {
    return Fail(
        CG_TRAIN_INVALID_ARGUMENT,
        "CgTrainOutput is null or has an incompatible struct_size");
  }
  if (output->slot_capacity < batch_count || !HasOutputPointers(*output)) {
    return Fail(
        CG_TRAIN_INSUFFICIENT_CAPACITY,
        "CgTrainOutput slot arrays are null or too small");
  }

  std::vector<RowOutputCounts> row_counts(batch_count);
  executor->ParallelFor(
      batch_count, worker_count, [&](std::size_t row_index) {
        const Row& row = rows[row_index];
        RowOutputCounts& counts = row_counts[row_index];
        if (row.data != nullptr) {
          counts.options = row.data->state.options.size();
          const PublicCounts public_counts =
              CountPublicState(row.data->state);
          counts.visible_cards = public_counts.visible_cards;
          counts.attachments = public_counts.attachments;
        }
        counts.logs = CountPublicLogs(row);
      });

  plan->option_offsets.assign(batch_count + 1, 0);
  plan->visible_card_offsets.assign(batch_count + 1, 0);
  plan->attachment_offsets.assign(batch_count + 1, 0);
  plan->log_offsets.assign(batch_count + 1, 0);
  std::uint64_t option_total = 0;
  std::uint64_t visible_card_total = 0;
  std::uint64_t attachment_total = 0;
  std::uint64_t log_total = 0;
  for (std::size_t row_index = 0; row_index < row_counts.size();
       ++row_index) {
    option_total += row_counts[row_index].options;
    visible_card_total += row_counts[row_index].visible_cards;
    attachment_total += row_counts[row_index].attachments;
    log_total += row_counts[row_index].logs;
    if (option_total <= UINT32_MAX) {
      plan->option_offsets[row_index + 1] =
          static_cast<std::uint32_t>(option_total);
    }
    if (visible_card_total <= UINT32_MAX) {
      plan->visible_card_offsets[row_index + 1] =
          static_cast<std::uint32_t>(visible_card_total);
    }
    if (attachment_total <= UINT32_MAX) {
      plan->attachment_offsets[row_index + 1] =
          static_cast<std::uint32_t>(attachment_total);
    }
    if (log_total <= UINT32_MAX) {
      plan->log_offsets[row_index + 1] =
          static_cast<std::uint32_t>(log_total);
    }
  }
  if (option_total > output->option_capacity || option_total > UINT32_MAX) {
    return Fail(
        CG_TRAIN_INSUFFICIENT_CAPACITY,
        "CgTrainOutput option arrays are too small");
  }
  if (visible_card_total > output->visible_card_capacity ||
      visible_card_total > UINT32_MAX) {
    return Fail(
        CG_TRAIN_INSUFFICIENT_CAPACITY,
        "CgTrainOutput visible-card arrays are too small");
  }
  if (attachment_total > output->attachment_capacity ||
      attachment_total > UINT32_MAX) {
    return Fail(
        CG_TRAIN_INSUFFICIENT_CAPACITY,
        "CgTrainOutput attachment arrays are too small");
  }
  if (log_total > output->log_capacity || log_total > UINT32_MAX) {
    return Fail(
        CG_TRAIN_INSUFFICIENT_CAPACITY,
        "CgTrainOutput public-log arrays are too small");
  }
  return CG_TRAIN_OK;
}

void WriteOutput(
    const std::vector<Row>& rows, const OutputPlan& plan,
    CgTrainOutput* output, ParallelExecutor* executor,
    std::uint32_t worker_count, const std::uint32_t* slots,
    std::vector<std::u8string>* public_observations) {
  std::copy(
      plan.option_offsets.begin(), plan.option_offsets.end(),
      output->option_offsets);
  std::copy(
      plan.visible_card_offsets.begin(), plan.visible_card_offsets.end(),
      output->visible_card_offsets);
  std::copy(
      plan.attachment_offsets.begin(), plan.attachment_offsets.end(),
      output->attachment_offsets);
  std::copy(
      plan.log_offsets.begin(), plan.log_offsets.end(),
      output->log_offsets);
  executor->ParallelFor(
      rows.size(), worker_count, [&](std::size_t row_index) {
    const Row& row = rows[row_index];
    output->status[row_index] = row.status;
    output->error[row_index] = row.error;
    output->selection_advance_count[row_index] =
        row.selection_advance_count;
    output->context_card_row[row_index] = UINT32_MAX;
    output->effect_card_row[row_index] = UINT32_MAX;

    if (row.data == nullptr) {
      output->select_player[row_index] = -1;
      output->select_type[row_index] =
          static_cast<std::int32_t>(SelectType::None);
      output->select_context[row_index] =
          static_cast<std::int32_t>(SelectContext::None);
      output->select_min[row_index] = 0;
      output->select_max[row_index] = 0;
      output->result[row_index] = -1;
      output->turn[row_index] = 0;
      output->turn_action_count[row_index] = 0;
      output->first_player[row_index] = -1;
      output->turn_flags[row_index] = 0;
      output->remain_damage_counter[row_index] = 0;
      output->remain_energy_cost[row_index] = 0;
      output->player0_deck_count[row_index] = -1;
      output->player1_deck_count[row_index] = -1;
      output->player0_hand_count[row_index] = -1;
      output->player1_hand_count[row_index] = -1;
      output->player0_prize_count[row_index] = -1;
      output->player1_prize_count[row_index] = -1;
      output->player0_bench_max[row_index] = -1;
      output->player1_bench_max[row_index] = -1;
      output->player0_status_flags[row_index] = 0;
      output->player1_status_flags[row_index] = 0;
      output->looking_mode[row_index] = CG_TRAIN_LOOKING_NULL;
      output->select_deck_visible[row_index] = 0;
      return;
    }

    State& state = row.data->state;
    output->select_player[row_index] = state.selectPlayer;
    output->select_type[row_index] =
        static_cast<std::int32_t>(state.selectType);
    output->select_context[row_index] =
        static_cast<std::int32_t>(state.selectContext);
    output->select_min[row_index] = state.selectMin;
    output->select_max[row_index] = state.selectMax;
    output->result[row_index] = state.apiResult();
    output->turn[row_index] = state.turn;
    output->turn_action_count[row_index] = state.turnActionCount;
    output->first_player[row_index] = state.firstPlayer;
    output->turn_flags[row_index] = TurnFlags(state);
    output->remain_damage_counter[row_index] = state.remainDamageCounter;
    output->remain_energy_cost[row_index] = state.remainEnergyCost;
    output->player0_deck_count[row_index] = state.players[0].deck.size();
    output->player1_deck_count[row_index] = state.players[1].deck.size();
    output->player0_hand_count[row_index] = state.players[0].hand.size();
    output->player1_hand_count[row_index] = state.players[1].hand.size();
    output->player0_prize_count[row_index] =
        state.players[0].prize.size();
    output->player1_prize_count[row_index] =
        state.players[1].prize.size();
    output->player0_bench_max[row_index] = state.benchCapacity(0);
    output->player1_bench_max[row_index] = state.benchCapacity(1);
    output->player0_status_flags[row_index] =
        PlayerStatusFlags(state.players[0]);
    output->player1_status_flags[row_index] =
        PlayerStatusFlags(state.players[1]);

    std::uint32_t option_offset = plan.option_offsets[row_index];
    for (const SelectOption& option : state.options) {
      output->option_type[option_offset] =
          static_cast<std::int32_t>(option.type);
      output->option_p0[option_offset] = option.param0;
      output->option_p1[option_offset] = option.param1;
      output->option_p2[option_offset] = option.param2;
      output->option_p3[option_offset] = option.param3;
      output->option_p4[option_offset] = option.param4;
      ++option_offset;
    }

    PublicStateSink sink(
        state, output, plan.visible_card_offsets[row_index],
        plan.attachment_offsets[row_index]);
    TraversePublicState(state, row_index, output, &sink);

    if (!IsSuccessfulOutput(row)) {
      return;
    }
    const int perspective = state.selectPlayer;
    const std::size_t log_start = PublicLogStart(state, perspective);
    JsonBuilder observation;
    ToJsonApi(state, observation, static_cast<int>(log_start));
    public_observations->at(slots[row_index]) = std::move(observation.buf);
    std::uint32_t log_offset = plan.log_offsets[row_index];
    for (std::size_t index = log_start; index < state.logs.size(); ++index) {
      const Log& log = state.logs[index];
      if (log.logType > LogType::Result) {
        continue;
      }
      WritePublicLog(
          ProjectPublicLog(log, perspective), log_offset, output);
      ++log_offset;
    }
    state.logIndex.at(perspective) = static_cast<int>(state.logs.size());
    const int consumed_logs =
        std::min(state.logIndex.at(0), state.logIndex.at(1));
    if (consumed_logs > 0) {
      state.logs.erase(
          state.logs.begin(), state.logs.begin() + consumed_logs);
      state.logIndex.at(0) -= consumed_logs;
      state.logIndex.at(1) -= consumed_logs;
    }
  });
}

}  // namespace

struct CgTrainLane {
  CgTrainLane(
      std::uint32_t requested_capacity,
      std::uint32_t requested_worker_count)
      : capacity(requested_capacity),
        worker_count(ResolveParallelWorkerCount(
            requested_worker_count, requested_capacity)),
        slots(requested_capacity),
        public_observations(requested_capacity) {}

  const std::uint32_t capacity;
  const std::uint32_t worker_count;
  std::vector<Slot> slots;
  std::vector<std::u8string> public_observations;
  // Whole calls remain serialized to preserve the existing transactional ABI.
  // Row work inside a call executes through the process-wide worker pool.
  mutable std::mutex operation_mutex;
};

namespace {

bool HasUniqueValidSlots(
    const CgTrainLane* lane, std::uint32_t batch_count,
    const std::uint32_t* slots, std::string* error) {
  if (lane == nullptr || batch_count == 0 || slots == nullptr) {
    *error = "lane, slots, and non-zero batch_count are required";
    return false;
  }
  if (batch_count > lane->capacity) {
    *error = "batch_count exceeds lane capacity";
    return false;
  }

  std::unordered_set<std::uint32_t> seen;
  seen.reserve(batch_count);
  for (std::uint32_t row = 0; row < batch_count; ++row) {
    if (slots[row] >= lane->capacity) {
      *error = "slot id exceeds lane capacity";
      return false;
    }
    if (!seen.insert(slots[row]).second) {
      *error = "slot ids must be unique within a batch";
      return false;
    }
  }
  return true;
}

void CommitPending(CgTrainLane* lane, std::vector<PendingSlot>* pending) {
  for (PendingSlot& item : *pending) {
    if (!item.commit) {
      continue;
    }
    Slot& destination = lane->slots[item.slot];
    if (destination.data == nullptr) {
      destination.data = std::move(item.data);
    } else {
      *destination.data = std::move(*item.data);
      RebindState(destination.data.get());
    }
  }
}

}  // namespace

extern "C" {

const CgTrainAbiDescriptor* CgTrainGetAbiDescriptor(void) {
  return &kAbiDescriptor;
}

const char* CgTrainLastError(void) { return g_last_error.c_str(); }

std::int32_t CgTrainInitialize(void) {
  try {
    InitializeEngine();
    SetLastError("");
    return CG_TRAIN_OK;
  } catch (const std::exception& error) {
    return Fail(CG_TRAIN_ENGINE_EXCEPTION, error.what());
  } catch (...) {
    return Fail(
        CG_TRAIN_ENGINE_EXCEPTION, "unknown engine initialization exception");
  }
}

CgTrainLane* CgTrainCreateWithWorkers(
    std::uint32_t capacity, std::uint32_t worker_count) {
  try {
    InitializeEngine();
    if (capacity == 0 || capacity > kMaxLaneCapacity) {
      SetLastError("lane capacity is outside the supported range");
      return nullptr;
    }
    auto lane = std::make_unique<CgTrainLane>(capacity, worker_count);
    SetLastError("");
    return lane.release();
  } catch (const std::exception& error) {
    SetLastError(error.what());
    return nullptr;
  } catch (...) {
    SetLastError("unknown lane creation exception");
    return nullptr;
  }
}

CgTrainLane* CgTrainCreate(std::uint32_t capacity) {
  return CgTrainCreateWithWorkers(capacity, 0);
}

void CgTrainDestroy(CgTrainLane* lane) { delete lane; }

std::uint32_t CgTrainCapacity(const CgTrainLane* lane) {
  if (lane == nullptr) {
    SetLastError("lane is null");
    return 0;
  }
  SetLastError("");
  return lane->capacity;
}

std::uint32_t CgTrainWorkerCount(const CgTrainLane* lane) {
  if (lane == nullptr) {
    SetLastError("lane is null");
    return 0;
  }
  SetLastError("");
  return lane->worker_count;
}

std::int32_t CgTrainClearSlots(
    CgTrainLane* lane, std::uint32_t batch_count,
    const std::uint32_t* slots) {
  try {
    std::string slot_error;
    if (!HasUniqueValidSlots(lane, batch_count, slots, &slot_error)) {
      return Fail(CG_TRAIN_INVALID_ARGUMENT, slot_error);
    }

    std::scoped_lock<std::mutex> lock(lane->operation_mutex);
    for (std::uint32_t row = 0; row < batch_count; ++row) {
      const std::uint32_t slot = slots[row];
      lane->slots[slot].data.reset();
      std::u8string().swap(lane->public_observations[slot]);
    }
    SetLastError("");
    return CG_TRAIN_OK;
  } catch (const std::exception& error) {
    return Fail(CG_TRAIN_ENGINE_EXCEPTION, error.what());
  } catch (...) {
    return Fail(CG_TRAIN_ENGINE_EXCEPTION, "unknown slot cleanup exception");
  }
}

std::int32_t CgTrainReset(
    CgTrainLane* lane, std::uint32_t batch_count,
    const std::uint32_t* slots, const std::int32_t* decks,
    const std::uint32_t* seeds, CgTrainOutput* output) {
  try {
    InitializeEngine();
    std::string slot_error;
    if (!HasUniqueValidSlots(
            lane, batch_count, slots, &slot_error) ||
        decks == nullptr || seeds == nullptr) {
      return Fail(
          CG_TRAIN_INVALID_ARGUMENT,
          slot_error.empty() ? "decks and seeds are required" : slot_error);
    }

    std::scoped_lock<std::mutex> lock(lane->operation_mutex);
    ParallelExecutor& executor = ParallelExecutor::Global();
    std::vector<PendingSlot> pending(batch_count);
    std::vector<Row> rows(batch_count);
    executor.ParallelFor(
        batch_count, lane->worker_count, [&](std::size_t row) {
      PendingSlot& item = pending[row];
      item.slot = slots[row];
      const std::int32_t* row_decks =
          decks + row * static_cast<std::ptrdiff_t>(2 * DECK_SIZE);
      const std::int32_t deck_error = ValidateDeckPair(row_decks);
      if (deck_error != CG_TRAIN_SLOT_ERROR_NONE) {
        item.row = {
            .data = lane->slots[item.slot].data.get(),
            .status = CG_TRAIN_SLOT_RESET_ERROR,
            .error = deck_error,
        };
      } else {
        try {
          NewBattleResult reset = NewBattle(row_decks, seeds[row]);
          item.data = std::move(reset.data);
          item.row = {
              .data = item.data.get(),
              .status = ReadyStatus(*item.data),
              .error = CG_TRAIN_SLOT_ERROR_NONE,
              .selection_advance_count =
                  reset.selection_advance_count,
          };
          item.commit = true;
        } catch (const std::exception&) {
          item.row = {
              .data = lane->slots[item.slot].data.get(),
              .status = CG_TRAIN_SLOT_ENGINE_ERROR,
              .error = CG_TRAIN_SLOT_ERROR_ENGINE_EXCEPTION,
          };
        }
      }
      rows[row] = item.row;
    });

    OutputPlan output_plan;
    const std::int32_t output_error =
        ValidateOutput(
            output, batch_count, rows, &executor, lane->worker_count,
            &output_plan);
    if (output_error != CG_TRAIN_OK) {
      return output_error;
    }
    CommitPending(lane, &pending);
    for (std::uint32_t row = 0; row < batch_count; ++row) {
      if (pending[row].commit) {
        rows[row].data = lane->slots[pending[row].slot].data.get();
      }
    }
    WriteOutput(
        rows, output_plan, output, &executor, lane->worker_count, slots,
        &lane->public_observations);
    SetLastError("");
    return CG_TRAIN_OK;
  } catch (const std::exception& error) {
    return Fail(CG_TRAIN_ENGINE_EXCEPTION, error.what());
  } catch (...) {
    return Fail(CG_TRAIN_ENGINE_EXCEPTION, "unknown reset exception");
  }
}

std::int32_t CgTrainStep(
    CgTrainLane* lane, std::uint32_t batch_count,
    const std::uint32_t* slots, const std::uint32_t* action_offsets,
    const std::int32_t* actions, std::uint32_t action_count,
    CgTrainOutput* output) {
  try {
    InitializeEngine();
    std::string slot_error;
    if (!HasUniqueValidSlots(lane, batch_count, slots, &slot_error)) {
      return Fail(CG_TRAIN_INVALID_ARGUMENT, slot_error);
    }
    if (action_offsets == nullptr || action_offsets[0] != 0 ||
        action_offsets[batch_count] != action_count ||
        (action_count > 0 && actions == nullptr)) {
      return Fail(CG_TRAIN_INVALID_ARGUMENT, "invalid CSR action buffers");
    }
    for (std::uint32_t row = 0; row < batch_count; ++row) {
      if (action_offsets[row] > action_offsets[row + 1]) {
        return Fail(
            CG_TRAIN_INVALID_ARGUMENT,
            "CSR action offsets must be monotonic");
      }
    }

    std::scoped_lock<std::mutex> lock(lane->operation_mutex);
    ParallelExecutor& executor = ParallelExecutor::Global();
    for (std::uint32_t row = 0; row < batch_count; ++row) {
      if (lane->slots[slots[row]].data == nullptr) {
        return Fail(
            CG_TRAIN_INVALID_ARGUMENT,
            "CgTrainStep references an uninitialized slot");
      }
    }

    std::vector<PendingSlot> pending(batch_count);
    std::vector<Row> rows(batch_count);
    executor.ParallelFor(
        batch_count, lane->worker_count, [&](std::size_t row) {
      PendingSlot& item = pending[row];
      item.slot = slots[row];
      BattleData& source = *lane->slots[item.slot].data;
      if (source.state.isFinish()) {
        item.row = {
            .data = &source,
            .status = CG_TRAIN_SLOT_FINISHED,
            .error = CG_TRAIN_SLOT_ERROR_TERMINAL,
        };
        rows[row] = item.row;
        return;
      }

      item.data = CloneBattle(source);
      State& state = item.data->state;
      const std::uint32_t begin = action_offsets[row];
      const std::uint32_t end = action_offsets[row + 1];
      state.selected.clear();
      state.selected.reserve(end - begin);
      for (std::uint32_t action = begin; action < end; ++action) {
        state.selected.push_back(actions[action]);
      }
      const std::int32_t select_error = state.checkPlayerSelect();
      if (select_error != 0) {
        item.data.reset();
        item.row = {
            .data = &source,
            .status = CG_TRAIN_SLOT_ACTION_ERROR,
            .error = select_error,
        };
      } else {
        try {
          item.data->next();
          const std::uint32_t forced_advances =
              AdvanceForcedChain(item.data.get());
          if (forced_advances ==
              std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(
                "selection advance count overflow");
          }
          item.row = {
              .data = item.data.get(),
              .status = ReadyStatus(*item.data),
              .error = CG_TRAIN_SLOT_ERROR_NONE,
              .selection_advance_count = forced_advances + 1,
          };
          item.commit = true;
        } catch (const std::exception&) {
          item.data.reset();
          item.row = {
              .data = &source,
              .status = CG_TRAIN_SLOT_ENGINE_ERROR,
              .error = CG_TRAIN_SLOT_ERROR_ENGINE_EXCEPTION,
          };
        }
      }
      rows[row] = item.row;
    });

    OutputPlan output_plan;
    const std::int32_t output_error =
        ValidateOutput(
            output, batch_count, rows, &executor, lane->worker_count,
            &output_plan);
    if (output_error != CG_TRAIN_OK) {
      return output_error;
    }
    CommitPending(lane, &pending);
    for (std::uint32_t row = 0; row < batch_count; ++row) {
      if (pending[row].commit) {
        rows[row].data = lane->slots[pending[row].slot].data.get();
      }
    }
    WriteOutput(
        rows, output_plan, output, &executor, lane->worker_count, slots,
        &lane->public_observations);
    SetLastError("");
    return CG_TRAIN_OK;
  } catch (const std::exception& error) {
    return Fail(CG_TRAIN_ENGINE_EXCEPTION, error.what());
  } catch (...) {
    return Fail(CG_TRAIN_ENGINE_EXCEPTION, "unknown step exception");
  }
}

std::int32_t CgTrainExportPublicStateTokens(
    CgTrainLane* lane, std::uint32_t batch_count,
    const std::uint32_t* slots, std::uint32_t token_capacity,
    std::uint32_t* token_offsets, char* token_data) {
  try {
    InitializeEngine();
    std::string slot_error;
    if (!HasUniqueValidSlots(lane, batch_count, slots, &slot_error) ||
        token_offsets == nullptr ||
        (token_capacity > 0 && token_data == nullptr)) {
      return Fail(
          CG_TRAIN_INVALID_ARGUMENT,
          slot_error.empty()
              ? "token offsets and capacity-aligned data are required"
              : slot_error);
    }

    std::scoped_lock<std::mutex> lock(lane->operation_mutex);
    for (std::uint32_t row = 0; row < batch_count; ++row) {
      if (lane->slots[slots[row]].data == nullptr) {
        return Fail(
            CG_TRAIN_INVALID_ARGUMENT,
            "public state token export references an uninitialized slot");
      }
    }

    std::vector<std::vector<char>> tokens(batch_count);
    for (std::uint32_t row = 0; row < batch_count; ++row) {
      const State& source = lane->slots[slots[row]].data->state;
      State public_state;
      public_state.clear();
      public_state = source;
      public_state.erasePlayerData(public_state.selectPlayer);
      BinaryWriter writer;
      public_state.serialize(writer);
      writer.toBase64();
      tokens[row] = std::move(writer.base64);
    }

    std::uint64_t total = 0;
    token_offsets[0] = 0;
    for (std::uint32_t row = 0; row < batch_count; ++row) {
      total += tokens[row].size();
      if (total > std::numeric_limits<std::uint32_t>::max()) {
        return Fail(
            CG_TRAIN_INSUFFICIENT_CAPACITY,
            "public state token batch exceeds uint32 capacity");
      }
      token_offsets[row + 1] = static_cast<std::uint32_t>(total);
    }
    if (total > token_capacity) {
      return Fail(
          CG_TRAIN_INSUFFICIENT_CAPACITY,
          "public state token buffer is too small");
    }
    std::uint32_t offset = 0;
    for (const std::vector<char>& token : tokens) {
      std::copy(token.begin(), token.end(), token_data + offset);
      offset += static_cast<std::uint32_t>(token.size());
    }
    SetLastError("");
    return CG_TRAIN_OK;
  } catch (const std::exception& error) {
    return Fail(CG_TRAIN_ENGINE_EXCEPTION, error.what());
  } catch (...) {
    return Fail(
        CG_TRAIN_ENGINE_EXCEPTION,
        "unknown public state token export exception");
  }
}

std::int32_t CgTrainExportPublicObservations(
    CgTrainLane* lane, std::uint32_t batch_count,
    const std::uint32_t* slots, std::uint32_t observation_capacity,
    std::uint32_t* observation_offsets, char* observation_data) {
  try {
    InitializeEngine();
    std::string slot_error;
    if (!HasUniqueValidSlots(lane, batch_count, slots, &slot_error) ||
        observation_offsets == nullptr ||
        (observation_capacity > 0 && observation_data == nullptr)) {
      return Fail(
          CG_TRAIN_INVALID_ARGUMENT,
          slot_error.empty()
              ? "observation offsets and capacity-aligned data are required"
              : slot_error);
    }

    std::scoped_lock<std::mutex> lock(lane->operation_mutex);
    std::uint64_t total = 0;
    observation_offsets[0] = 0;
    for (std::uint32_t row = 0; row < batch_count; ++row) {
      if (lane->slots[slots[row]].data == nullptr ||
          lane->public_observations[slots[row]].empty()) {
        return Fail(
            CG_TRAIN_INVALID_ARGUMENT,
            "public observation export references an uninitialized slot");
      }
      total += lane->public_observations[slots[row]].size();
      if (total > std::numeric_limits<std::uint32_t>::max()) {
        return Fail(
            CG_TRAIN_INSUFFICIENT_CAPACITY,
            "public observation batch exceeds uint32 capacity");
      }
      observation_offsets[row + 1] = static_cast<std::uint32_t>(total);
    }
    if (total > observation_capacity) {
      return Fail(
          CG_TRAIN_INSUFFICIENT_CAPACITY,
          "public observation buffer is too small");
    }
    std::uint32_t offset = 0;
    for (std::uint32_t row = 0; row < batch_count; ++row) {
      const std::u8string& observation =
          lane->public_observations[slots[row]];
      std::copy(
          observation.begin(), observation.end(),
          reinterpret_cast<char8_t*>(observation_data + offset));
      offset += static_cast<std::uint32_t>(observation.size());
    }
    SetLastError("");
    return CG_TRAIN_OK;
  } catch (const std::exception& error) {
    return Fail(CG_TRAIN_ENGINE_EXCEPTION, error.what());
  } catch (...) {
    return Fail(
        CG_TRAIN_ENGINE_EXCEPTION,
        "unknown public observation export exception");
  }
}

}  // extern "C"

// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only

#include "state_token_validation.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "All.h"

namespace {

constexpr std::size_t kMaximumTokenBytes = 1U << 25;
constexpr std::size_t kMaximumExpandedBase64Bytes = 1U << 26;
constexpr std::size_t kMaximumDecodedBytes = 1U << 25;
constexpr std::uint32_t kMaximumVectorElements = 1U << 20;

int Base64Value(char value) {
  if (value >= 'A' && value <= 'Z') {
    return value - 'A';
  }
  if (value >= 'a' && value <= 'z') {
    return value - 'a' + 26;
  }
  if (value >= '0' && value <= '9') {
    return value - '0' + 52;
  }
  if (value == '+') {
    return 62;
  }
  if (value == '/') {
    return 63;
  }
  return -1;
}

bool AppendRun(
    std::vector<char>* expanded,
    std::size_t count,
    std::string* error) {
  if (count == 0 || count > kMaximumExpandedBase64Bytes - expanded->size()) {
    *error = "state token has an invalid compressed run";
    return false;
  }
  expanded->insert(expanded->end(), count, 'A');
  return true;
}

bool ExpandToken(
    const char* token,
    int token_size,
    std::vector<char>* expanded,
    std::string* error) {
  if (token == nullptr || token_size <= 0 ||
      static_cast<std::size_t>(token_size) > kMaximumTokenBytes) {
    *error = "state token size is outside the native cap";
    return false;
  }
  expanded->clear();
  expanded->reserve(static_cast<std::size_t>(token_size));
  for (int index = 0; index < token_size; ++index) {
    const char value = token[index];
    int run_digits = 0;
    if (value == 'A') {
      run_digits = 1;
    } else if (value == '-') {
      run_digits = 2;
    } else if (value == '*') {
      run_digits = 3;
    }
    if (run_digits > 0) {
      if (index + run_digits >= token_size) {
        *error = "state token ends inside a compressed run";
        return false;
      }
      std::size_t count = 0;
      std::size_t multiplier = 1;
      for (int digit = 0; digit < run_digits; ++digit) {
        const int decoded = Base64Value(token[++index]);
        if (decoded < 0) {
          *error = "state token compressed run has an invalid digit";
          return false;
        }
        count += static_cast<std::size_t>(decoded) * multiplier;
        multiplier *= 64;
      }
      if (!AppendRun(expanded, count, error)) {
        return false;
      }
      continue;
    }
    if (value != '=' && Base64Value(value) < 0) {
      *error = "state token has a non-base64 character";
      return false;
    }
    if (expanded->size() >= kMaximumExpandedBase64Bytes) {
      *error = "state token expansion exceeds the native cap";
      return false;
    }
    expanded->push_back(value);
  }
  return true;
}

bool DecodeBase64(
    const std::vector<char>& expanded,
    std::vector<std::uint8_t>* decoded,
    std::string* error) {
  if (expanded.size() < 4 || expanded.size() % 4 != 0) {
    *error = "expanded state token is not padded base64";
    return false;
  }
  std::size_t padding = 0;
  if (expanded.back() == '=') {
    ++padding;
  }
  if (expanded[expanded.size() - 2] == '=') {
    ++padding;
  }
  for (std::size_t index = 0; index < expanded.size() - padding; ++index) {
    if (expanded[index] == '=' || Base64Value(expanded[index]) < 0) {
      *error = "expanded state token has invalid base64 padding";
      return false;
    }
  }
  for (std::size_t index = expanded.size() - padding;
       index < expanded.size(); ++index) {
    if (expanded[index] != '=') {
      *error = "expanded state token has invalid base64 padding";
      return false;
    }
  }
  const std::size_t output_size = expanded.size() / 4 * 3 - padding;
  if (output_size == 0 || output_size > kMaximumDecodedBytes) {
    *error = "decoded state token size is outside the native cap";
    return false;
  }
  decoded->assign(output_size, 0);
  std::size_t output_index = 0;
  for (std::size_t index = 0; index < expanded.size(); index += 4) {
    std::uint32_t block = 0;
    for (int digit = 0; digit < 4; ++digit) {
      const char value = expanded[index + digit];
      const int base64 = value == '=' ? 0 : Base64Value(value);
      block = (block << 6U) | static_cast<std::uint32_t>(base64);
    }
    for (int shift : {16, 8, 0}) {
      if (output_index < output_size) {
        (*decoded)[output_index++] =
            static_cast<std::uint8_t>((block >> shift) & 0xffU);
      }
    }
  }
  return true;
}

bool ConsumeBytes(
    const std::vector<std::uint8_t>& decoded,
    std::size_t count,
    std::size_t* position) {
  if (*position > decoded.size() || count > decoded.size() - *position) {
    return false;
  }
  *position += count;
  return true;
}

bool ReadCount(
    const std::vector<std::uint8_t>& decoded,
    std::size_t* position,
    std::uint32_t* count) {
  if (!ConsumeBytes(decoded, sizeof(std::int32_t), position)) {
    return false;
  }
  const std::size_t start = *position - sizeof(std::int32_t);
  const std::uint32_t raw =
      static_cast<std::uint32_t>(decoded[start]) |
      (static_cast<std::uint32_t>(decoded[start + 1]) << 8U) |
      (static_cast<std::uint32_t>(decoded[start + 2]) << 16U) |
      (static_cast<std::uint32_t>(decoded[start + 3]) << 24U);
  if (raw > kMaximumVectorElements) {
    return false;
  }
  *count = raw;
  return true;
}

template <typename Element>
bool ConsumeVector(
    const std::vector<std::uint8_t>& decoded,
    std::size_t* position) {
  std::uint32_t count = 0;
  if (!ReadCount(decoded, position, &count)) {
    return false;
  }
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(Element)) {
    return false;
  }
  return ConsumeBytes(decoded, count * sizeof(Element), position);
}

bool ValidateStateLayout(
    const std::vector<std::uint8_t>& decoded,
    std::string* error) {
  State layout;
  const auto* fixed_begin =
      reinterpret_cast<const unsigned char*>(&layout.turn);
  const auto* fixed_end =
      reinterpret_cast<const unsigned char*>(&layout.options);
  const std::size_t fixed_size = static_cast<std::size_t>(
      fixed_end - fixed_begin);
  std::size_t position = 0;
  const bool valid =
      ConsumeBytes(decoded, fixed_size, &position) &&
      ConsumeVector<SelectOption>(decoded, &position) &&
      ConsumeVector<int>(decoded, &position) &&
      ConsumeVector<AreaRef>(decoded, &position) &&
      ConsumeVector<AreaRef>(decoded, &position) &&
      ConsumeVector<AreaRef>(decoded, &position) &&
      ConsumeVector<TriggeredAbility>(decoded, &position) &&
      ConsumeVector<TriggeredAbility>(decoded, &position) &&
      ConsumeVector<TriggeredAbility>(decoded, &position) &&
      ConsumeVector<int>(decoded, &position) &&
      ConsumeVector<CardRef>(decoded, &position) &&
      ConsumeVector<CardRef>(decoded, &position) &&
      ConsumeVector<EvolveInfo>(decoded, &position) &&
      ConsumeVector<GameFunction>(decoded, &position) &&
      ConsumeVector<Log>(decoded, &position) &&
      position == decoded.size();
  if (!valid) {
    *error = "decoded state token has an invalid serialized layout";
  }
  return valid;
}

template <typename List>
bool ValidFixedList(const List& values) {
  return values.size() >= 0 && values.size() <= values.capacity();
}

template <typename Enum>
bool EnumInClosedRange(Enum value, Enum first, Enum last) {
  static_assert(std::is_enum_v<Enum>);
  using Underlying = std::underlying_type_t<Enum>;
  const Underlying raw = static_cast<Underlying>(value);
  return raw >= static_cast<Underlying>(first) &&
         raw <= static_cast<Underlying>(last);
}

template <typename Enum>
bool IntegerInEnumRange(int value, Enum first, Enum last) {
  static_assert(std::is_enum_v<Enum>);
  return value >= static_cast<int>(first) &&
         value <= static_cast<int>(last);
}

bool ValidBoolObject(const bool& value) {
  static_assert(sizeof(bool) == sizeof(unsigned char));
  unsigned char raw = 0;
  std::memcpy(&raw, &value, sizeof(raw));
  return raw <= 1;
}

bool ValidPlayerIndex(int player_index) {
  return player_index == 0 || player_index == 1;
}

bool ValidCardRef(CardRef reference) {
  return reference.isNull() || reference.cardIndex < 128;
}

bool ValidAreaRef(AreaRef reference) {
  return ValidCardRef(reference.card);
}

template <typename List>
bool ValidCardRefs(const List& values) {
  for (CardRef reference : values) {
    if (!ValidCardRef(reference)) {
      return false;
    }
  }
  return true;
}

template <typename List>
bool ValidAreaRefs(const List& values) {
  for (AreaRef reference : values) {
    if (!ValidAreaRef(reference)) {
      return false;
    }
  }
  return true;
}

bool ValidFixedPlayerLists(const PlayerState& player) {
  return ValidFixedList(player.active) && ValidFixedList(player.bench) &&
         ValidFixedList(player.prize) && ValidFixedList(player.hand) &&
         ValidFixedList(player.deck) && ValidFixedList(player.trash) &&
         ValidFixedList(player.energy) && ValidFixedList(player.tool) &&
         ValidFixedList(player.preEvolution) &&
         ValidFixedList(player.temporary);
}

bool ValidCopiedFixedListCounts(const State& state) {
  if (!ValidFixedPlayerLists(state.players[0]) ||
      !ValidFixedPlayerLists(state.players[1]) ||
      !ValidFixedList(state.stadium) || !ValidFixedList(state.looking) ||
      !ValidFixedList(state.selectedList) ||
      !ValidFixedList(state.eachList) || !ValidFixedList(state.playing) ||
      !ValidFixedList(state.checkList)) {
    return false;
  }
  for (const Card& card : state.allCard) {
    if (!ValidFixedList(card.abilityUsed)) {
      return false;
    }
  }
  for (const Log& log : state.logs) {
    if (!ValidFixedList(log.param)) {
      return false;
    }
  }
  return true;
}

bool ValidPlayerState(const PlayerState& player, int player_index) {
  return player.playerIndex == player_index &&
         ValidBoolObject(player.koPrizeOnceChanged) &&
         ValidBoolObject(player.burned) &&
         EnumInClosedRange(
             player.badStatus, BadStatusType::None, BadStatusType::Confused) &&
         ValidCardRefs(player.active) && ValidCardRefs(player.bench) &&
         ValidCardRefs(player.prize) && ValidCardRefs(player.hand) &&
         ValidCardRefs(player.deck) && ValidCardRefs(player.trash) &&
         ValidCardRefs(player.energy) && ValidCardRefs(player.tool) &&
         ValidCardRefs(player.preEvolution) &&
         ValidCardRefs(player.temporary);
}

bool ValidAttackId(int attack_id) {
  return attack_id == 0 || AttackTable.contains(attack_id);
}

bool ValidCard(const Card& card) {
  if (!ValidPlayerIndex(card.playerIndex) ||
      !EnumInClosedRange(card.area, AreaType::All, AreaType::Temporary) ||
      !EnumInClosedRange(card.preArea, AreaType::All, AreaType::Temporary) ||
      !ValidBoolObject(card.reverse) || card.koCauseRef >= 128 ||
      (card.cardId != 0 && !CardTable.contains(card.cardId)) ||
      !ValidAttackId(card.cannotUseAttackIdNonActive)) {
    return false;
  }
  for (short card_id : card.abilityUsed) {
    if (card_id != 0 && !CardTable.contains(card_id)) {
      return false;
    }
  }
  return true;
}

bool ValidActivateAbility(
    const ActivateAbilityInfo& ability,
    bool require_skill) {
  if (!ValidAreaRef(ability.effectCard) ||
      !ValidPlayerIndex(ability.usePlayerIndex) ||
      !ValidBoolObject(ability.isEffectStack) ||
      !ValidBoolObject(ability.isSpecialCondition)) {
    return false;
  }
  if (ability.skillId == 0) {
    return !require_skill || ability.isSpecialCondition;
  }
  return SkillTable.contains(ability.skillId);
}

bool ValidTriggerInfo(const TriggerInfo& trigger) {
  return EnumInClosedRange(
             trigger.type, TriggerType::None, TriggerType::Attach) &&
         trigger.depth >= 0 && ValidAreaRef(trigger.subject) &&
         ValidAreaRef(trigger.object);
}

bool ValidTriggeredAbility(const TriggeredAbility& ability) {
  return ValidActivateAbility(ability.activateInfo, true) &&
         ValidTriggerInfo(ability.trigger);
}

bool ValidSelectOption(const SelectOption& option) {
  if (!EnumInClosedRange(
          option.type, SelectOptionType::Number,
          SelectOptionType::SpecialCondition)) {
    return false;
  }
  switch (option.type) {
    case SelectOptionType::Card:
    case SelectOptionType::ToolCard:
    case SelectOptionType::EnergyCard:
    case SelectOptionType::Energy:
      return ValidPlayerIndex(option.param2) && option.param1 >= 0 &&
             IntegerInEnumRange(
                 option.param0, AreaType::Deck, AreaType::Temporary);
    case SelectOptionType::Attach:
    case SelectOptionType::Evolve:
      return option.param1 >= 0 && option.param3 >= 0 &&
             IntegerInEnumRange(
                 option.param0, AreaType::Deck, AreaType::Temporary) &&
             IntegerInEnumRange(
                 option.param2, AreaType::Deck, AreaType::Temporary);
    case SelectOptionType::Ability:
    case SelectOptionType::Discard:
      return option.param1 >= 0 &&
             IntegerInEnumRange(
                 option.param0, AreaType::Deck, AreaType::Temporary);
    case SelectOptionType::Play:
      return option.param0 >= 0;
    case SelectOptionType::Attack:
      return ValidAttackId(option.param0) && ValidAttackId(option.param1);
    case SelectOptionType::Skill:
      return option.param1 >= 0 && option.param1 < 128 &&
             (option.param0 == 0 || CardTable.contains(option.param0));
    case SelectOptionType::SpecialCondition:
      return option.param0 >=
                 static_cast<int>(SelectSpecialConditionType::Poison) &&
             option.param0 <=
                 static_cast<int>(SelectSpecialConditionType::Confuse);
    default:
      return true;
  }
}

int RequiredLogParameterCount(LogType type) {
  switch (type) {
    case LogType::Shuffle:
    case LogType::TurnStart:
    case LogType::TurnEnd:
    case LogType::DrawReverse:
      return 1;
    case LogType::HasBasicPokemon:
    case LogType::Coin:
    case LogType::Result:
      return 2;
    case LogType::Draw:
    case LogType::MoveCardReverse:
    case LogType::Play:
      return 3;
    case LogType::Attack:
    case LogType::Poisoned:
    case LogType::Burned:
    case LogType::Asleep:
    case LogType::Paralyzed:
    case LogType::Confused:
      return 4;
    case LogType::Switch:
    case LogType::Change:
    case LogType::Attach:
    case LogType::Evolve:
    case LogType::Devolve:
    case LogType::HpChange:
      return 5;
    case LogType::MoveCard:
      return 6;
    case LogType::MoveAttached:
      return 7;
  }
  return -1;
}

bool ValidLog(const Log& log) {
  if (!EnumInClosedRange(log.logType, LogType::Shuffle, LogType::Result) ||
      log.param.size() != RequiredLogParameterCount(log.logType)) {
    return false;
  }
  return log.logType == LogType::Result || ValidPlayerIndex(log.param[0]);
}

bool ValidGameFunction(const GameFunction& function) {
  return function.functionIndex >= 0 &&
         static_cast<std::size_t>(function.functionIndex) <
             FunctionTable.size() &&
         EnumInClosedRange(
             function.argType, ArgType::None, ArgType::III) &&
         function.callCount > 0 && function.calledCount < function.callCount;
}

bool ValidStateBooleans(const State& state) {
  if (!ValidBoolObject(state.setupDone[0]) ||
      !ValidBoolObject(state.setupDone[1]) ||
      !ValidBoolObject(state.mulligan[0]) ||
      !ValidBoolObject(state.mulligan[1]) ||
      !ValidBoolObject(state.lookingReverse) ||
      !ValidBoolObject(state.isBreak) ||
      !ValidBoolObject(state.effectLoopStop) ||
      !ValidBoolObject(state.changed) ||
      !ValidBoolObject(state.stateChanged) ||
      !ValidBoolObject(state.updateOrder) ||
      !ValidBoolObject(state.failRetreat) ||
      !ValidBoolObject(state.selectDeck) ||
      !ValidBoolObject(state.attachActive) ||
      !ValidBoolObject(state.effectState.onEffect) ||
      !ValidBoolObject(state.postAttackEffect) ||
      !ValidBoolObject(state.postEffectActivate) ||
      !ValidBoolObject(state.failAttack) ||
      !ValidBoolObject(state.secondAttack)) {
    return false;
  }
  for (const TurnHistory& history : state.turnHistories) {
    if (!ValidBoolObject(history.ko) ||
        !ValidBoolObject(history.koTeamRocket) ||
        !ValidBoolObject(history.koAttackDamage) ||
        !ValidBoolObject(history.koAttackDamageEthan) ||
        !ValidBoolObject(history.koAttackDamageHop)) {
      return false;
    }
  }
  return true;
}

bool ValidSerializedVectors(const State& state) {
  for (const SelectOption& option : state.options) {
    if (!ValidSelectOption(option)) {
      return false;
    }
  }
  for (int selected : state.selected) {
    if (selected < 0 ||
        static_cast<std::size_t>(selected) >= state.options.size()) {
      return false;
    }
  }
  if (!ValidAreaRefs(state.preTargetList) ||
      !ValidAreaRefs(state.targetList) || !ValidAreaRefs(state.koList)) {
    return false;
  }
  for (const TriggeredAbility& ability : state.delayTriggerStack) {
    if (!ValidTriggeredAbility(ability)) {
      return false;
    }
  }
  for (const TriggeredAbility& ability : state.temporaryTriggerStack) {
    if (!ValidTriggeredAbility(ability)) {
      return false;
    }
  }
  for (const TriggeredAbility& ability : state.triggerStack) {
    if (!ValidTriggeredAbility(ability)) {
      return false;
    }
  }
  for (int skill_id : state.turnUsedSkill) {
    if (!SkillTable.contains(skill_id)) {
      return false;
    }
  }
  if (!ValidCardRefs(state.turnPlay) || !ValidCardRefs(state.turnHeal)) {
    return false;
  }
  for (const EvolveInfo& evolve : state.turnEvolve) {
    if (!ValidCardRef(evolve.preRef) || !ValidCardRef(evolve.ref)) {
      return false;
    }
  }
  for (const GameFunction& function : state.functionStack) {
    if (!ValidGameFunction(function)) {
      return false;
    }
  }
  for (const Log& log : state.logs) {
    if (!ValidLog(log)) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool DecodeValidatedStateToken(
    const char* token,
    int token_size,
    std::vector<std::uint8_t>* decoded,
    std::string* error) {
  if (decoded == nullptr || error == nullptr) {
    return false;
  }
  std::vector<char> expanded;
  if (!ExpandToken(token, token_size, &expanded, error) ||
      !DecodeBase64(expanded, decoded, error)) {
    return false;
  }
  return ValidateStateLayout(*decoded, error);
}

bool ValidateDeserializedPlannerRoot(const State& state, int root_player) {
  // Do this first.  Copying State before these checks would invoke
  // FixedListBase::operator= with untrusted serialized counts.
  if (!ValidCopiedFixedListCounts(state)) {
    return false;
  }

  if (!ValidStateBooleans(state) ||
      !ValidPlayerState(state.players[0], 0) ||
      !ValidPlayerState(state.players[1], 1) ||
      !ValidCardRefs(state.stadium) || !ValidCardRefs(state.looking) ||
      !ValidCardRefs(state.selectedList) || !ValidCardRefs(state.eachList) ||
      !ValidCardRefs(state.playing) || !ValidCardRefs(state.checkList) ||
      !ValidActivateAbility(state.effectState.ability, false) ||
      !ValidTriggerInfo(state.triggerInfo) ||
      !ValidCardRef(state.contextCard) ||
      !ValidCardRef(state.selectingEnergyPokemonRef) ||
      !ValidCardRef(state.attacker) || !ValidSerializedVectors(state)) {
    return false;
  }
  for (const TurnHistory& history : state.turnHistories) {
    if (!ValidCardRef(history.turnAttackCard) ||
        !ValidAttackId(history.turnAttackId)) {
      return false;
    }
  }
  for (const Card& card : state.allCard) {
    if (!ValidCard(card)) {
      return false;
    }
  }

  const int select_type = static_cast<int>(state.selectType);
  const int select_context = static_cast<int>(state.selectContext);
  if (state.selectPlayer != root_player || state.isFinish() ||
      !EnumInClosedRange(
          state.phase, GamePhase::Setup, GamePhase::PokemonCheckupEnd) ||
      !EnumInClosedRange(
          state.gameResult, GameResult::None, GameResult::Draw) ||
      select_type <= static_cast<int>(SelectType::None) ||
      select_type > static_cast<int>(SelectType::SpecialCondition) ||
      select_context <= static_cast<int>(SelectContext::None) ||
      select_context >
          static_cast<int>(SelectContext::RecoverSpecialCondition) ||
      state.options.size() > 128 || state.selected.size() > 128 ||
      state.selectMin < 0 || state.selectMax < state.selectMin ||
      state.selectMax > static_cast<int>(state.options.size()) ||
      state.turn < 0 || state.turn == std::numeric_limits<int>::max() ||
      !ValidAttackId(state.currentAttackId) ||
      !ValidAttackId(state.srcAttackId)) {
    return false;
  }
  return state.firstPlayer == -1 || ValidPlayerIndex(state.firstPlayer);
}

// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®,
// and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use
// only; the full license is in data/ptcg_engine/ptcgProgram 22/LICENSES/.

#include "probe_features.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <numeric>
#include <vector>

namespace cg_probe_internal {
namespace {

constexpr double kMaxDamage = 400.0;
constexpr double kMaxBenchDamage = 1000.0;
constexpr double kMaxBenchCount = 5.0;
constexpr double kMaxDraw = 10.0;
constexpr double kMaxEnergyDelta = 10.0;
constexpr double kMaxPrize = 6.0;
constexpr double kMaxCoin = 16.0;
constexpr double kMaxDiscardInflow = 20.0;
constexpr double kMaxKnockoutCount = 6.0;

struct PokemonKey {
  int player_index;
  int card_id;
  int serial;

  bool operator==(const PokemonKey&) const = default;
};

struct PokemonSnapshot {
  PokemonKey key;
  int hp;
  int energy_count;
  bool active;
};

struct Amount {
  PokemonKey key;
  int value;
};

struct StatusFlags {
  PokemonKey key;
  std::array<bool, 5> values = {};
};

double Norm(double value, double denominator) {
  return std::clamp(value, 0.0, denominator) / denominator;
}

double SignedNorm(double value, double denominator) {
  return std::clamp(value, -denominator, denominator) / denominator;
}

int ParamAt(const WireLog& log, int index) {
  if (index < 0 || index >= static_cast<int>(log.params.size())) {
    return 0;
  }
  return log.params[index];
}

PokemonKey LogKey(const WireLog& log) {
  if (log.type >= LogType::Poisoned && log.type <= LogType::Confused) {
    return {ParamAt(log, 0), ParamAt(log, 2), ParamAt(log, 3)};
  }
  return {ParamAt(log, 0), ParamAt(log, 1), ParamAt(log, 2)};
}

void AddAmount(std::vector<Amount>& amounts, const PokemonKey& key, int value) {
  auto found = std::find_if(
      amounts.begin(), amounts.end(),
      [&](const Amount& amount) { return amount.key == key; });
  if (found == amounts.end()) {
    amounts.push_back({key, value});
  } else {
    found->value += value;
  }
}

int GetAmount(const std::vector<Amount>& amounts, const PokemonKey* key) {
  if (key == nullptr) {
    return 0;
  }
  auto found = std::find_if(
      amounts.begin(), amounts.end(),
      [&](const Amount& amount) { return amount.key == *key; });
  return found == amounts.end() ? 0 : found->value;
}

bool ContainsKey(
    const std::vector<PokemonKey>& values,
    const PokemonKey& key) {
  return std::find(values.begin(), values.end(), key) != values.end();
}

const PokemonSnapshot* FindSnapshot(
    const std::vector<PokemonSnapshot>& snapshots,
    int player_index,
    int serial) {
  auto found = std::find_if(
      snapshots.begin(), snapshots.end(), [&](const PokemonSnapshot& value) {
        return value.key.player_index == player_index &&
               value.key.serial == serial;
      });
  return found == snapshots.end() ? nullptr : &*found;
}

void AddPokemon(
    const State& state,
    std::vector<PokemonSnapshot>& snapshots,
    CardRef ref,
    int player_index,
    bool active) {
  if (ref.isNull()) {
    return;
  }
  const Card& card = state.getCard(ref);
  if (card.reverse) {
    return;
  }
  std::vector<CardRef> energy_cards;
  state.getEnergyCards(ref, energy_cards);
  snapshots.push_back({
      {player_index, card.getMaster().cardId, ref.cardIndex},
      state.getHp(card),
      static_cast<int>(energy_cards.size()),
      active,
  });
}

std::vector<PokemonSnapshot> PokemonSnapshots(const State& state) {
  std::vector<PokemonSnapshot> result;
  for (int player_index = 0; player_index < 2; ++player_index) {
    const PlayerState& player = state.players[player_index];
    for (CardRef ref : player.active) {
      AddPokemon(state, result, ref, player_index, true);
    }
    for (CardRef ref : player.bench) {
      AddPokemon(state, result, ref, player_index, false);
    }
  }
  return result;
}

const PokemonKey* ActiveKey(
    const std::vector<PokemonSnapshot>& snapshots,
    int player_index) {
  auto found = std::find_if(
      snapshots.begin(), snapshots.end(), [&](const PokemonSnapshot& value) {
        return value.key.player_index == player_index && value.active;
      });
  return found == snapshots.end() ? nullptr : &found->key;
}

std::vector<PokemonKey> BenchKeys(
    const std::vector<PokemonSnapshot>& snapshots,
    int player_index) {
  std::vector<PokemonKey> result;
  for (const PokemonSnapshot& snapshot : snapshots) {
    if (snapshot.key.player_index == player_index && !snapshot.active) {
      result.push_back(snapshot.key);
    }
  }
  return result;
}

int TotalEnergy(
    const std::vector<PokemonSnapshot>& snapshots,
    int player_index) {
  int total = 0;
  for (const PokemonSnapshot& snapshot : snapshots) {
    if (snapshot.key.player_index == player_index) {
      total += snapshot.energy_count;
    }
  }
  return total;
}

void AddStatus(
    std::vector<StatusFlags>& statuses,
    const PokemonKey& key,
    int status_index) {
  auto found = std::find_if(
      statuses.begin(), statuses.end(),
      [&](const StatusFlags& status) { return status.key == key; });
  if (found == statuses.end()) {
    statuses.push_back({key, {}});
    found = std::prev(statuses.end());
  }
  found->values[status_index] = true;
}

std::array<bool, 5> GetStatus(
    const std::vector<StatusFlags>& statuses,
    const PokemonKey* key) {
  if (key == nullptr) {
    return {};
  }
  auto found = std::find_if(
      statuses.begin(), statuses.end(),
      [&](const StatusFlags& status) { return status.key == *key; });
  return found == statuses.end() ? std::array<bool, 5>{} : found->values;
}

std::vector<PokemonKey> ConservativeKnockouts(
    const std::vector<PokemonSnapshot>& before,
    const std::vector<PokemonSnapshot>& after,
    const std::vector<Amount>& damage,
    const std::vector<PokemonKey>& moved_to_discard,
    const std::array<int, 2>& prizes_taken) {
  std::vector<PokemonKey> knockouts;
  for (const PokemonSnapshot& snapshot : before) {
    const PokemonSnapshot* successor = FindSnapshot(
        after, snapshot.key.player_index, snapshot.key.serial);
    if (successor != nullptr && successor->hp > 0) {
      continue;
    }
    const bool lethal_damage =
        GetAmount(damage, &snapshot.key) >= std::max(1, snapshot.hp);
    const int prize_taker = 1 - snapshot.key.player_index;
    const bool prize_evidence = ContainsKey(moved_to_discard, snapshot.key) &&
                                prizes_taken[prize_taker] > 0;
    if (lethal_damage || prize_evidence) {
      knockouts.push_back(snapshot.key);
    }
  }
  return knockouts;
}

std::array<float, 3> TerminalFeatures(
    const State& after,
    int perspective) {
  const int result = after.apiResult();
  if (result < 0) {
    return {0.0F, 0.0F, 0.0F};
  }
  if (result == 2) {
    return {0.0F, 0.0F, 1.0F};
  }
  return result == perspective ? std::array<float, 3>{1.0F, 0.0F, 0.0F}
                               : std::array<float, 3>{0.0F, 1.0F, 0.0F};
}

}  // namespace

DynamicEffectFeatures BuildDynamicEffectFeatures(
    const State& before,
    const State& after,
    const std::vector<WireLog>& logs) {
  const int perspective = before.selectPlayer;
  const int opponent = 1 - perspective;
  const std::vector<PokemonSnapshot> before_pokemon =
      PokemonSnapshots(before);
  const std::vector<PokemonSnapshot> after_pokemon = PokemonSnapshots(after);
  const PokemonKey* opponent_active = ActiveKey(before_pokemon, opponent);
  const PokemonKey* self_active = ActiveKey(before_pokemon, perspective);
  const std::vector<PokemonKey> opponent_bench =
      BenchKeys(before_pokemon, opponent);
  const std::vector<PokemonKey> self_bench =
      BenchKeys(before_pokemon, perspective);

  std::vector<Amount> damage;
  std::vector<Amount> healing;
  std::vector<StatusFlags> statuses;
  std::vector<PokemonKey> moved_to_discard;
  std::array<int, 2> draws = {};
  int self_discard_inflow = 0;
  int opponent_discard_inflow = 0;
  int coin_count = 0;
  int coin_heads = 0;

  for (const WireLog& log : logs) {
    if (log.type == LogType::HpChange) {
      const PokemonKey key = LogKey(log);
      const int raw_value = ParamAt(log, 3);
      if (raw_value < 0) {
        AddAmount(damage, key, -raw_value);
      } else if (raw_value > 0) {
        AddAmount(healing, key, raw_value);
      } else {
        const PokemonSnapshot* before_value =
            FindSnapshot(before_pokemon, key.player_index, key.serial);
        const PokemonSnapshot* after_value =
            FindSnapshot(after_pokemon, key.player_index, key.serial);
        if (before_value != nullptr && after_value != nullptr) {
          const int delta = before_value->hp - after_value->hp;
          if (delta > 0) {
            AddAmount(damage, key, delta);
          } else if (delta < 0) {
            AddAmount(healing, key, -delta);
          }
        }
      }
      continue;
    }
    if (log.type >= LogType::Poisoned && log.type <= LogType::Confused) {
      if (ParamAt(log, 1) == 0) {
        AddStatus(
            statuses,
            LogKey(log),
            static_cast<int>(log.type) - static_cast<int>(LogType::Poisoned));
      }
      continue;
    }
    if (log.type == LogType::Coin) {
      ++coin_count;
      coin_heads += ParamAt(log, 1) != 0 ? 1 : 0;
      continue;
    }
    if (log.type == LogType::Draw || log.type == LogType::DrawReverse) {
      const int player_index = ParamAt(log, 0);
      if (player_index >= 0 && player_index < 2) {
        ++draws[player_index];
      }
      continue;
    }
    if (log.type == LogType::MoveCard ||
        log.type == LogType::MoveCardReverse) {
      const int player_index = ParamAt(log, 0);
      const int from_area = ParamAt(log, log.type == LogType::MoveCard ? 3 : 1);
      const int to_area = ParamAt(log, log.type == LogType::MoveCard ? 4 : 2);
      if (to_area == static_cast<int>(AreaType::Trash)) {
        if (player_index == perspective) {
          ++self_discard_inflow;
        } else if (player_index == opponent) {
          ++opponent_discard_inflow;
        }
      }
      if (log.type == LogType::MoveCard &&
          (from_area == static_cast<int>(AreaType::Active) ||
           from_area == static_cast<int>(AreaType::Bench)) &&
          to_area == static_cast<int>(AreaType::Trash)) {
        moved_to_discard.push_back(LogKey(log));
      }
    }
  }

  const std::array<int, 2> prizes_taken = {
      std::max(
          0,
          static_cast<int>(before.players[0].prize.size()) -
              static_cast<int>(after.players[0].prize.size())),
      std::max(
          0,
          static_cast<int>(before.players[1].prize.size()) -
              static_cast<int>(after.players[1].prize.size())),
  };
  const std::vector<PokemonKey> knockouts = ConservativeKnockouts(
      before_pokemon,
      after_pokemon,
      damage,
      moved_to_discard,
      prizes_taken);

  std::vector<int> opponent_bench_damage;
  for (const PokemonKey& key : opponent_bench) {
    opponent_bench_damage.push_back(GetAmount(damage, &key));
  }
  std::vector<int> self_bench_damage;
  for (const PokemonKey& key : self_bench) {
    self_bench_damage.push_back(GetAmount(damage, &key));
  }
  const int opponent_bench_total = std::accumulate(
      opponent_bench_damage.begin(), opponent_bench_damage.end(), 0);
  const int self_bench_total =
      std::accumulate(self_bench_damage.begin(), self_bench_damage.end(), 0);
  const int opponent_bench_max = opponent_bench_damage.empty()
                                     ? 0
                                     : *std::max_element(
                                           opponent_bench_damage.begin(),
                                           opponent_bench_damage.end());
  const int opponent_bench_damaged = static_cast<int>(std::count_if(
      opponent_bench_damage.begin(),
      opponent_bench_damage.end(),
      [](int value) { return value > 0; }));
  const int self_knockouts = static_cast<int>(std::count_if(
      knockouts.begin(), knockouts.end(), [&](const PokemonKey& key) {
        return key.player_index == perspective;
      }));
  const int opponent_knockouts = static_cast<int>(std::count_if(
      knockouts.begin(), knockouts.end(), [&](const PokemonKey& key) {
        return key.player_index == opponent;
      }));
  const std::array<bool, 5> opponent_status =
      GetStatus(statuses, opponent_active);
  const std::array<bool, 5> self_status = GetStatus(statuses, self_active);
  const std::array<float, 3> terminal = TerminalFeatures(after, perspective);

  DynamicEffectFeatures result = {};
  int index = 0;
  auto append = [&](double value) { result[index++] = static_cast<float>(value); };
  append(Norm(GetAmount(damage, opponent_active), kMaxDamage));
  append(opponent_active != nullptr && ContainsKey(knockouts, *opponent_active));
  append(Norm(opponent_bench_total, kMaxBenchDamage));
  append(Norm(opponent_bench_max, kMaxDamage));
  append(Norm(opponent_bench_damaged, kMaxBenchCount));
  append(Norm(GetAmount(damage, self_active), kMaxDamage));
  append(self_active != nullptr && ContainsKey(knockouts, *self_active));
  append(Norm(self_bench_total, kMaxBenchDamage));
  for (bool value : opponent_status) {
    append(value);
  }
  append(Norm(draws[perspective], kMaxDraw));
  append(SignedNorm(
      TotalEnergy(after_pokemon, perspective) -
          TotalEnergy(before_pokemon, perspective),
      kMaxEnergyDelta));
  append(Norm(prizes_taken[perspective], kMaxPrize));
  append(Norm(coin_count, kMaxCoin));
  append(Norm(coin_heads, kMaxCoin));
  for (float value : terminal) {
    append(value);
  }
  for (bool value : self_status) {
    append(value);
  }
  append(Norm(GetAmount(healing, self_active), kMaxDamage));
  append(SignedNorm(
      TotalEnergy(after_pokemon, opponent) -
          TotalEnergy(before_pokemon, opponent),
      kMaxEnergyDelta));
  append(Norm(draws[opponent], kMaxDraw));
  append(Norm(self_discard_inflow, kMaxDiscardInflow));
  append(Norm(opponent_discard_inflow, kMaxDiscardInflow));
  append(Norm(self_knockouts, kMaxKnockoutCount));
  append(Norm(opponent_knockouts, kMaxKnockoutCount));
  return result;
}

}  // namespace cg_probe_internal

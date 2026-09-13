// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®,
// and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use
// only; the full license is in data/ptcg_engine/ptcgProgram 22/LICENSES/.

// Debug-only reference executable. This file is not linked into
// libcg_train.so. It emits the canonical ToJsonApi observation for a
// deterministic minimum-selection trace so the columnar ABI can be compared
// against the engine's public visibility contract.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "All.h"

namespace {

constexpr char kSelectionAdvancePrefix[] =
    "CG_TRAIN_SELECTION_ADVANCE_COUNT ";

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

bool ReadDecks(std::array<Deck, 2>* decks) {
  for (Deck& deck : *decks) {
    for (CardId& card_id : deck.cards) {
      if (!(std::cin >> card_id)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: cg_train_parity_probe SEED MAX_STEPS < 120_ids\n";
    return 2;
  }
  const std::uint32_t seed =
      static_cast<std::uint32_t>(std::strtoull(argv[1], nullptr, 10));
  const int maximum_steps = std::atoi(argv[2]);
  if (maximum_steps <= 0) {
    std::cerr << "MAX_STEPS must be positive\n";
    return 2;
  }

  GameConfig config = {};
  if (!ReadDecks(&config.decks)) {
    std::cerr << "expected exactly two 60-card decks on stdin\n";
    return 2;
  }
  config.seed = seed;
  config.recordLog = true;
  config.manualCoin = false;
  config.sendDeck = false;
  config.deviceRand = false;

  InitializeAll();
  BattleData data;
  data.init(config, false);
  data.game.config.seed = seed;
  data.game.rng = std::mt19937(seed);
  data.start();
  data.next();
  std::uint32_t selection_advance_count = AdvanceForcedChain(&data);

  for (int step = 0; step < maximum_steps; ++step) {
    std::cerr << kSelectionAdvancePrefix << selection_advance_count << '\n';
    JsonBuilder json;
    const int log_start = data.state.nextLogStart();
    ToJsonApi(data.state, json, log_start);
    std::cout.write(
        reinterpret_cast<const char*>(json.buf.data()), json.buf.size());
    std::cout << '\n';
    if (data.state.isFinish()) {
      break;
    }
    data.state.selected.clear();
    for (int selected = 0; selected < data.state.selectMin; ++selected) {
      data.state.selected.push_back(selected);
    }
    if (data.state.checkPlayerSelect() != 0) {
      std::cerr << "minimum-selection trace became illegal\n";
      return 3;
    }
    data.next();
    const std::uint32_t forced_advances = AdvanceForcedChain(&data);
    if (forced_advances == std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("selection advance count overflow");
    }
    selection_advance_count = forced_advances + 1;
  }
  return 0;
}

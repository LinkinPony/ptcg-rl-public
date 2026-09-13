// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®,
// and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use
// only; the full license is in data/ptcg_engine/ptcgProgram 22/LICENSES/.

#ifndef SRC_NATIVE_CG_TRAIN_PUBLIC_LOG_H_
#define SRC_NATIVE_CG_TRAIN_PUBLIC_LOG_H_

#include <array>
#include <cstdint>
#include <stdexcept>

#include "Game.h"

namespace cg_train_internal {

struct PublicLog {
  std::int32_t type = 0;
  std::uint32_t param_count = 0;
  std::array<std::int32_t, 7> params = {};
};

inline PublicLog ProjectPublicLog(const Log& log, int perspective) {
  PublicLog output = {
      .type = static_cast<std::int32_t>(log.logType),
  };
  auto copy_params = [&log, &output](std::uint32_t count) {
    output.param_count = count;
    for (std::uint32_t index = 0; index < count; ++index) {
      output.params[index] = log.param[index];
    }
  };

  switch (log.logType) {
    case LogType::Shuffle:
    case LogType::TurnStart:
    case LogType::TurnEnd:
    case LogType::DrawReverse:
      copy_params(1);
      break;
    case LogType::HasBasicPokemon:
    case LogType::Coin:
    case LogType::Result:
      copy_params(2);
      break;
    case LogType::Draw:
      if (log.param[0] == perspective || perspective == 2) {
        copy_params(3);
      } else {
        output.type = static_cast<std::int32_t>(LogType::DrawReverse);
        output.param_count = 1;
        output.params[0] = log.param[0];
      }
      break;
    case LogType::MoveCard: {
      const bool visible =
          log.param[5] == 0 ||
          (log.param[5] == 1 && log.param[0] == perspective) ||
          (log.param[5] == 3 && perspective == 0) ||
          (log.param[5] == 4 && perspective == 1) ||
          perspective == 2;
      if (visible) {
        copy_params(5);
      } else {
        output.type =
            static_cast<std::int32_t>(LogType::MoveCardReverse);
        output.param_count = 3;
        output.params[0] = log.param[0];
        output.params[1] = log.param[3];
        output.params[2] = log.param[4];
      }
      break;
    }
    case LogType::MoveCardReverse:
    case LogType::Play:
      copy_params(3);
      break;
    case LogType::Attack:
    case LogType::Poisoned:
    case LogType::Burned:
    case LogType::Asleep:
    case LogType::Paralyzed:
    case LogType::Confused:
      copy_params(4);
      break;
    case LogType::Switch:
    case LogType::Change:
    case LogType::Attach:
    case LogType::Evolve:
    case LogType::Devolve:
    case LogType::HpChange:
      copy_params(5);
      break;
    case LogType::MoveAttached:
      copy_params(7);
      break;
    default:
      throw std::runtime_error("unsupported public engine log type");
  }
  return output;
}

}  // namespace cg_train_internal

#endif  // SRC_NATIVE_CG_TRAIN_PUBLIC_LOG_H_

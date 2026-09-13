// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®,
// and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use
// only; the full license is in data/ptcg_engine/ptcgProgram 22/LICENSES/.

// Debug-only exhaustive projection probe. This is not linked into
// libcg_train.so.

#include <iostream>

#include "All.h"
#include "public_log.h"

namespace {

Log ExampleLog(LogType type, int open_type) {
  Log log(type);
  log.add(0);
  log.add(1);
  log.add(102);
  log.add(3);
  log.add(1);
  log.add(open_type);
  log.add(106);
  return log;
}

void WriteCase(const Log& log, int perspective) {
  JsonBuilder reference;
  LogJson(reference, log, perspective, false);
  const cg_train_internal::PublicLog projected =
      cg_train_internal::ProjectPublicLog(log, perspective);

  std::cout << "{\"reference\":";
  std::cout.write(
      reinterpret_cast<const char*>(reference.buf.data()),
      reference.buf.size());
  std::cout << ",\"projected\":{\"type\":" << projected.type
            << ",\"params\":[";
  for (std::uint32_t index = 0; index < projected.param_count; ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << projected.params[index];
  }
  std::cout << "]}}\n";
}

}  // namespace

int main() {
  for (int type = static_cast<int>(LogType::Shuffle);
       type <= static_cast<int>(LogType::Result); ++type) {
    for (int perspective = 0; perspective < 2; ++perspective) {
      if (type == static_cast<int>(LogType::MoveCard)) {
        for (int open_type = 0; open_type <= 4; ++open_type) {
          WriteCase(
              ExampleLog(static_cast<LogType>(type), open_type),
              perspective);
        }
      } else {
        WriteCase(
            ExampleLog(static_cast<LogType>(type), 2), perspective);
      }
    }
  }
  return 0;
}

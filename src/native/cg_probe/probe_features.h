// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®,
// and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use
// only; the full license is in data/ptcg_engine/ptcgProgram 22/LICENSES/.

#pragma once

#include <array>
#include <vector>

#include "All.h"

namespace cg_probe_internal {

constexpr int kDynamicEffectFeatureWidth = 33;

struct WireLog {
  LogType type;
  std::array<int, 7> params;
};

using DynamicEffectFeatures =
    std::array<float, kDynamicEffectFeatureWidth>;

// This is the native equivalent of native_transition_feature_vector().  It is
// deliberately fed the same privacy-projected logs as the diagnostic payload
// path; parity against that path is a required build/runtime gate.
DynamicEffectFeatures BuildDynamicEffectFeatures(
    const State& before,
    const State& after,
    const std::vector<WireLog>& logs);

}  // namespace cg_probe_internal

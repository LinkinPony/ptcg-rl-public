// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only

#pragma once

#include <array>
#include <cstdint>

// Recompute the Python request packer's v5 digest from the raw buffers that
// native execution actually consumes.
std::array<std::uint8_t, 32> ConsequenceRequestFingerprint(
    const char* state_token,
    int state_token_count,
    const int* hidden_counts,
    int hidden_count_count,
    const int* hidden_values,
    int hidden_value_count,
    int world_count,
    const int* candidate_counts,
    int candidate_count_count,
    const int* candidate_values,
    int candidate_value_count,
    int candidate_count,
    int root_player,
    bool manual_coin,
    std::uint64_t stochastic_seed,
    int max_cells,
    int max_engine_steps,
    int max_forced_steps,
    int max_observation_bytes);

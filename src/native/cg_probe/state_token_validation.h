// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only

#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct State;

// Decode the engine's run-length-compressed base64 state token and validate
// every fixed/vector boundary before the bundled BinaryReader sees it.
bool DecodeValidatedStateToken(
    const char* token,
    int token_size,
    std::vector<std::uint8_t>* decoded,
    std::string* error);

// Validate every serialized invariant that must hold before State is copied or
// iterated.  In particular, the bundled FixedList copy constructor trusts its
// serialized count, so all top-level and nested counts must be checked in
// place first.
bool ValidateDeserializedPlannerRoot(const State& state, int root_player);

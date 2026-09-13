// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only

#include "request_fingerprint.h"

#include <array>
#include <cstdint>
#include <span>

#include "sha256.h"

namespace {

constexpr char kDomain[] = "ptcg-rl/native-consequence-request/v5\0";

void UpdateByte(Sha256* digest, std::uint8_t value) {
  digest->Update(std::span<const std::uint8_t>(&value, 1));
}

void UpdateLittleEndian32(Sha256* digest, std::uint32_t value) {
  const std::array<std::uint8_t, 4> bytes = {
      static_cast<std::uint8_t>(value & 0xffU),
      static_cast<std::uint8_t>((value >> 8U) & 0xffU),
      static_cast<std::uint8_t>((value >> 16U) & 0xffU),
      static_cast<std::uint8_t>((value >> 24U) & 0xffU),
  };
  digest->Update(bytes);
}

void UpdateLittleEndian64(Sha256* digest, std::uint64_t value) {
  const std::array<std::uint8_t, 8> bytes = {
      static_cast<std::uint8_t>(value & 0xffU),
      static_cast<std::uint8_t>((value >> 8U) & 0xffU),
      static_cast<std::uint8_t>((value >> 16U) & 0xffU),
      static_cast<std::uint8_t>((value >> 24U) & 0xffU),
      static_cast<std::uint8_t>((value >> 32U) & 0xffU),
      static_cast<std::uint8_t>((value >> 40U) & 0xffU),
      static_cast<std::uint8_t>((value >> 48U) & 0xffU),
      static_cast<std::uint8_t>((value >> 56U) & 0xffU),
  };
  digest->Update(bytes);
}

void UpdateIntegers(Sha256* digest, const int* values, int count) {
  UpdateLittleEndian32(digest, static_cast<std::uint32_t>(count));
  for (int index = 0; index < count; ++index) {
    UpdateLittleEndian32(
        digest, static_cast<std::uint32_t>(values[index]));
  }
}

}  // namespace

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
    int max_observation_bytes) {
  Sha256 digest;
  digest.Update(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(kDomain), sizeof(kDomain) - 1));
  UpdateLittleEndian32(&digest, static_cast<std::uint32_t>(state_token_count));
  digest.Update(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(state_token),
      static_cast<std::size_t>(state_token_count)));
  for (int value : {
           world_count,
           candidate_count,
           root_player,
           max_cells,
           max_engine_steps,
           max_forced_steps,
           max_observation_bytes,
       }) {
    UpdateLittleEndian32(&digest, static_cast<std::uint32_t>(value));
  }
  UpdateByte(&digest, manual_coin ? 1U : 0U);
  UpdateLittleEndian64(&digest, stochastic_seed);
  UpdateIntegers(&digest, hidden_counts, hidden_count_count);
  UpdateIntegers(&digest, hidden_values, hidden_value_count);
  UpdateIntegers(&digest, candidate_counts, candidate_count_count);
  UpdateIntegers(&digest, candidate_values, candidate_value_count);
  return digest.Finalize();
}

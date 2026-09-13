// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

// Minimal incremental SHA-256 used to bind native ABI inputs without a
// platform crypto-library dependency.
class Sha256 {
 public:
  Sha256();

  void Update(std::span<const std::uint8_t> bytes);
  std::array<std::uint8_t, 32> Finalize();

 private:
  void Transform(const std::uint8_t* block);

  std::array<std::uint32_t, 8> state_;
  std::array<std::uint8_t, 64> buffer_{};
  std::uint64_t total_bytes_ = 0;
  std::size_t buffered_bytes_ = 0;
  bool finalized_ = false;
};

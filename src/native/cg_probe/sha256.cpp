// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only

#include "sha256.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

constexpr std::uint32_t RotateRight(std::uint32_t value, int bits) {
  return (value >> bits) | (value << (32 - bits));
}

}  // namespace

Sha256::Sha256()
    : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
             0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

void Sha256::Update(std::span<const std::uint8_t> bytes) {
  if (finalized_) {
    throw std::logic_error("cannot update a finalized SHA-256 digest");
  }
  total_bytes_ += bytes.size();
  std::size_t offset = 0;
  if (buffered_bytes_ > 0) {
    const std::size_t copied =
        std::min(bytes.size(), buffer_.size() - buffered_bytes_);
    std::copy_n(bytes.data(), copied, buffer_.data() + buffered_bytes_);
    buffered_bytes_ += copied;
    offset += copied;
    if (buffered_bytes_ == buffer_.size()) {
      Transform(buffer_.data());
      buffered_bytes_ = 0;
    }
  }
  while (bytes.size() - offset >= buffer_.size()) {
    Transform(bytes.data() + offset);
    offset += buffer_.size();
  }
  if (offset < bytes.size()) {
    buffered_bytes_ = bytes.size() - offset;
    std::copy_n(bytes.data() + offset, buffered_bytes_, buffer_.data());
  }
}

std::array<std::uint8_t, 32> Sha256::Finalize() {
  if (finalized_) {
    throw std::logic_error("cannot finalize SHA-256 more than once");
  }
  finalized_ = true;
  const std::uint64_t total_bits = total_bytes_ * 8U;
  buffer_[buffered_bytes_++] = 0x80U;
  if (buffered_bytes_ > 56) {
    std::fill(buffer_.begin() + buffered_bytes_, buffer_.end(), 0U);
    Transform(buffer_.data());
    buffered_bytes_ = 0;
  }
  std::fill(buffer_.begin() + buffered_bytes_, buffer_.begin() + 56, 0U);
  for (int index = 0; index < 8; ++index) {
    buffer_[56 + index] = static_cast<std::uint8_t>(
        (total_bits >> (56 - 8 * index)) & 0xffU);
  }
  Transform(buffer_.data());

  std::array<std::uint8_t, 32> digest{};
  for (std::size_t word = 0; word < state_.size(); ++word) {
    for (int byte = 0; byte < 4; ++byte) {
      digest[word * 4 + byte] = static_cast<std::uint8_t>(
          (state_[word] >> (24 - 8 * byte)) & 0xffU);
    }
  }
  return digest;
}

void Sha256::Transform(const std::uint8_t* block) {
  std::array<std::uint32_t, 64> words{};
  for (int index = 0; index < 16; ++index) {
    const int offset = index * 4;
    words[index] =
        (static_cast<std::uint32_t>(block[offset]) << 24U) |
        (static_cast<std::uint32_t>(block[offset + 1]) << 16U) |
        (static_cast<std::uint32_t>(block[offset + 2]) << 8U) |
        static_cast<std::uint32_t>(block[offset + 3]);
  }
  for (int index = 16; index < 64; ++index) {
    const std::uint32_t s0 =
        RotateRight(words[index - 15], 7) ^
        RotateRight(words[index - 15], 18) ^ (words[index - 15] >> 3U);
    const std::uint32_t s1 =
        RotateRight(words[index - 2], 17) ^
        RotateRight(words[index - 2], 19) ^ (words[index - 2] >> 10U);
    words[index] = words[index - 16] + s0 + words[index - 7] + s1;
  }

  std::uint32_t a = state_[0];
  std::uint32_t b = state_[1];
  std::uint32_t c = state_[2];
  std::uint32_t d = state_[3];
  std::uint32_t e = state_[4];
  std::uint32_t f = state_[5];
  std::uint32_t g = state_[6];
  std::uint32_t h = state_[7];
  for (int index = 0; index < 64; ++index) {
    const std::uint32_t sum1 =
        RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
    const std::uint32_t choose = (e & f) ^ (~e & g);
    const std::uint32_t temp1 =
        h + sum1 + choose + kRoundConstants[index] + words[index];
    const std::uint32_t sum0 =
        RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
    const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t temp2 = sum0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }
  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}

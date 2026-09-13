// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®,
// and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only

#include "parallel_executor.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t kRows = 257;

void Require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void RunSuccessfulWave(
    cg_train_internal::ParallelExecutor* executor,
    std::uint32_t worker_count) {
  std::vector<std::atomic<std::uint32_t>> visits(kRows);
  executor->ParallelFor(
      kRows, worker_count, [&](std::size_t row) {
        visits[row].fetch_add(1, std::memory_order_relaxed);
      });
  for (const auto& visit : visits) {
    Require(
        visit.load(std::memory_order_relaxed) == 1,
        "parallel row was not executed exactly once");
  }
}

}  // namespace

int main() {
  cg_train_internal::ParallelExecutor& executor =
      cg_train_internal::ParallelExecutor::Global();
  const std::uint32_t workers = executor.worker_count();
  Require(workers > 0, "parallel executor has no workers");

  for (std::size_t wave = 0; wave < 32; ++wave) {
    bool caught = false;
    try {
      executor.ParallelFor(
          kRows, workers, [](std::size_t row) {
            if (row == 17) {
              throw std::runtime_error("synthetic worker failure");
            }
          });
    } catch (const std::runtime_error&) {
      caught = true;
    }
    Require(caught, "parallel worker exception was not propagated");
    RunSuccessfulWave(&executor, workers);
  }

  std::vector<std::thread> callers;
  callers.reserve(3);
  for (std::size_t caller = 0; caller < 3; ++caller) {
    callers.emplace_back(
        [&executor, workers] { RunSuccessfulWave(&executor, workers); });
  }
  for (std::thread& caller : callers) {
    caller.join();
  }
  return 0;
}

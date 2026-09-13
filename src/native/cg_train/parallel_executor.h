// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®,
// and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use
// only; the full license is in data/ptcg_engine/ptcgProgram 22/LICENSES/.

#ifndef SRC_NATIVE_CG_TRAIN_PARALLEL_EXECUTOR_H_
#define SRC_NATIVE_CG_TRAIN_PARALLEL_EXECUTOR_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace cg_train_internal {

// Process-wide persistent worker pool. Jobs from independent lanes share the
// same affinity-sized pool, so Python arena shards cannot multiply native
// thread counts or oversubscribe the training process.
class ParallelExecutor {
 public:
  static ParallelExecutor& Global();

  ParallelExecutor(const ParallelExecutor&) = delete;
  ParallelExecutor& operator=(const ParallelExecutor&) = delete;

  ~ParallelExecutor();

  std::uint32_t worker_count() const;

  // Execute [0, count) exactly once. max_workers limits concurrent chunks for
  // this job without reserving threads from other lanes. The first exception
  // is rethrown after all scheduled chunks have retired.
  void ParallelFor(
      std::size_t count, std::uint32_t max_workers,
      const std::function<void(std::size_t)>& function);

 private:
  class Impl;

  explicit ParallelExecutor(std::uint32_t worker_count);

  std::unique_ptr<Impl> impl_;
};

// Zero requests the automatic affinity-sized value. Explicit requests and the
// CG_TRAIN_WORKER_COUNT environment override are clamped to the process worker
// pool and lane capacity.
std::uint32_t ResolveParallelWorkerCount(
    std::uint32_t requested, std::uint32_t lane_capacity);

}  // namespace cg_train_internal

#endif  // SRC_NATIVE_CG_TRAIN_PARALLEL_EXECUTOR_H_

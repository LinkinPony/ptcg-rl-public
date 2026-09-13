// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®,
// and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use
// only; the full license is in data/ptcg_engine/ptcgProgram 22/LICENSES/.

#include "parallel_executor.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace cg_train_internal {
namespace {

constexpr std::size_t kMinimumParallelRows = 16;
constexpr std::size_t kChunksPerWorker = 4;

std::uint32_t AvailableCpuCount() {
#if defined(__linux__)
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
    const int count = CPU_COUNT(&affinity);
    if (count > 0) {
      return static_cast<std::uint32_t>(count);
    }
  }
#endif
  const std::uint32_t reported = std::thread::hardware_concurrency();
  return std::max<std::uint32_t>(reported, 1);
}

std::uint32_t EnvironmentWorkerCount() {
  const char* raw = std::getenv("CG_TRAIN_WORKER_COUNT");
  if (raw == nullptr || raw[0] == '\0') {
    return 0;
  }
  char* end = nullptr;
  errno = 0;
  const unsigned long value = std::strtoul(raw, &end, 10);
  if (errno != 0 || end == raw || *end != '\0' || value == 0 ||
      value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(
        "CG_TRAIN_WORKER_COUNT must be a positive uint32");
  }
  return static_cast<std::uint32_t>(value);
}

}  // namespace

class ParallelExecutor::Impl {
 public:
  struct Job {
    Job(
        std::size_t row_count, std::uint32_t concurrency,
        const std::function<void(std::size_t)>* row_function)
        : count(row_count),
          max_active(concurrency),
          function(row_function),
          remaining(row_count) {
      const std::size_t denominator =
          static_cast<std::size_t>(max_active) * kChunksPerWorker;
      grain = std::max<std::size_t>(
          1, (count + denominator - 1) / denominator);
    }

    const std::size_t count;
    const std::uint32_t max_active;
    const std::function<void(std::size_t)>* function;
    std::size_t grain = 1;
    std::size_t next = 0;
    std::size_t remaining;
    std::uint32_t active = 0;
    bool queued = false;
    std::atomic<bool> cancelled = false;
    std::exception_ptr exception;
    std::condition_variable done;
  };

  explicit Impl(std::uint32_t requested_workers)
      : worker_count(std::max<std::uint32_t>(requested_workers, 1)) {
    workers.reserve(worker_count);
    for (std::uint32_t index = 0; index < worker_count; ++index) {
      workers.emplace_back([this] { WorkerLoop(); });
    }
  }

  ~Impl() {
    {
      const std::scoped_lock<std::mutex> lock(mutex);
      stopping = true;
    }
    ready.notify_all();
    for (std::thread& worker : workers) {
      worker.join();
    }
  }

  void Enqueue(Job* job) {
    if (job->queued || job->next >= job->count ||
        job->active >= job->max_active) {
      return;
    }
    job->queued = true;
    jobs.push_back(job);
    ready.notify_one();
  }

  void WorkerLoop() {
    for (;;) {
      Job* job = nullptr;
      std::size_t begin = 0;
      std::size_t end = 0;
      {
        std::unique_lock<std::mutex> lock(mutex);
        ready.wait(lock, [this] { return stopping || !jobs.empty(); });
        if (stopping && jobs.empty()) {
          return;
        }
        job = jobs.front();
        jobs.pop_front();
        job->queued = false;
        begin = job->next;
        end = std::min(job->count, begin + job->grain);
        job->next = end;
        ++job->active;
        Enqueue(job);
      }

      for (std::size_t row = begin; row < end; ++row) {
        if (job->cancelled.load(std::memory_order_relaxed)) {
          continue;
        }
        try {
          (*job->function)(row);
        } catch (...) {
          const std::scoped_lock<std::mutex> lock(mutex);
          if (job->exception == nullptr) {
            job->exception = std::current_exception();
          }
          job->cancelled.store(true, std::memory_order_relaxed);
        }
      }

      {
        const std::scoped_lock<std::mutex> lock(mutex);
        --job->active;
        job->remaining -= end - begin;
        if (job->remaining == 0) {
          job->done.notify_one();
        } else {
          Enqueue(job);
        }
      }
    }
  }

  const std::uint32_t worker_count;
  std::mutex mutex;
  std::condition_variable ready;
  std::deque<Job*> jobs;
  std::vector<std::thread> workers;
  bool stopping = false;
};

ParallelExecutor& ParallelExecutor::Global() {
  static ParallelExecutor executor(AvailableCpuCount());
  return executor;
}

ParallelExecutor::ParallelExecutor(std::uint32_t worker_count)
    : impl_(std::make_unique<Impl>(worker_count)) {}

ParallelExecutor::~ParallelExecutor() = default;

std::uint32_t ParallelExecutor::worker_count() const {
  return impl_->worker_count;
}

void ParallelExecutor::ParallelFor(
    std::size_t count, std::uint32_t max_workers,
    const std::function<void(std::size_t)>& function) {
  if (count == 0) {
    return;
  }
  const std::uint32_t concurrency = std::min<std::uint32_t>(
      std::max<std::uint32_t>(max_workers, 1),
      static_cast<std::uint32_t>(
          std::min<std::size_t>(count, impl_->worker_count)));
  if (concurrency == 1 || count < kMinimumParallelRows) {
    for (std::size_t row = 0; row < count; ++row) {
      function(row);
    }
    return;
  }

  Impl::Job job(count, concurrency, &function);
  std::exception_ptr exception;
  {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    impl_->Enqueue(&job);
    job.done.wait(lock, [&job] { return job.remaining == 0; });
    exception = job.exception;
  }
  if (exception != nullptr) {
    std::rethrow_exception(exception);
  }
}

std::uint32_t ResolveParallelWorkerCount(
    std::uint32_t requested, std::uint32_t lane_capacity) {
  const std::uint32_t pool_workers =
      ParallelExecutor::Global().worker_count();
  std::uint32_t resolved = requested;
  if (resolved == 0) {
    resolved = EnvironmentWorkerCount();
  }
  if (resolved == 0) {
    resolved = pool_workers;
  }
  return std::max<std::uint32_t>(
      1, std::min({resolved, pool_workers, lane_capacity}));
}

}  // namespace cg_train_internal

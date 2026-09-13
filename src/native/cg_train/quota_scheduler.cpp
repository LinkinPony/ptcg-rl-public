#include "quota_scheduler.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cg_train.h"

namespace {

constexpr std::uint32_t kNoFrozenArtifact =
    std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t kMaximumArtifacts = 20;
thread_local std::string g_quota_scheduler_error;

struct QuotaRowState {
  std::uint64_t initial = 0;
  std::uint64_t remaining = 0;
  std::uint32_t frozen_artifact_slot = kNoFrozenArtifact;
  std::uint32_t cohort_artifact_slot = kNoFrozenArtifact;
};

std::uint64_t ArtifactMask(std::uint32_t slot) {
  if (slot == kNoFrozenArtifact) {
    return 0;
  }
  if (slot >= kMaximumArtifacts) {
    throw std::invalid_argument("quota artifact slot exceeds the native limit");
  }
  return UINT64_C(1) << slot;
}

std::uint32_t Popcount(std::uint64_t value) {
  return static_cast<std::uint32_t>(__builtin_popcountll(value));
}

struct MaskProgress {
  std::uint64_t initial = 0;
  std::uint64_t issued = 0;
};

MaskProgress CohortProgress(
    const std::vector<QuotaRowState>& rows, std::uint64_t mask) {
  MaskProgress progress;
  for (const QuotaRowState& row : rows) {
    const std::uint64_t row_mask = ArtifactMask(row.cohort_artifact_slot);
    if (row_mask == 0 || (row_mask & mask) == 0) {
      continue;
    }
    progress.initial += row.initial;
    progress.issued += row.initial - row.remaining;
  }
  return progress;
}

bool EarlierProgress(
    const MaskProgress& left, std::uint64_t left_mask,
    const MaskProgress& right, std::uint64_t right_mask) {
  if (left.initial == 0 || right.initial == 0) {
    if (left.initial != right.initial) {
      return left.initial != 0;
    }
    return left_mask < right_mask;
  }
  const unsigned __int128 left_scaled =
      static_cast<unsigned __int128>(left.issued) * right.initial;
  const unsigned __int128 right_scaled =
      static_cast<unsigned __int128>(right.issued) * left.initial;
  if (left_scaled != right_scaled) {
    return left_scaled < right_scaled;
  }
  return left_mask < right_mask;
}

bool GreaterRowDeficit(
    const QuotaRowState& left, std::uint32_t left_index,
    const QuotaRowState& right, std::uint32_t right_index,
    std::uint64_t eligible_initial, std::uint64_t next_eligible_issued) {
  const unsigned __int128 left_issued = left.initial - left.remaining;
  const unsigned __int128 right_issued = right.initial - right.remaining;
  const unsigned __int128 left_score =
      static_cast<unsigned __int128>(left.initial) * next_eligible_issued +
      right_issued * eligible_initial;
  const unsigned __int128 right_score =
      static_cast<unsigned __int128>(right.initial) * next_eligible_issued +
      left_issued * eligible_initial;
  if (left_score != right_score) {
    return left_score > right_score;
  }
  return left_index < right_index;
}

}  // namespace

struct CgTrainQuotaScheduler {
  std::vector<QuotaRowState> rows;
  std::uint64_t remaining = 0;
  std::uint32_t artifact_count = 0;
};

extern "C" {

const char* CgTrainQuotaSchedulerLastError(void) {
  return g_quota_scheduler_error.c_str();
}

CgTrainQuotaScheduler* CgTrainQuotaSchedulerCreate(
    const CgTrainQuotaRow* rows, std::uint32_t row_count,
    std::uint32_t artifact_count) {
  try {
    if (rows == nullptr || row_count == 0 || artifact_count == 0 ||
        artifact_count > kMaximumArtifacts) {
      throw std::invalid_argument("quota scheduler shape is invalid");
    }
    auto scheduler = std::make_unique<CgTrainQuotaScheduler>();
    scheduler->rows.reserve(row_count);
    scheduler->artifact_count = artifact_count;
    for (std::uint32_t index = 0; index < row_count; ++index) {
      const CgTrainQuotaRow& row = rows[index];
      if (row.game_count == 0) {
        throw std::invalid_argument("quota rows must be non-empty");
      }
      if ((row.frozen_artifact_slot != kNoFrozenArtifact &&
           row.frozen_artifact_slot >= artifact_count) ||
          (row.cohort_artifact_slot != kNoFrozenArtifact &&
           row.cohort_artifact_slot >= artifact_count)) {
        throw std::invalid_argument("quota row artifact slot is invalid");
      }
      if (scheduler->remaining >
          std::numeric_limits<std::uint64_t>::max() - row.game_count) {
        throw std::overflow_error("quota game count overflows uint64");
      }
      scheduler->rows.push_back(
          {.initial = row.game_count,
           .remaining = row.game_count,
           .frozen_artifact_slot = row.frozen_artifact_slot,
           .cohort_artifact_slot = row.cohort_artifact_slot});
      scheduler->remaining += row.game_count;
    }
    g_quota_scheduler_error.clear();
    return scheduler.release();
  } catch (const std::exception& error) {
    g_quota_scheduler_error = error.what();
    return nullptr;
  } catch (...) {
    g_quota_scheduler_error = "unknown quota scheduler creation error";
    return nullptr;
  }
}

void CgTrainQuotaSchedulerDestroy(CgTrainQuotaScheduler* scheduler) {
  delete scheduler;
}

std::uint64_t CgTrainQuotaSchedulerRemaining(
    const CgTrainQuotaScheduler* scheduler) {
  return scheduler == nullptr ? 0 : scheduler->remaining;
}

int32_t CgTrainQuotaSchedulerCanTake(
    const CgTrainQuotaScheduler* scheduler, std::uint32_t game_count,
    std::uint32_t frozen_artifact_limit,
    std::uint32_t required_artifact_slot) {
  if (scheduler == nullptr || game_count == 0 ||
      frozen_artifact_limit == 0 ||
      frozen_artifact_limit > scheduler->artifact_count ||
      (required_artifact_slot != kNoFrozenArtifact &&
       required_artifact_slot >= scheduler->artifact_count)) {
    return 0;
  }
  const std::uint64_t required_mask = ArtifactMask(required_artifact_slot);
  const std::uint64_t subset_limit = UINT64_C(1) << scheduler->artifact_count;
  for (std::uint64_t mask = 0; mask < subset_limit; ++mask) {
    if ((mask & required_mask) != required_mask ||
        Popcount(mask) > frozen_artifact_limit) {
      continue;
    }
    std::uint64_t available = 0;
    bool required_available = required_artifact_slot == kNoFrozenArtifact;
    for (const QuotaRowState& row : scheduler->rows) {
      const std::uint64_t row_mask = ArtifactMask(row.cohort_artifact_slot);
      if (row_mask == 0 || (row_mask & mask) != 0) {
        available += row.remaining;
        required_available = required_available ||
            (row.frozen_artifact_slot == required_artifact_slot &&
             row.remaining > 0);
      }
    }
    if (required_available && available >= game_count) {
      return 1;
    }
  }
  return 0;
}

int32_t CgTrainQuotaSchedulerTake(
    CgTrainQuotaScheduler* scheduler, std::uint32_t game_count,
    std::uint32_t frozen_artifact_limit,
    std::uint32_t required_artifact_slot, std::uint32_t* output_row_indices,
    std::uint32_t output_capacity, std::uint32_t* output_count) {
  try {
    if (scheduler == nullptr || game_count == 0 || output_count == nullptr ||
        output_row_indices == nullptr || output_capacity < game_count ||
        frozen_artifact_limit == 0 ||
        frozen_artifact_limit > scheduler->artifact_count ||
        (required_artifact_slot != kNoFrozenArtifact &&
         required_artifact_slot >= scheduler->artifact_count)) {
      throw std::invalid_argument("quota take arguments are invalid");
    }
    const std::uint64_t required_mask = ArtifactMask(required_artifact_slot);
    const std::uint64_t subset_limit = UINT64_C(1) << scheduler->artifact_count;
    std::uint64_t best_mask = 0;
    MaskProgress best_progress;
    bool found = false;
    for (std::uint64_t mask = 0; mask < subset_limit; ++mask) {
      if ((mask & required_mask) != required_mask ||
          Popcount(mask) > frozen_artifact_limit) {
        continue;
      }
      std::uint64_t available = 0;
      bool required_available = required_artifact_slot == kNoFrozenArtifact;
      for (const QuotaRowState& row : scheduler->rows) {
        const std::uint64_t row_mask = ArtifactMask(row.cohort_artifact_slot);
        if (row_mask == 0 || (row_mask & mask) != 0) {
          available += row.remaining;
          required_available = required_available ||
              (row.frozen_artifact_slot == required_artifact_slot &&
               row.remaining > 0);
        }
      }
      if (!required_available || available < game_count) {
        continue;
      }
      const MaskProgress progress = CohortProgress(scheduler->rows, mask);
      if (!found || EarlierProgress(progress, mask, best_progress, best_mask)) {
        found = true;
        best_mask = mask;
        best_progress = progress;
      }
    }
    if (!found) {
      *output_count = 0;
      g_quota_scheduler_error = "no artifact-coherent quota shard is available";
      return CG_TRAIN_INSUFFICIENT_CAPACITY;
    }

    std::uint64_t eligible_initial = 0;
    std::uint64_t eligible_issued = 0;
    for (const QuotaRowState& row : scheduler->rows) {
      const std::uint64_t row_mask = ArtifactMask(row.cohort_artifact_slot);
      if (row_mask != 0 && (row_mask & best_mask) == 0) {
        continue;
      }
      eligible_initial += row.initial;
      eligible_issued += row.initial - row.remaining;
    }
    std::uint32_t written = 0;
    if (required_artifact_slot != kNoFrozenArtifact) {
      std::uint32_t selected = std::numeric_limits<std::uint32_t>::max();
      for (std::uint32_t index = 0; index < scheduler->rows.size(); ++index) {
        const QuotaRowState& row = scheduler->rows[index];
        const std::uint64_t row_mask = ArtifactMask(row.cohort_artifact_slot);
        if (row.remaining == 0 ||
            row.frozen_artifact_slot != required_artifact_slot ||
            (row_mask != 0 && (row_mask & best_mask) == 0)) {
          continue;
        }
        if (selected == std::numeric_limits<std::uint32_t>::max() ||
            GreaterRowDeficit(
                row, index, scheduler->rows[selected], selected,
                eligible_initial, eligible_issued + 1)) {
          selected = index;
        }
      }
      if (selected == std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("required quota artifact disappeared");
      }
      output_row_indices[written++] = selected;
      --scheduler->rows[selected].remaining;
      --scheduler->remaining;
      ++eligible_issued;
    }

    while (written < game_count) {
      std::uint32_t selected = std::numeric_limits<std::uint32_t>::max();
      for (std::uint32_t index = 0; index < scheduler->rows.size(); ++index) {
        const QuotaRowState& row = scheduler->rows[index];
        const std::uint64_t row_mask = ArtifactMask(row.cohort_artifact_slot);
        if (row.remaining == 0 ||
            (row_mask != 0 && (row_mask & best_mask) == 0)) {
          continue;
        }
        if (selected == std::numeric_limits<std::uint32_t>::max() ||
            GreaterRowDeficit(
                row, index, scheduler->rows[selected], selected,
                eligible_initial, eligible_issued + 1)) {
          selected = index;
        }
      }
      if (selected == std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("quota scheduler exhausted a validated shard");
      }
      output_row_indices[written++] = selected;
      --scheduler->rows[selected].remaining;
      --scheduler->remaining;
      ++eligible_issued;
    }
    *output_count = written;
    g_quota_scheduler_error.clear();
    return CG_TRAIN_OK;
  } catch (const std::exception& error) {
    g_quota_scheduler_error = error.what();
    return CG_TRAIN_INVALID_ARGUMENT;
  } catch (...) {
    g_quota_scheduler_error = "unknown quota scheduler take error";
    return CG_TRAIN_ENGINE_EXCEPTION;
  }
}

}  // extern "C"

#ifndef SRC_NATIVE_CG_TRAIN_QUOTA_SCHEDULER_H_
#define SRC_NATIVE_CG_TRAIN_QUOTA_SCHEDULER_H_

#include <stdint.h>

typedef struct CgTrainQuotaScheduler CgTrainQuotaScheduler;

typedef struct CgTrainQuotaRow {
  uint64_t game_count;
  // UINT32_MAX means that the row needs only the current policy.
  uint32_t frozen_artifact_slot;
  // UINT32_MAX leaves the row available to every artifact cohort.  A cohort
  // slot constrains scheduling only; it does not make the row use that frozen
  // policy.
  uint32_t cohort_artifact_slot;
} CgTrainQuotaRow;

#endif  // SRC_NATIVE_CG_TRAIN_QUOTA_SCHEDULER_H_

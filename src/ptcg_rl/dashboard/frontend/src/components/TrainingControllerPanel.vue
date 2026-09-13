<script setup lang="ts">
import { NAlert, NCard, NSelect } from 'naive-ui'
import { percent } from '../format'
import type {
  OutcomeStats,
  TrainingController,
  TrainingDeckStrengthPayload,
  TrainingEvidenceRange,
} from '../types'

defineProps<{
  evidence: TrainingDeckStrengthPayload
  range: TrainingEvidenceRange
  rangeOptions: Array<{ label: string; value: TrainingEvidenceRange }>
}>()

const emit = defineEmits<{
  'update:range': [range: TrainingEvidenceRange]
}>()

const controllerNames = [
  ['self_play', 'Self-play'],
  ['sentinel', 'Sentinel'],
  ['adaptive_history', 'Adaptive history'],
  ['scripted', 'Scripted'],
] as const satisfies ReadonlyArray<readonly [TrainingController, string]>

function outcomes(
  scores: Partial<Record<TrainingController, OutcomeStats>>,
  controller: TrainingController,
) {
  return scores[controller]
}
</script>

<template>
  <n-card class="work-card" :bordered="false">
    <template #header>
      <div class="section-title">
        <span>对手分层切片</span>
        <small>分别观察移动的训练对手与固定基线，避免聚合结果掩盖退化</small>
      </div>
    </template>
    <template #header-extra>
      <n-select
        :value="range"
        :options="rangeOptions"
        class="deck-filter-select"
        @update:value="emit('update:range', $event)"
      />
    </template>
    <n-alert
      v-if="!evidence.ranges[range].metadata.available"
      type="warning"
    >
      {{ evidence.ranges[range].metadata.unavailable_reason }}
    </n-alert>
    <div class="training-controller-list">
      <div
        v-for="standing in evidence.ranges[range].standings"
        :key="standing.deck_label"
        class="training-controller-row"
      >
        <div class="training-controller-deck">
          <strong>{{ standing.display_name }}</strong>
          <small class="mono">{{ standing.deck_hash }}</small>
        </div>
        <div
          v-for="[key, label] in controllerNames"
          :key="key"
          class="training-controller-cell"
        >
          <span>{{ label }}</span>
          <strong>{{ percent(outcomes(standing.controller_scores, key)?.score_rate ?? null, 1) }}</strong>
          <small>
            {{ outcomes(standing.controller_scores, key)?.games.toLocaleString() ?? 0 }} 局
            · {{ outcomes(standing.controller_scores, key)?.wins ?? 0 }}-{{ outcomes(standing.controller_scores, key)?.draws ?? 0 }}-{{ outcomes(standing.controller_scores, key)?.losses ?? 0 }}
          </small>
        </div>
      </div>
    </div>
  </n-card>
</template>

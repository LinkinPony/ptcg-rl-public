<script setup lang="ts">
import {
  NAlert,
  NDrawer,
  NDrawerContent,
  NSelect,
} from 'naive-ui'
import type {
  OpponentSet,
  TrainingController,
  TrainingDeckMatchupsPayload,
  TrainingEvidenceRange,
} from '../types'
import TrainingMatchupProfile from './TrainingMatchupProfile.vue'

defineProps<{
  show: boolean
  deckLabel: string | null
  deckHash: string | null
  displayName: string
  detail: TrainingDeckMatchupsPayload | null
  loading: boolean
  range: TrainingEvidenceRange
  controller: TrainingController
  seat: string
  opponentSet: OpponentSet
  rangeOptions: Array<{ label: string; value: TrainingEvidenceRange }>
  controllerOptions: Array<{ label: string; value: TrainingController }>
  seatOptions: Array<{ label: string; value: string }>
}>()

const emit = defineEmits<{
  'update:show': [show: boolean]
  'update:range': [range: TrainingEvidenceRange]
  'update:controller': [controller: TrainingController]
  'update:seat': [seat: string]
  'update:opponentSet': [opponentSet: OpponentSet]
}>()
</script>

<template>
  <n-drawer
    :show="show"
    width="min(1040px, 96vw)"
    placement="right"
    @update:show="emit('update:show', $event)"
  >
    <n-drawer-content v-if="deckLabel" closable>
      <template #header>
        <div class="drawer-title">
          <span>{{ displayName }}</span>
          <small class="mono">旧编号 · {{ deckHash ?? '不可用' }}</small>
        </div>
      </template>
      <div class="drawer-stack">
        <div class="deck-filter-row">
          <n-select
            :value="range"
            :options="rangeOptions"
            class="deck-filter-select"
            @update:value="emit('update:range', $event)"
          />
          <n-select
            :value="controller"
            :options="controllerOptions"
            class="deck-filter-select"
            @update:value="emit('update:controller', $event)"
          />
          <n-select
            :value="seat"
            :options="seatOptions"
            class="deck-filter-select"
            @update:value="emit('update:seat', $event)"
          />
          <n-select
            :value="opponentSet"
            :options="[
              { label: '当前训练 Roster', value: 'active' },
              { label: '全部训练对手', value: 'all' },
            ]"
            class="deck-filter-select"
            @update:value="emit('update:opponentSet', $event)"
          />
        </div>
        <n-alert v-if="detail && !detail.available" type="warning">
          {{ detail.unavailable_reason }}
        </n-alert>
        <training-matchup-profile :detail="detail" :loading="loading" />
      </div>
    </n-drawer-content>
  </n-drawer>
</template>

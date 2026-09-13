<script setup lang="ts">
import { computed } from 'vue'
import { NButton, NFlex, NSelect, NTag } from 'naive-ui'
import type { CheckpointInfo, RunInfo, WindowName } from '../types'

const props = defineProps<{
  runs: RunInfo[]
  run: string
  checkpoints: CheckpointInfo[]
  checkpoint: number | null
  window: WindowName
  loading: boolean
  showWindow?: boolean
}>()
const emit = defineEmits<{
  'update:run': [value: string]
  'update:checkpoint': [value: number | null]
  'update:window': [value: WindowName]
  refresh: []
}>()

const runOptions = computed(() => {
  const option = (run: RunInfo) => ({
    label: `${run.current ? 'CURRENT · ' : ''}${run.run_id}`,
    value: run.run_id,
  })
  const active = props.runs.filter((run) => run.current || run.data_state === 'ready')
  const lineaged = props.runs.filter(
    (run) => !active.includes(run) && run.lineage_id != null,
  )
  const history = props.runs.filter(
    (run) => !active.includes(run) && !lineaged.includes(run),
  )
  return [
    { type: 'group', label: 'Current / ready', key: 'active', children: active.map(option) },
    { type: 'group', label: 'Lineage history', key: 'lineage', children: lineaged.map(option) },
    { type: 'group', label: 'Standalone history', key: 'history', children: history.map(option) },
  ]
})

const checkpointOptions = computed(() =>
  props.checkpoints.map((checkpoint) => ({
    label: `v${checkpoint.version}${checkpoint.current ? ' · latest' : ''}${checkpoint.metric_available ? ' · metrics' : ''}`,
    value: checkpoint.version,
  })),
)

const windowOptions = [
  { label: '近 15 分钟', value: '15m' },
  { label: '近 60 分钟', value: '60m' },
  { label: '累计', value: 'cumulative' },
]
</script>

<template>
  <header class="context-bar">
    <div class="context-title">
      <span>ACTIVE CONTEXT</span>
      <strong>{{ runs.find((item) => item.run_id === run)?.display_name ?? run }}</strong>
    </div>
    <n-flex align="center" :wrap="false" class="context-controls">
      <n-tag
        size="small"
        round
        :type="runs.find((item) => item.run_id === run)?.data_state === 'ready' ? 'success' : 'warning'"
      >
        {{ runs.find((item) => item.run_id === run)?.data_state ?? 'loading' }}
      </n-tag>
      <n-select
        :value="run"
        :options="runOptions"
        filterable
        virtual-scroll
        class="context-run"
        @update:value="emit('update:run', $event)"
      />
      <n-select
        :value="checkpoint"
        :options="checkpointOptions"
        clearable
        placeholder="latest checkpoint"
        class="context-checkpoint"
        @update:value="emit('update:checkpoint', $event)"
      />
      <n-select
        v-if="showWindow !== false"
        :value="window"
        :options="windowOptions"
        class="context-window"
        @update:value="emit('update:window', $event)"
      />
      <n-button quaternary :loading="loading" @click="emit('refresh')">刷新</n-button>
    </n-flex>
  </header>
</template>

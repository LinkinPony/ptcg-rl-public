<script setup lang="ts">
import {
  computed,
  defineAsyncComponent,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from 'vue'
import {
  darkTheme,
  NAlert,
  NConfigProvider,
  NSpin,
} from 'naive-ui'
import { api } from './api'
import RunContextBar from './components/RunContextBar.vue'
import WorkbenchNav from './components/WorkbenchNav.vue'
import type {
  CheckpointInfo,
  DashboardSession,
  LearnerSeriesPayload,
  OpponentAllocationSummary,
  RunInfo,
  WindowName,
  TrainingDeckStrengthPayload,
  WorkbenchSummary,
  WorkbenchView,
} from './types'

const NowPage = defineAsyncComponent(() => import('./pages/NowPage.vue'))
const LearningPage = defineAsyncComponent(() => import('./pages/LearningPage.vue'))
const AllocationPage = defineAsyncComponent(() => import('./pages/AllocationPage.vue'))
const DecksPage = defineAsyncComponent(() => import('./pages/DecksPage.vue'))
const ComparePage = defineAsyncComponent(() => import('./pages/ComparePage.vue'))
const TasksPage = defineAsyncComponent(() => import('./pages/TasksPage.vue'))
const LeaguePage = defineAsyncComponent(() => import('./pages/LeaguePage.vue'))

const views: WorkbenchView[] = [
  'now',
  'learning',
  'allocation',
  'decks',
  'compare',
  'tasks',
  'league',
]
const windows: WindowName[] = ['cumulative', '15m', '60m']
const query = new URLSearchParams(window.location.search)

const runs = ref<RunInfo[]>([])
const selectedRun = ref(query.get('run') ?? '')
const selectedCheckpoint = ref<number | null>(
  query.get('checkpoint') == null ? null : Number(query.get('checkpoint')),
)
const windowName = ref<WindowName>(
  windows.includes(query.get('window') as WindowName)
    ? query.get('window') as WindowName
    : '15m',
)
const view = ref<WorkbenchView>(
  views.includes(query.get('view') as WorkbenchView)
    ? query.get('view') as WorkbenchView
    : 'now',
)
const checkpoints = ref<CheckpointInfo[]>([])
const session = ref<DashboardSession | null>(null)
const summary = ref<WorkbenchSummary | null>(null)
const learnerSeries = ref<LearnerSeriesPayload | null>(null)
const allocationSummary = ref<OpponentAllocationSummary | null>(null)
const deckStrength = ref<TrainingDeckStrengthPayload | null>(null)
const loading = ref(false)
const error = ref('')
const deckRefreshVersion = ref(0)
let timer: number | undefined
let generation = 0
let initialized = false
let disposed = false
let polling = false

const blockingLoading = computed(() => {
  if (!loading.value) return false
  if (view.value === 'now') {
    return summary.value?.run.run_id !== selectedRun.value
  }
  if (view.value === 'learning') {
    return summary.value?.run.run_id !== selectedRun.value
      || learnerSeries.value?.run_id !== selectedRun.value
  }
  if (view.value === 'allocation') {
    return allocationSummary.value?.run_id !== selectedRun.value
  }
  if (view.value === 'decks') {
    return deckStrength.value?.run_id !== selectedRun.value
      || deckStrength.value?.checkpoint_version !== selectedCheckpoint.value
  }
  return false
})

const themeOverrides = {
  common: {
    primaryColor: '#63e6be',
    primaryColorHover: '#8cefd1',
    primaryColorPressed: '#38d9a9',
    borderRadius: '8px',
    fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
  },
  Card: { borderRadius: '12px' },
  Select: { peers: { InternalSelection: { borderRadius: '8px' } } },
}

onMounted(async () => {
  loading.value = true
  try {
    ;[runs.value, session.value] = await Promise.all([api.runs(), api.session()])
    if (!runs.value.some((run) => run.run_id === selectedRun.value)) {
      selectedRun.value = runs.value.find((run) => run.current)?.run_id
        ?? runs.value[0]?.run_id
        ?? ''
    }
    await loadCheckpoints()
    await refresh({ refreshDeckTrend: false })
    initialized = true
    schedulePoll()
    document.addEventListener('visibilitychange', handleVisibilityChange)
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : String(reason)
  } finally {
    loading.value = false
  }
})
onBeforeUnmount(() => {
  disposed = true
  window.clearTimeout(timer)
  document.removeEventListener('visibilitychange', handleVisibilityChange)
})

watch([selectedRun, windowName, view], async ([nextRun], [previousRun]) => {
  if (!initialized) return
  updateUrl()
  if (nextRun !== previousRun) await loadCheckpoints()
  await refresh({ refreshDeckTrend: false })
})
watch(selectedCheckpoint, async () => {
  updateUrl()
  if (initialized && view.value === 'decks') {
    await refresh({ refreshDeckTrend: false })
  }
})

async function loadCheckpoints(reset = true) {
  if (!selectedRun.value) return
  const next = await api.checkpoints(selectedRun.value)
  checkpoints.value = next
  const exists = next.some((checkpoint) => checkpoint.version === selectedCheckpoint.value)
  if ((reset || !exists) && !exists) {
    selectedCheckpoint.value = next.find((checkpoint) => checkpoint.current)?.version
      ?? next[0]?.version
      ?? null
  }
}

async function refresh(
  options: { refreshDeckTrend?: boolean } = {},
) {
  if (!selectedRun.value || view.value === 'compare' || view.value === 'tasks') return
  const requestGeneration = ++generation
  loading.value = true
  error.value = ''
  try {
    if (view.value === 'league') {
      return
    } else if (view.value === 'now') {
      const next = await api.summary(selectedRun.value, windowName.value)
      if (requestGeneration === generation) summary.value = next
    } else if (view.value === 'learning') {
      const [nextSummary, nextSeries] = await Promise.all([
        api.summary(selectedRun.value, windowName.value),
        api.learnerSeries(selectedRun.value),
      ])
      if (requestGeneration === generation) {
        summary.value = nextSummary
        learnerSeries.value = nextSeries
      }
    } else if (view.value === 'allocation') {
      const next = await api.opponentAllocation(selectedRun.value)
      if (requestGeneration === generation) allocationSummary.value = next
    } else if (view.value === 'decks') {
      const next = await api.trainingDeckStrength(
        selectedRun.value,
        selectedCheckpoint.value,
      )
      if (requestGeneration === generation) {
        deckStrength.value = next
        if (options.refreshDeckTrend !== false) {
          deckRefreshVersion.value += 1
        }
      }
    }
  } catch (reason) {
    if (requestGeneration === generation) {
      error.value = reason instanceof Error ? reason.message : String(reason)
    }
  } finally {
    if (requestGeneration === generation) loading.value = false
  }
}

function schedulePoll(delay?: number) {
  window.clearTimeout(timer)
  if (disposed) return
  timer = window.setTimeout(
    () => void poll(),
    delay ?? (session.value?.refresh_seconds ?? 15) * 1_000,
  )
}

async function poll() {
  if (
    polling
    || document.hidden
    || view.value === 'compare'
    || view.value === 'tasks'
    || view.value === 'league'
  ) {
    schedulePoll()
    return
  }
  polling = true
  try {
    const [nextRuns] = await Promise.all([
      api.runs(),
      loadCheckpoints(false),
    ])
    runs.value = nextRuns
    await refresh()
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : String(reason)
  } finally {
    polling = false
    schedulePoll()
  }
}

function handleVisibilityChange() {
  if (!document.hidden) schedulePoll(0)
}

function updateUrl() {
  const params = new URLSearchParams()
  params.set('view', view.value)
  if (selectedRun.value) params.set('run', selectedRun.value)
  params.set('window', windowName.value)
  if (selectedCheckpoint.value != null) {
    params.set('checkpoint', `${selectedCheckpoint.value}`)
  }
  window.history.replaceState(null, '', `${window.location.pathname}?${params}`)
}
</script>

<template>
  <n-config-provider :theme="darkTheme" :theme-overrides="themeOverrides">
    <div class="workbench-shell">
      <workbench-nav v-model="view" />
      <div class="workbench-main">
        <run-context-bar
          v-if="view !== 'league'"
          :runs="runs"
          :run="selectedRun"
          :checkpoints="checkpoints"
          :checkpoint="selectedCheckpoint"
          :window="windowName"
          :loading="loading"
          :show-window="view !== 'decks'"
          @update:run="selectedRun = $event"
          @update:checkpoint="selectedCheckpoint = $event"
          @update:window="windowName = $event"
          @refresh="refresh"
        />
        <main class="workbench-content">
          <n-alert v-if="error" type="error" title="数据或任务请求失败" class="global-alert">
            {{ error }}
          </n-alert>
          <n-spin :show="blockingLoading" class="page-spinner">
            <now-page v-if="view === 'now'" :summary="summary" :loading="blockingLoading" />
            <learning-page
              v-else-if="view === 'learning'"
              :summary="summary"
              :series="learnerSeries"
              :checkpoint="selectedCheckpoint"
            />
            <allocation-page
              v-else-if="view === 'allocation'"
              :run="selectedRun"
              :summary="allocationSummary"
            />
            <decks-page
              v-else-if="view === 'decks'"
              :run="selectedRun"
              :checkpoint="checkpoints.find((item) => item.version === selectedCheckpoint) ?? null"
              :evidence="deckStrength"
              :refresh-version="deckRefreshVersion"
            />
            <compare-page
              v-else-if="view === 'compare'"
              :runs="runs"
              :current-run="selectedRun"
              :current-checkpoint="selectedCheckpoint"
            />
            <tasks-page v-else-if="view === 'tasks'" :session="session" />
            <league-page v-else />
          </n-spin>
        </main>
      </div>
    </div>
  </n-config-provider>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import {
  NAlert,
  NButton,
  NCard,
  NDrawer,
  NDrawerContent,
  NInput,
  NInputNumber,
  NModal,
  NProgress,
  NSelect,
  NTabPane,
  NTabs,
  NTag,
} from 'naive-ui'
import { api } from '../api'
import DeckSelectionMatrix from '../components/DeckSelectionMatrix.vue'
import { compactHash, dateTime } from '../format'
import type {
  DashboardSession,
  DeckSelectionPayload,
  TaskArtifact,
  TaskArtifactContent,
  TaskCatalog,
  TaskCatalogItem,
  TaskCreateRequest,
  TaskExecution,
  TaskKind,
  TaskLogPayloadV2,
  TaskReceipt,
  TaskResultSummary,
  TaskTablePayload,
} from '../types'

const props = defineProps<{ session: DashboardSession | null }>()

const catalog = ref<TaskCatalog | null>(null)
const tasks = ref<TaskReceipt[]>([])
const totalTasks = ref(0)
const step = ref(1)
const kind = ref<TaskKind>('bundle_strength')
const mode = ref<'formal' | 'diagnostic'>('formal')
const label = ref('正式 Bundle 强度评测')
const outputLabel = ref('bundle-strength')
const execution = ref<TaskExecution>('distributed')
const games = ref(20)
const sideObservationsId = ref('')
const candidateIds = ref<string[]>([])
const opponentIds = ref<string[]>([])
const checkpointId = ref('')
const deckIds = ref<string[]>([])
const publicCatalogId = ref('')
const runtimeTemplateId = ref('')
const h2hCandidateId = ref('')
const h2hOpponentId = ref('')
const submissionProfileId = ref('')
const trainingProfileId = ref('')
const confirmStart = ref(false)
const loading = ref(false)
const error = ref('')

const selectedTask = ref<TaskReceipt | null>(null)
const result = ref<TaskResultSummary | null>(null)
const table = ref<TaskTablePayload | null>(null)
const selectedTable = ref('')
const artifacts = ref<TaskArtifact[]>([])
const report = ref<TaskArtifactContent | null>(null)
const log = ref<TaskLogPayloadV2 | null>(null)
const detailTab = ref('overview')
const runtimeLadder = ref<DeckSelectionPayload | null>(null)
const highlightedRuntimeDeck = ref<string | null>(null)
let timer: number | undefined
let refreshRunning = false

const workflow = computed(() =>
  catalog.value?.workflows.find((item) => item.kind === kind.value) ?? null,
)
const releaseOptions = computed(() => options(catalog.value?.releases))
const checkpointOptions = computed(() => options(catalog.value?.checkpoints))
const deckOptions = computed(() => options(catalog.value?.decks))
const publicCatalogOptions = computed(() => options(catalog.value?.public_catalogs))
const runtimeOptions = computed(() => options(catalog.value?.runtime_templates))
const sideOptions = computed(() => options(catalog.value?.side_observations))
const registeredOptions = computed(() => options(catalog.value?.registered_opponents))
const opponentOptions = computed(() => [
  ...releaseOptions.value,
  ...registeredOptions.value,
])
const submissionOptions = computed(() => options(catalog.value?.submission_profiles))
const trainingOptions = computed(() => options(catalog.value?.training_profiles))
const executionOptions = computed(() => {
  if (kind.value === 'runtime_elo') return [{ label: '本机 CUDA（训练时自动排队）', value: 'local' }]
  if (kind.value === 'release_h2h') {
    return [
      { label: '远端分布式', value: 'distributed' },
      { label: '本机', value: 'local' },
    ]
  }
  return [
    { label: '本机 + 远端分布式', value: 'distributed' },
    { label: '仅远端', value: 'remote' },
    { label: '仅本机', value: 'local' },
  ]
})
const activeTasks = computed(() =>
  tasks.value.filter((task) => ['queued', 'starting', 'running', 'cancelling'].includes(task.state)),
)
const terminalTasks = computed(() =>
  tasks.value.filter((task) => !['queued', 'starting', 'running', 'cancelling'].includes(task.state)),
)
const estimatedGames = computed(() => {
  if (kind.value === 'bundle_strength') {
    const candidates = mode.value === 'formal' ? candidateIds.value.length : (checkpointId.value && deckIds.value.length ? 1 : 0)
    return candidates * opponentIds.value.length * games.value
  }
  if (kind.value === 'release_h2h') return games.value
  if (kind.value === 'runtime_elo') {
    return (deckIds.value.length * (deckIds.value.length - 1) / 2) * games.value
  }
  return null
})
const canContinue = computed(() => {
  if (step.value === 1) return true
  if (step.value === 2) return label.value.trim().length > 0
  if (kind.value === 'bundle_strength') {
    const candidateReady = mode.value === 'formal'
      ? candidateIds.value.length > 0
      : Boolean(checkpointId.value && deckIds.value.length === 1 && publicCatalogId.value && runtimeTemplateId.value)
    return candidateReady && opponentIds.value.length > 0 && Boolean(sideObservationsId.value)
  }
  if (kind.value === 'release_h2h') {
    return Boolean(h2hCandidateId.value && h2hOpponentId.value && h2hCandidateId.value !== h2hOpponentId.value)
  }
  if (kind.value === 'runtime_elo') {
    return Boolean(checkpointId.value && deckIds.value.length >= 2 && publicCatalogId.value && runtimeTemplateId.value)
  }
  if (kind.value === 'package_validation') return Boolean(submissionProfileId.value)
  return Boolean(trainingProfileId.value)
})
const reviewLines = computed(() => {
  const lines = [
    ['任务', workflow.value?.label ?? kind.value],
    ['语义通道', modeLabel()],
    ['执行', executionLabel()],
  ]
  if (estimatedGames.value != null) lines.push(['预计对局', `${estimatedGames.value}`])
  lines.push(['Receipt', '启动前原子发布；可取消、重试和审计'])
  return lines
})

watch(kind, resetForKind)
watch(mode, () => {
  if (kind.value === 'bundle_strength') {
    candidateIds.value = []
    checkpointId.value = ''
    deckIds.value = []
  }
})
watch(selectedTable, async (name) => {
  if (selectedTask.value && name) await loadTable(selectedTask.value.task_id, name)
})
watch(detailTab, async (name) => {
  if (name === 'artifacts' && selectedTask.value && isTerminal(selectedTask.value) && artifacts.value.length === 0) {
    const next = await api.taskArtifacts(selectedTask.value.task_id)
    artifacts.value = next.artifacts
  }
})

onMounted(async () => {
  await load()
  timer = window.setInterval(refresh, 3_000)
})
onBeforeUnmount(() => window.clearInterval(timer))

async function load() {
  error.value = ''
  try {
    const [nextCatalog, history] = await Promise.all([api.taskCatalog(), api.tasks()])
    catalog.value = nextCatalog
    tasks.value = history.tasks
    totalTasks.value = history.total
    initializeDefaults()
  } catch (reason) {
    error.value = message(reason)
  }
}

async function refresh() {
  if (refreshRunning || document.hidden) return
  refreshRunning = true
  try {
    const history = await api.tasks()
    tasks.value = history.tasks
    totalTasks.value = history.total
    if (selectedTask.value) {
      selectedTask.value = await api.task(selectedTask.value.task_id)
      log.value = await api.taskLog(selectedTask.value.task_id)
      result.value = await api.taskResult(selectedTask.value.task_id)
      if (isTerminal(selectedTask.value) && result.value?.tables.length && !selectedTable.value) {
        await loadTerminalResult(selectedTask.value)
      }
    }
    if (activeTasks.value.length > 0) catalog.value = await api.taskCatalog()
  } catch {
    // Durable data from the previous successful refresh remains visible.
  } finally {
    refreshRunning = false
  }
}

function initializeDefaults() {
  const current = catalog.value
  if (!current) return
  sideObservationsId.value ||= current.side_observations[0]?.artifact_id ?? ''
  publicCatalogId.value ||= current.public_catalogs[0]?.artifact_id ?? ''
  runtimeTemplateId.value ||= current.runtime_templates[0]?.artifact_id ?? ''
  submissionProfileId.value ||= current.submission_profiles[0]?.artifact_id ?? ''
  trainingProfileId.value ||= current.training_profiles[0]?.artifact_id ?? ''
}

function resetForKind(next: TaskKind) {
  step.value = 2
  mode.value = next === 'bundle_strength' ? 'formal' : next === 'runtime_elo' ? 'diagnostic' : 'formal'
  execution.value = next === 'runtime_elo' ? 'local' : 'distributed'
  games.value = next === 'release_h2h' ? 400 : next === 'runtime_elo' ? 4 : 20
  label.value = {
    bundle_strength: '正式 Bundle 强度评测',
    release_h2h: 'Release H2H',
    runtime_elo: 'Checkpoint 牌组 Elo 诊断',
    package_validation: 'Submission 包校验',
    config_dry_run: '训练配置 Dry-run',
  }[next]
  outputLabel.value = {
    bundle_strength: 'bundle-strength',
    release_h2h: 'release-h2h',
    runtime_elo: 'runtime-elo',
    package_validation: 'package-validation',
    config_dry_run: 'config-dry-run',
  }[next]
}

function nextStep() {
  if (!canContinue.value) return
  if (step.value < 4) step.value += 1
  else confirmStart.value = true
}

async function start() {
  const token = props.session?.request_token
  if (!token) return
  let payload: TaskCreateRequest
  try {
    payload = buildPayload()
  } catch (reason) {
    error.value = message(reason)
    return
  }
  loading.value = true
  error.value = ''
  try {
    const receipt = await api.startTask(payload, token)
    tasks.value = [receipt, ...tasks.value]
    totalTasks.value += 1
    confirmStart.value = false
    await inspect(receipt)
  } catch (reason) {
    error.value = message(reason)
  } finally {
    loading.value = false
  }
}

function buildPayload(): TaskCreateRequest {
  const common = { label: label.value.trim() }
  if (kind.value === 'bundle_strength') {
    const candidates = mode.value === 'formal'
      ? candidateIds.value.map((artifact_id) => ({ source: 'release_bundle' as const, artifact_id }))
      : [{
          source: 'checkpoint' as const,
          artifact_id: checkpointId.value,
          deck_id: deckIds.value[0],
          public_catalog_id: publicCatalogId.value,
          runtime_template_id: runtimeTemplateId.value,
        }]
    return {
      ...common,
      kind: 'bundle_strength',
      mode: mode.value,
      candidates,
      opponents: opponentIds.value.map((artifact_id) => ({
        source: artifact_id.startsWith('registered:')
          ? 'registered_opponent' as const
          : 'release_bundle' as const,
        artifact_id,
      })),
      games_per_matchup: evenGames(),
      execution: execution.value,
      side_observations_id: sideObservationsId.value,
      output_label: outputLabel.value,
    }
  }
  if (kind.value === 'release_h2h') {
    return {
      ...common,
      kind: 'release_h2h',
      mode: 'formal',
      candidate_id: h2hCandidateId.value,
      opponent_id: h2hOpponentId.value,
      games: evenGames(),
      execution: execution.value,
      output_label: outputLabel.value,
    }
  }
  if (kind.value === 'runtime_elo') {
    return {
      ...common,
      kind: 'runtime_elo',
      mode: 'diagnostic',
      checkpoint_id: checkpointId.value,
      deck_ids: deckIds.value,
      public_catalog_id: publicCatalogId.value,
      runtime_template_id: runtimeTemplateId.value,
      games_per_pair: evenGames(),
      execution: 'local',
      output_label: outputLabel.value,
    }
  }
  if (kind.value === 'package_validation') {
    return {
      ...common,
      kind: 'package_validation',
      mode: 'utility',
      submission_profile_id: submissionProfileId.value,
      output_label: outputLabel.value,
    }
  }
  return {
    ...common,
    kind: 'config_dry_run',
    mode: 'utility',
    training_profile_id: trainingProfileId.value,
  }
}

async function inspect(task: TaskReceipt) {
  selectedTask.value = task
  result.value = null
  table.value = null
  selectedTable.value = ''
  artifacts.value = []
  report.value = null
  runtimeLadder.value = null
  highlightedRuntimeDeck.value = null
  detailTab.value = 'overview'
  try {
    const [nextResult, nextLog] = await Promise.all([
      api.taskResult(task.task_id),
      api.taskLog(task.task_id),
    ])
    result.value = nextResult
    log.value = nextLog
    if (isTerminal(task)) await loadTerminalResult(task)
  } catch (reason) {
    error.value = message(reason)
  }
}

async function loadTerminalResult(task: TaskReceipt) {
  if (task.kind === 'runtime_elo' && task.state === 'succeeded') {
    runtimeLadder.value = await api.runtimeLadderResult(task.task_id)
  }
  if (result.value?.tables.length) {
    selectedTable.value ||= result.value.tables[0]
  }
  if (result.value?.report_artifact_id) {
    report.value = await api.taskArtifactContent(task.task_id, result.value.report_artifact_id)
  }
}

async function loadTable(taskId: string, name: string, offset = 0) {
  try {
    table.value = await api.taskTable(taskId, name, offset, 50)
  } catch (reason) {
    error.value = message(reason)
  }
}

async function cancel(task: TaskReceipt) {
  const token = props.session?.request_token
  if (!token || !window.confirm(`确认取消任务 ${task.label}？已提交产物不会删除。`)) return
  try {
    const updated = await api.cancelTask(task.task_id, token)
    replaceTask(updated)
    if (selectedTask.value?.task_id === updated.task_id) selectedTask.value = updated
  } catch (reason) {
    error.value = message(reason)
  }
}

async function retry(task: TaskReceipt) {
  const token = props.session?.request_token
  if (!token) return
  try {
    const receipt = await api.retryTask(task.task_id, token)
    tasks.value = [receipt, ...tasks.value]
    await inspect(receipt)
  } catch (reason) {
    error.value = message(reason)
  }
}

function replaceTask(updated: TaskReceipt) {
  tasks.value = tasks.value.map((task) => task.task_id === updated.task_id ? updated : task)
}

function isTerminal(task: TaskReceipt) {
  return !['queued', 'starting', 'running', 'cancelling'].includes(task.state)
}

function chooseWorkflow(next: TaskKind) {
  kind.value = next
  step.value = 2
}

function options(items?: TaskCatalogItem[]) {
  return (items ?? []).map((item) => ({
    label: item.detail ? `${item.label} · ${item.detail}` : item.label,
    value: item.artifact_id,
    disabled: !item.available,
  }))
}

function evenGames() {
  const value = Math.max(2, Math.floor(games.value || 2))
  return value % 2 === 0 ? value : value + 1
}

function modeLabel() {
  if (kind.value === 'runtime_elo') return '诊断：共享 checkpoint，不代表可部署强度'
  if (kind.value === 'package_validation' || kind.value === 'config_dry_run') return '工具：不产生竞赛结论'
  return mode.value === 'formal' ? '正式：不可变部署证据' : '诊断：裸 checkpoint 证据'
}

function executionLabel() {
  return executionOptions.value.find((item) => item.value === execution.value)?.label ?? '本机'
}

function stateType(state: TaskReceipt['state']) {
  if (state === 'succeeded') return 'success'
  if (state === 'failed' || state === 'unknown') return 'error'
  if (state === 'cancelled') return 'default'
  return 'warning'
}

function scalar(value: unknown) {
  if (value == null) return '—'
  if (typeof value === 'number') return Number.isInteger(value) ? `${value}` : value.toFixed(4)
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function bytes(value: number) {
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KiB`
  if (value < 1024 * 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MiB`
  return `${(value / 1024 / 1024 / 1024).toFixed(2)} GiB`
}

function message(reason: unknown) {
  return reason instanceof Error ? reason.message : String(reason)
}
</script>

<template>
  <section class="page-stack">
    <div class="page-heading">
      <div>
        <span class="page-eyebrow">RESOURCE-AWARE EVALUATION WORKBENCH</span>
        <h1>评测与校验任务</h1>
        <p>从可验证工件中选择证据，任务自动排队；正式评测和诊断结果始终分开解释。</p>
      </div>
      <div class="task-heading-tags">
        <n-tag :type="session?.actions_enabled ? 'success' : 'warning'" round>
          {{ session?.actions_enabled ? 'ACTIONS ENABLED' : 'READ ONLY' }}
        </n-tag>
        <n-tag v-if="catalog?.resources.training_active" type="warning" round>
          TRAINING ACTIVE · LOCAL HEAVY QUEUED
        </n-tag>
      </div>
    </div>

    <n-alert v-if="error" type="error" closable @close="error = ''">{{ error }}</n-alert>
    <n-alert v-if="catalog" :type="catalog.resources.training_active ? 'warning' : 'info'">
      {{ catalog.resources.detail }}
    </n-alert>

    <div class="task-workbench-grid">
      <n-card class="work-card task-wizard-card" :bordered="false">
        <template #header>
          <div class="section-title">
            <span>创建任务</span>
            <small>第 {{ step }} / 4 步</small>
          </div>
        </template>

        <div class="wizard-steps">
          <button v-for="index in 4" :key="index" :class="{ active: step === index, done: step > index }" @click="index <= step && (step = index)">
            <span>{{ index }}</span>
            {{ ['任务类型', '证据语义', '输入工件', '工作量确认'][index - 1] }}
          </button>
        </div>

        <div v-if="step === 1" class="workflow-grid">
          <button
            v-for="item in catalog?.workflows ?? []"
            :key="item.kind"
            class="workflow-choice"
            :class="{ selected: kind === item.kind }"
            @click="chooseWorkflow(item.kind)"
          >
            <strong>{{ item.label }}</strong>
            <span>{{ item.description }}</span>
            <small>{{ item.modes.map((value) => value.toUpperCase()).join(' / ') }}</small>
          </button>
        </div>

        <div v-else-if="step === 2" class="wizard-form">
          <div class="wizard-explainer">
            <strong>{{ workflow?.label }}</strong>
            <span>{{ workflow?.description }}</span>
          </div>
          <label>
            <span>任务名称</span>
            <n-input v-model:value="label" maxlength="120" />
          </label>
          <label v-if="kind === 'bundle_strength'">
            <span>证据语义</span>
            <n-select v-model:value="mode" :options="[
              { label: '正式 · 不可变 deployment bundle', value: 'formal' },
              { label: '诊断 · 裸 checkpoint + 精确 route', value: 'diagnostic' },
            ]" />
          </label>
          <label v-if="!['package_validation', 'config_dry_run'].includes(kind)">
            <span>执行位置</span>
            <n-select v-model:value="execution" :options="executionOptions" />
          </label>
          <n-alert :type="mode === 'formal' ? 'success' : 'warning'">
            {{ modeLabel() }}
          </n-alert>
        </div>

        <div v-else-if="step === 3" class="wizard-form">
          <template v-if="kind === 'bundle_strength'">
            <template v-if="mode === 'formal'">
              <label>
                <span>候选部署 Bundle（可多选）</span>
                <n-select v-model:value="candidateIds" multiple filterable :options="releaseOptions" />
              </label>
            </template>
            <template v-else>
              <label>
                <span>诊断 Checkpoint</span>
                <n-select v-model:value="checkpointId" filterable :options="checkpointOptions" />
              </label>
              <label>
                <span>精确牌组 Route</span>
                <n-select v-model:value="deckIds" multiple :max-tag-count="2" filterable :options="deckOptions" />
                <small>Bundle 诊断一次选择一个牌组；后端会校验该 deck digest 是否存在于 checkpoint。</small>
              </label>
              <label>
                <span>Public catalog</span>
                <n-select v-model:value="publicCatalogId" :options="publicCatalogOptions" />
              </label>
              <label>
                <span>Runtime 模板</span>
                <n-select v-model:value="runtimeTemplateId" :options="runtimeOptions" />
              </label>
            </template>
            <label>
              <span>统一对手支撑集（Release / 注册公开对手）</span>
              <n-select v-model:value="opponentIds" multiple filterable :options="opponentOptions" />
            </label>
            <label>
              <span>Meta side observations</span>
              <n-select v-model:value="sideObservationsId" filterable :options="sideOptions" />
            </label>
          </template>

          <template v-else-if="kind === 'release_h2h'">
            <label><span>Candidate release</span><n-select v-model:value="h2hCandidateId" filterable :options="releaseOptions" /></label>
            <label><span>Opponent release</span><n-select v-model:value="h2hOpponentId" filterable :options="releaseOptions" /></label>
          </template>

          <template v-else-if="kind === 'runtime_elo'">
            <label><span>诊断 Checkpoint</span><n-select v-model:value="checkpointId" filterable :options="checkpointOptions" /></label>
            <label><span>精确牌组（至少 2 个）</span><n-select v-model:value="deckIds" multiple filterable :options="deckOptions" /></label>
            <label><span>Public catalog</span><n-select v-model:value="publicCatalogId" :options="publicCatalogOptions" /></label>
            <label><span>Runtime 模板</span><n-select v-model:value="runtimeTemplateId" :options="runtimeOptions" /></label>
          </template>

          <label v-else-if="kind === 'package_validation'">
            <span>Submission profile</span>
            <n-select v-model:value="submissionProfileId" filterable :options="submissionOptions" />
          </label>
          <label v-else>
            <span>训练 Profile</span>
            <n-select v-model:value="trainingProfileId" filterable :options="trainingOptions" />
          </label>
        </div>

        <div v-else class="wizard-form">
          <label v-if="!['package_validation', 'config_dry_run'].includes(kind)">
            <span>{{ kind === 'release_h2h' ? '总对局数' : kind === 'runtime_elo' ? '每个牌组对局对的局数' : '每个 Candidate × Opponent 的对局数' }}</span>
            <n-input-number v-model:value="games" :min="2" :step="2" />
            <small>镜像先后手要求偶数；提交时会自动向上修正为偶数。</small>
          </label>
          <label v-if="kind !== 'config_dry_run'">
            <span>输出标签</span>
            <n-input v-model:value="outputLabel" placeholder="safe-artifact-label" />
            <small>仅允许字母、数字、点、下划线和连字符，且不能包含 latest。</small>
          </label>
          <div class="review-grid">
            <div v-for="[name, value] in reviewLines" :key="name"><span>{{ name }}</span><strong>{{ value }}</strong></div>
          </div>
          <n-alert v-if="catalog?.resources.training_active && execution !== 'remote'" type="warning">
            当前训练不会被打断。若该任务属于本机重任务或 CUDA 任务，receipt 会先进入队列，训练结束或资源释放后自动启动。
          </n-alert>
        </div>

        <div class="wizard-actions">
          <n-button v-if="step > 1" @click="step -= 1">上一步</n-button>
          <span />
          <n-button
            type="primary"
            :disabled="!canContinue || !session?.actions_enabled"
            @click="nextStep"
          >
            {{ step === 4 ? '最终检查并启动' : '继续' }}
          </n-button>
        </div>
      </n-card>

      <div class="task-queue-column">
        <n-card class="work-card" :bordered="false">
          <template #header><div class="section-title"><span>正在进行</span><small>{{ activeTasks.length }} active</small></div></template>
          <div v-if="activeTasks.length" class="task-card-list">
            <button v-for="task in activeTasks" :key="task.task_id" class="task-history-card task-history-card--active" @click="inspect(task)">
              <div><strong>{{ task.label }}</strong><n-tag size="small" :type="stateType(task.state)">{{ task.state }}</n-tag></div>
              <span>{{ task.progress?.phase }} · {{ task.progress?.detail ?? task.queue_reason ?? task.resource_class }}</span>
              <n-progress
                v-if="task.progress?.percent != null"
                type="line"
                :percentage="task.progress.percent"
                :show-indicator="false"
                :height="6"
              />
              <small class="mono">{{ compactHash(task.task_id) }} · attempt {{ task.attempt }}</small>
            </button>
          </div>
          <div v-else class="empty-state"><strong>没有活动任务</strong><span>本机训练资源保持优先。</span></div>
        </n-card>

        <n-card class="work-card" :bordered="false">
          <template #header><div class="section-title"><span>安全边界</span><small>server enforced</small></div></template>
          <ul class="task-boundary-list">
            <li>浏览器只提交 catalog ID，不提交路径或 shell。</li>
            <li>正式评测候选必须是完整不可变 release bundle。</li>
            <li>裸 checkpoint 只能进入诊断通道，并校验精确 deck route。</li>
            <li>包校验永远不调用 submission；任务系统不控制训练。</li>
          </ul>
        </n-card>
      </div>
    </div>

    <n-card class="work-card" :bordered="false">
      <template #header>
        <div class="section-title"><span>任务历史</span><small>{{ totalTasks }} receipts · durable across restart</small></div>
      </template>
      <div v-if="terminalTasks.length" class="task-history-table">
        <button v-for="task in terminalTasks" :key="task.task_id" class="task-history-row" @click="inspect(task)">
          <span class="mono">{{ compactHash(task.task_id) }}</span>
          <div><strong>{{ task.label }}</strong><small>{{ task.kind }} · {{ task.mode }}</small></div>
          <n-tag size="small" :type="stateType(task.state)">{{ task.state }}</n-tag>
          <span>{{ dateTime(task.created_at_utc) }}</span>
          <span>{{ task.detail ?? task.output_dir ?? '—' }}</span>
        </button>
      </div>
      <div v-else class="empty-state"><strong>尚无已完成任务</strong><span>新的结果会在这里形成可追溯分析入口。</span></div>
    </n-card>

    <n-modal v-model:show="confirmStart" preset="card" class="confirm-card" title="最终检查">
      <div class="confirmation">
        <div class="review-grid">
          <div v-for="[name, value] in reviewLines" :key="name"><span>{{ name }}</span><strong>{{ value }}</strong></div>
        </div>
        <n-alert type="info">
          确认后立即发布 receipt。资源不足时任务会排队，不会绕过当前训练；这不是一个额外的预检任务。
        </n-alert>
        <div class="confirmation__actions">
          <n-button @click="confirmStart = false">返回修改</n-button>
          <n-button type="primary" :loading="loading" @click="start">确认并发布 Receipt</n-button>
        </div>
      </div>
    </n-modal>

    <n-drawer
      :show="selectedTask != null"
      width="min(1120px, 96vw)"
      placement="right"
      @update:show="!$event && (selectedTask = null)"
    >
      <n-drawer-content v-if="selectedTask" closable>
        <template #header>
          <div class="drawer-title">
            <span>{{ selectedTask.label }}</span>
            <small class="mono">{{ selectedTask.task_id }} · {{ selectedTask.state }}</small>
          </div>
        </template>
        <div class="task-detail-header">
          <div>
            <n-tag :type="stateType(selectedTask.state)">{{ selectedTask.state }}</n-tag>
            <n-tag>{{ selectedTask.mode }}</n-tag>
            <n-tag>{{ selectedTask.resource_class }}</n-tag>
          </div>
          <div>
            <n-button
              v-if="['queued', 'starting', 'running', 'cancelling'].includes(selectedTask.state)"
              type="error"
              ghost
              @click="cancel(selectedTask)"
            >取消</n-button>
            <n-button
              v-else-if="selectedTask.state !== 'succeeded'"
              type="primary"
              ghost
              @click="retry(selectedTask)"
            >按原 Spec 重试</n-button>
          </div>
        </div>
        <n-progress
          v-if="selectedTask.progress?.percent != null"
          type="line"
          :percentage="selectedTask.progress.percent"
          :status="selectedTask.state === 'failed' ? 'error' : 'default'"
        />
        <n-alert v-if="selectedTask.progress?.detail || selectedTask.queue_reason" :type="selectedTask.state === 'queued' ? 'warning' : 'info'">
          {{ selectedTask.progress?.detail ?? selectedTask.queue_reason }}
        </n-alert>

        <n-tabs v-model:value="detailTab" type="line" animated>
          <n-tab-pane name="overview" tab="结论">
            <div class="drawer-stack">
              <div class="task-headline">
                <span>{{ result?.semantics ?? selectedTask.kind }}</span>
                <strong>{{ result?.headline ?? '正在等待结构化结果…' }}</strong>
              </div>
              <div v-if="result && Object.keys(result.metrics).length" class="metric-kv-grid">
                <div v-for="(value, name) in result.metrics" :key="name"><span>{{ name }}</span><strong>{{ scalar(value) }}</strong></div>
              </div>
              <n-alert v-for="warning in result?.quality_warnings ?? []" :key="warning" type="warning">{{ warning }}</n-alert>
              <template v-if="runtimeLadder">
                <div class="selection-quality-strip">
                  <n-tag :type="runtimeLadder.quality.ready ? 'success' : 'warning'" size="small">
                    {{ runtimeLadder.quality.ready ? 'Round robin 完整' : '证据不完整' }}
                  </n-tag>
                  <span>{{ runtimeLadder.quality.observed_decks }} 卡组</span>
                  <span>{{ runtimeLadder.quality.observed_matchups }} matchups</span>
                  <span>Seat balanced: {{ runtimeLadder.quality.seat_balanced ? 'yes' : 'no' }}</span>
                </div>
                <deck-selection-matrix
                  :standings="runtimeLadder.standings"
                  :cells="runtimeLadder.cells"
                  :highlighted-deck-id="highlightedRuntimeDeck"
                  @select="highlightedRuntimeDeck = $event"
                />
              </template>
              <article v-if="report" class="markdown-report"><pre>{{ report.text }}</pre></article>
            </div>
          </n-tab-pane>
          <n-tab-pane name="tables" tab="数据表">
            <div class="drawer-stack">
              <n-select v-model:value="selectedTable" :options="(result?.tables ?? []).map((name) => ({ label: name, value: name }))" placeholder="选择结果表" />
              <div v-if="table" class="result-table-wrap">
                <table class="result-table">
                  <thead><tr><th v-for="column in table.columns" :key="column">{{ column }}</th></tr></thead>
                  <tbody>
                    <tr v-for="(row, index) in table.rows" :key="index">
                      <td v-for="column in table.columns" :key="column">{{ scalar(row[column]) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div v-if="table" class="table-pager">
                <n-button :disabled="table.offset === 0" @click="loadTable(selectedTask.task_id, table.table, Math.max(0, table.offset - table.limit))">上一页</n-button>
                <span>{{ table.offset + 1 }}–{{ table.offset + table.returned }}</span>
                <n-button :disabled="!table.has_more" @click="loadTable(selectedTask.task_id, table.table, table.offset + table.limit)">下一页</n-button>
              </div>
            </div>
          </n-tab-pane>
          <n-tab-pane name="artifacts" tab="产物">
            <div class="artifact-list">
              <div v-for="artifact in artifacts" :key="artifact.artifact_id" class="artifact-row">
                <div><strong>{{ artifact.label }}</strong><small class="mono">{{ bytes(artifact.size_bytes) }} · sha256 {{ compactHash(artifact.sha256) }}</small></div>
                <a :href="`/api/v2/tasks/${selectedTask.task_id}/artifacts/${artifact.artifact_id}/download`">下载</a>
              </div>
              <div v-if="!artifacts.length" class="empty-state"><strong>产物尚未发布</strong><span>只展示该 receipt 拥有的文件。</span></div>
            </div>
          </n-tab-pane>
          <n-tab-pane name="log" tab="日志">
            <pre class="job-log">{{ log?.text || '等待日志输出…' }}</pre>
            <small v-if="log?.truncated">仅显示日志尾部。</small>
          </n-tab-pane>
          <n-tab-pane name="receipt" tab="Receipt">
            <div class="drawer-stack">
              <div class="identity-block"><span>SPEC</span><code>{{ selectedTask.spec_fingerprint }}</code></div>
              <div class="identity-block"><span>ARGV</span><code>{{ selectedTask.argv.join(' ') }}</code></div>
              <div class="identity-block"><span>OUTPUT</span><code>{{ selectedTask.output_dir ?? 'none' }}</code></div>
            </div>
          </n-tab-pane>
        </n-tabs>
      </n-drawer-content>
    </n-drawer>
  </section>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import {
  NAlert,
  NButton,
  NCard,
  NFlex,
  NProgress,
  NSelect,
  NSpin,
  NTag,
} from 'naive-ui'
import { api } from '../api'
import type {
  EnvironmentCoverageState,
  EnvironmentMatchupCell,
  EnvironmentWindow,
  JobProgressPayload,
  JobReceipt,
  PublicEnvironmentPayload,
} from '../types'

const props = defineProps<{
  run: string
}>()

const windowDays = ref<EnvironmentWindow>(7)
const payload = ref<PublicEnvironmentPayload | null>(null)
const matrix = ref<EnvironmentMatchupCell[]>([])
const selectedDeckHash = ref<string | null>(null)
const loading = ref(false)
const refreshJob = ref<JobReceipt | null>(null)
const refreshProgress = ref<JobProgressPayload | null>(null)
const error = ref('')
let progressTimer: number | undefined
let disposed = false

const windowOptions: Array<{ label: string; value: EnvironmentWindow }> = [
  { label: '最近 1 个完整日', value: 1 },
  { label: '最近 2 个完整日', value: 2 },
  { label: '最近 7 个完整日', value: 7 },
  { label: '最近 14 个完整日', value: 14 },
]

const deckOptions = computed(() => (payload.value?.roster_standings ?? []).map(
  (row) => ({
    label: `${row.display_name ?? row.deck_label} · ${row.deck_hash}`,
    value: row.deck_hash,
  }),
))
const selectedMatchups = computed(() => matrix.value
  .filter((cell) => cell.candidate_deck_hash === selectedDeckHash.value)
  .sort((left, right) => right.games - left.games))
const refreshing = computed(() => refreshJob.value != null && isActive(refreshJob.value))
const refreshDisabled = computed(() => refreshing.value)

onMounted(async () => {
  await Promise.all([load(), restoreRefreshJob()])
})
onBeforeUnmount(() => {
  disposed = true
  window.clearTimeout(progressTimer)
})
watch(() => props.run, async () => {
  window.clearTimeout(progressTimer)
  refreshJob.value = null
  refreshProgress.value = null
  await Promise.all([load(), restoreRefreshJob()])
})
watch(windowDays, load)

async function load() {
  if (!props.run) return
  loading.value = true
  error.value = ''
  try {
    const summary = await api.publicEnvironment(
      props.run,
      windowDays.value,
      null,
    )
    payload.value = summary
    if (!summary.available) {
      matrix.value = []
      return
    }
    const detail = await api.publicEnvironmentMatrix(
      props.run,
      windowDays.value,
      null,
    )
    matrix.value = detail.cells
    const available = new Set(summary.roster_standings.map((row) => row.deck_hash))
    if (selectedDeckHash.value == null || !available.has(selectedDeckHash.value)) {
      selectedDeckHash.value = summary.roster_standings[0]?.deck_hash ?? null
    }
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : String(reason)
  } finally {
    loading.value = false
  }
}

async function refresh() {
  if (refreshing.value) return
  error.value = ''
  try {
    const receipt = await api.refreshPublicEnvironment(
      props.run,
      null,
    )
    bindRefreshJob(receipt)
    await pollRefreshJob()
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : String(reason)
  }
}

async function restoreRefreshJob() {
  try {
    const jobs = await api.jobs()
    const active = jobs.find((job) => matchesRun(job) && isActive(job))
    let candidate = active ?? null
    if (candidate == null) {
      const stored = window.localStorage.getItem(refreshStorageKey())
      if (stored) {
        try {
          const receipt = await api.job(stored)
          if (matchesRun(receipt)) candidate = receipt
        } catch {
          window.localStorage.removeItem(refreshStorageKey())
        }
      }
    }
    if (candidate != null) {
      bindRefreshJob(candidate)
      await pollRefreshJob()
    }
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : String(reason)
  }
}

function bindRefreshJob(receipt: JobReceipt) {
  refreshJob.value = receipt
  window.localStorage.setItem(refreshStorageKey(), receipt.job_id)
}

async function pollRefreshJob() {
  const job = refreshJob.value
  if (job == null || disposed) return
  window.clearTimeout(progressTimer)
  try {
    const [receipt, progress] = await Promise.all([
      api.job(job.job_id),
      api.jobProgress(job.job_id),
    ])
    refreshJob.value = receipt
    refreshProgress.value = progress
    if (isActive(receipt)) {
      scheduleProgressPoll()
    } else if (receipt.state === 'succeeded') {
      await load()
    } else {
      error.value = receipt.detail ?? `刷新任务结束于 ${receipt.state}`
    }
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : String(reason)
    if (refreshJob.value && isActive(refreshJob.value)) scheduleProgressPoll()
  }
}

function scheduleProgressPoll() {
  if (disposed) return
  progressTimer = window.setTimeout(pollRefreshJob, 2_000)
}

function matchesRun(job: JobReceipt) {
  return job.template_id === 'public_environment_refresh'
    && job.parameters.run === props.run
    && job.parameters.checkpoint_version == null
}

function refreshStorageKey() {
  return `ptcg-public-environment-refresh:${props.run}`
}

function isActive(job: JobReceipt) {
  return ['starting', 'running', 'cancelling'].includes(job.state)
}

function progressStatus() {
  if (refreshJob.value?.state === 'succeeded') return 'success'
  if (['failed', 'cancelled', 'unknown'].includes(refreshJob.value?.state ?? '')) {
    return 'error'
  }
  return 'default'
}

function jobStateLabel() {
  const state = refreshJob.value?.state
  if (state === 'succeeded') return '已完成'
  if (state === 'failed') return '失败'
  if (state === 'cancelled') return '已取消'
  if (state === 'unknown') return '状态未知'
  if (state === 'cancelling') return '正在取消'
  if (state === 'starting') return '正在启动'
  return '运行中'
}

function coverageLabel(state: EnvironmentCoverageState) {
  if (state === 'verified_responder') return '已有验证应对者'
  if (state === 'evidence_blind_spot') return '证据盲区'
  return '尚未判定'
}

function coverageType(state: EnvironmentCoverageState) {
  if (state === 'verified_responder') return 'success'
  if (state === 'evidence_blind_spot') return 'warning'
  return 'info'
}

function percent(value: number | null | undefined, digits = 1) {
  return value == null ? '—' : `${(value * 100).toFixed(digits)}%`
}

</script>

<template>
  <section class="page-stack public-environment-panel">
    <div class="page-heading">
      <div>
        <span class="page-eyebrow">KAGGLE DAILY PUBLIC ENVIRONMENT</span>
        <h1>公开环境</h1>
        <p>只使用全量 Daily 环境回放；不读取 leaderboard、submission 或训练胜率。</p>
      </div>
      <n-flex align="center">
        <n-select
          v-model:value="windowDays"
          :options="windowOptions"
          class="environment-window-select"
        />
        <n-button
          type="primary"
          secondary
          :disabled="refreshDisabled"
          :loading="refreshing"
          @click="refresh"
        >
          {{ refreshing ? '后台刷新中' : '手动刷新' }}
        </n-button>
      </n-flex>
    </div>

    <n-alert v-if="error" type="error" closable @close="error = ''">
      {{ error }}
    </n-alert>
    <n-card
      v-if="refreshJob"
      class="work-card environment-progress-card"
      :bordered="false"
      title="公开环境刷新任务"
    >
      <template #header-extra>
        <n-tag :type="progressStatus()">{{ jobStateLabel() }}</n-tag>
      </template>
      <n-progress
        type="line"
        :percentage="refreshProgress?.percent ?? (refreshJob.state === 'succeeded' ? 100 : 0)"
        :status="progressStatus()"
        :processing="refreshing"
      />
      <div class="environment-progress-meta">
        <strong>{{ refreshProgress?.message ?? '等待后台进程发布进度' }}</strong>
        <span v-if="refreshProgress?.total">
          {{ refreshProgress.completed }}/{{ refreshProgress.total }}
        </span>
        <span v-if="refreshProgress?.current_date">当前日期 {{ refreshProgress.current_date }}</span>
        <span class="mono">Job {{ refreshJob.job_id }}</span>
      </div>
      <n-alert type="info">
        任务由独立后台进程持有。退出、刷新或切换页面不会中止任务；重新进入本页会从持久化
        receipt 自动恢复进度。
      </n-alert>
    </n-card>

    <n-spin :show="loading">
      <n-alert v-if="payload && !payload.available" type="warning">
        {{ payload.unavailable_reason }}
        <template v-if="payload.missing_dates.length">
          ；缺少 {{ payload.missing_dates.join('、') }}。
        </template>
      </n-alert>

      <template v-if="payload?.available">
        <div class="metric-ribbon environment-metrics">
          <div><span>截止日</span><strong>{{ payload.as_of_date }}</strong></div>
          <div><span>有效对局</span><strong>{{ payload.quality.valid_episodes.toLocaleString() }}</strong></div>
          <div><span>Exact decks</span><strong>{{ payload.overview.exact_decks }}</strong></div>
          <div><span>可排名 roster</span><strong>{{ payload.quality.eligible_roster_decks }}/{{ payload.quality.roster_decks }}</strong></div>
          <div><span>Unknown tail</span><strong>{{ percent(payload.quality.unknown_tail_mass) }}</strong></div>
          <div><span>Top 10 占比</span><strong>{{ percent(payload.overview.top_10_share) }}</strong></div>
          <div><span>Meta HHI</span><strong>{{ payload.overview.meta_hhi.toFixed(3) }}</strong></div>
          <div><span>源缺失 episode</span><strong>{{ payload.quality.source_missing_episodes.toLocaleString() }}</strong></div>
        </div>

        <n-alert type="info">
          主排名为先后手各 50% 的 posterior 5% LCB；低样本卡组保留展示但不排名。
          当前 meta 按每个 Daily side 等权，draw 计 0.5，非 DONE 对局不进入 W/D/L。
        </n-alert>

        <n-card class="work-card" :bordered="false" title="当前训练 Roster · 公开环境排名">
          <div class="environment-table-wrap">
            <table class="environment-table">
              <thead>
                <tr>
                  <th>排名</th><th>卡组</th><th>证据</th><th>先/后</th>
                  <th>LCB</th><th>后验均值</th><th>95% CI</th>
                  <th>实际席位分</th><th>Prior-only meta</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in payload.roster_standings" :key="row.deck_digest">
                  <td>{{ row.rank ?? '—' }}</td>
                  <td class="environment-deck-cell">
                    <strong>{{ row.display_name ?? row.deck_label }}</strong>
                    <small>{{ row.deck_hash }}</small>
                  </td>
                  <td>
                    <n-tag size="small" :type="row.evidence_status === 'eligible' ? 'success' : 'warning'">
                      {{ row.valid_games }} 局
                    </n-tag>
                  </td>
                  <td>{{ row.first_games }}/{{ row.second_games }}</td>
                  <td><strong>{{ percent(row.deploy_lcb) }}</strong></td>
                  <td>{{ percent(row.deploy_mean) }}</td>
                  <td>{{ percent(row.deploy_credible_low) }}–{{ percent(row.deploy_credible_high) }}</td>
                  <td>{{ percent(row.observed_score) }}</td>
                  <td>{{ percent(row.prior_only_meta_mass) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </n-card>

        <div class="environment-grid">
          <n-card class="work-card" :bordered="false" title="全局环境雷达">
            <div class="environment-table-wrap environment-table-wrap--short">
              <table class="environment-table">
                <thead><tr><th>Exact deck</th><th>占比</th><th>变化</th><th>Pilots</th></tr></thead>
                <tbody>
                  <tr v-for="row in payload.meta_decks" :key="row.deck_digest">
                    <td class="environment-deck-cell">
                      <strong>{{ row.display_name ?? row.deck_label ?? '未命名牌表' }}</strong>
                      <small>{{ row.deck_hash ?? 'deck_hash unavailable' }}</small>
                    </td>
                    <td>{{ percent(row.share) }}</td>
                    <td :class="row.share_delta > 0 ? 'positive' : row.share_delta < 0 ? 'negative' : ''">
                      {{ row.share_delta > 0 ? '+' : '' }}{{ percent(row.share_delta) }}
                    </td>
                    <td>{{ row.unique_pilots }} · eff {{ row.effective_pilots.toFixed(1) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </n-card>

          <n-card class="work-card" :bordered="false" title="环境覆盖诊断">
            <div class="environment-table-wrap environment-table-wrap--short">
              <table class="environment-table">
                <thead><tr><th>主要对手</th><th>Meta</th><th>状态</th><th>最佳 LCB</th></tr></thead>
                <tbody>
                  <tr v-for="row in payload.coverage" :key="row.opponent_deck_digest">
                    <td class="environment-deck-cell">
                      <strong>{{ row.opponent_deck_label ?? '未命名牌表' }}</strong>
                      <small>{{ row.opponent_deck_hash ?? 'deck_hash unavailable' }}</small>
                    </td>
                    <td>{{ percent(row.meta_share) }}</td>
                    <td><n-tag size="small" :type="coverageType(row.state)">{{ coverageLabel(row.state) }}</n-tag></td>
                    <td>{{ percent(row.best_lcb) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </n-card>
        </div>

        <n-card class="work-card" :bordered="false" title="卡组 Matchup 下钻">
          <n-select
            v-model:value="selectedDeckHash"
            :options="deckOptions"
            class="environment-deck-select"
            placeholder="选择当前 roster 卡组"
          />
          <div class="environment-table-wrap environment-table-wrap--short">
            <table class="environment-table">
              <thead><tr><th>对手</th><th>证据</th><th>先/后</th><th>均值</th><th>LCB</th><th>95% CI</th></tr></thead>
              <tbody>
                <tr v-for="cell in selectedMatchups" :key="cell.opponent_deck_digest">
                  <td class="environment-deck-cell">
                    <strong>{{ cell.opponent_deck_label ?? '未命名牌表' }}</strong>
                    <small>{{ cell.opponent_deck_hash ?? 'deck_hash unavailable' }}</small>
                  </td>
                  <td>
                    <n-tag size="small" :type="cell.evidence_eligible ? 'success' : 'warning'">
                      {{ cell.games }} 局
                    </n-tag>
                  </td>
                  <td>{{ cell.first_games }}/{{ cell.second_games }}</td>
                  <td>{{ percent(cell.posterior_mean) }}</td>
                  <td>{{ percent(cell.lcb) }}</td>
                  <td>{{ percent(cell.credible_low) }}–{{ percent(cell.credible_high) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </n-card>
      </template>
    </n-spin>
  </section>
</template>

<style scoped>
.environment-window-select { width: 180px; }
.environment-deck-select { width: min(420px, 100%); margin-bottom: 10px; }
.environment-progress-card { margin-bottom: 14px; }
.environment-progress-meta { display: flex; flex-wrap: wrap; gap: 8px 16px; margin: 10px 0; color: #8190a3; font-size: 9px; }
.environment-progress-meta strong { color: #dce7ef; }
.environment-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.environment-table-wrap { max-width: 100%; overflow: auto; }
.environment-table-wrap--short { max-height: 440px; }
.environment-table { width: 100%; min-width: 760px; border-collapse: collapse; font-size: 9px; font-variant-numeric: tabular-nums; }
.environment-table th { position: sticky; top: 0; z-index: 1; padding: 8px; color: #8190a3; text-align: left; background: #0e1520; }
.environment-table td { padding: 8px; color: #b1bfce; border-top: 1px solid rgba(148, 163, 184, .07); white-space: nowrap; }
.environment-table td strong { color: #dce7ef; }
.environment-deck-cell strong, .environment-deck-cell small { display: block; }
.environment-deck-cell small { margin-top: 2px; color: #63d6b8; font-size: 7px; }
.positive { color: #63e6be !important; }
.negative { color: #ff8787 !important; }
@media (max-width: 1000px) { .environment-grid { grid-template-columns: 1fr; } }
</style>

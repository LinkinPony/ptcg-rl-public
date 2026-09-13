<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { NButton, NCard, NInput, NTabPane, NTabs } from 'naive-ui'
import { api } from '../api'
import LeagueStandingsTable from '../components/LeagueStandingsTable.vue'
import type { LeagueMatchup, LeagueStanding, LeagueSummary, LeagueWorker } from '../types'

interface WorkerGroup {
  key: string
  gpuIndex: number | null
  hostname: string
  slots: number
  active: number
  games: number
  gamesPerHour: number
  errors: number
  utilization: number | null
  freeGiB: number | null
  sourceCommit: string
  lastSeenAt: string
}

const summary = ref<LeagueSummary | null>(null)
const checkpoints = ref<LeagueStanding[]>([])
const decks = ref<LeagueStanding[]>([])
const bundles = ref<LeagueStanding[]>([])
const candidates = ref<LeagueStanding[]>([])
const workers = ref<LeagueWorker[]>([])
const matchups = ref<LeagueMatchup[]>([])
const loading = ref(false)
const error = ref('')
const notice = ref('')
const refreshedAt = ref<Date | null>(null)
const checkpointPath = ref('')
const checkpointLabel = ref('')
const deckPath = ref('')
const deckLabel = ref('')
const releasePath = ref('')
const releaseAlias = ref('')
const challengeSideA = ref('')
const challengeSideB = ref('')
let timer: number | undefined

const queueDepth = computed(() => (summary.value?.queued ?? 0) + (summary.value?.leased ?? 0))
const rankedDecks = computed(() => decks.value.filter((row) => row.rank_eligible).length)
const decidedRate = computed(() => {
  const completed = summary.value?.completed ?? 0
  return completed ? ((summary.value?.rating_events ?? 0) / completed) * 100 : 0
})
const bundleLabels = computed(() => new Map(bundles.value.map((row) => [row.identity, row.label])))
const workerGroups = computed<WorkerGroup[]>(() => {
  const groups = new Map<string, WorkerGroup>()
  for (const worker of workers.value) {
    const gpuValue = worker.resources.gpu_index
    const gpuIndex = typeof gpuValue === 'number' ? gpuValue : null
    const key = `${worker.hostname}:gpu-${gpuIndex ?? 'cpu'}`
    const utilizationValue = worker.resources.gpu_utilization_percent
    const memoryValue = worker.resources.gpu_memory_free_bytes
    const group = groups.get(key) ?? {
      key,
      gpuIndex,
      hostname: worker.hostname,
      slots: 0,
      active: 0,
      games: 0,
      gamesPerHour: 0,
      errors: 0,
      utilization: null,
      freeGiB: null,
      sourceCommit: worker.source_commit,
      lastSeenAt: worker.last_seen_at,
    }
    group.slots += 1
    group.active += worker.current_match_id ? 1 : 0
    group.games += worker.games_completed
    group.gamesPerHour += worker.games_per_hour
    group.errors += worker.errors
    if (typeof utilizationValue === 'number') {
      group.utilization = Math.max(group.utilization ?? 0, utilizationValue)
    }
    if (typeof memoryValue === 'number') {
      const freeGiB = memoryValue / 1024 ** 3
      group.freeGiB = Math.min(group.freeGiB ?? freeGiB, freeGiB)
    }
    if (worker.last_seen_at > group.lastSeenAt) group.lastSeenAt = worker.last_seen_at
    groups.set(key, group)
  }
  return [...groups.values()].sort((a, b) => a.key.localeCompare(b.key))
})

onMounted(async () => {
  await refresh()
  timer = window.setInterval(refresh, 10_000)
})
onBeforeUnmount(() => window.clearInterval(timer))

async function refresh() {
  if (loading.value || document.hidden) return
  loading.value = true
  try {
    ;[
      summary.value,
      checkpoints.value,
      decks.value,
      bundles.value,
      candidates.value,
      workers.value,
      matchups.value,
    ] = await Promise.all([
      api.leagueSummary(),
      api.leagueCheckpoints(),
      api.leagueDecks(),
      api.leagueBundles(),
      api.leagueCandidates(),
      api.leagueWorkers(),
      api.leagueMatchups(),
    ])
    refreshedAt.value = new Date()
    error.value = ''
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : String(reason)
  } finally {
    loading.value = false
  }
}

function matchupLabel(bundleId: string): string {
  return bundleLabels.value.get(bundleId) ?? bundleId.slice(0, 20)
}

async function addCheckpoint() {
  await runAction(() => api.addLeagueCheckpoint(checkpointPath.value, checkpointLabel.value || null))
}
async function addDeck() {
  await runAction(() => api.addLeagueDeck(deckPath.value, deckLabel.value || null))
}
async function addRelease() {
  await runAction(() => api.addLeagueRelease(releasePath.value, releaseAlias.value))
}
async function setControllerActive(row: LeagueStanding) {
  await runAction(() => api.setLeagueControllerActive(
    row.identity,
    !row.active,
    row.active ? 'disabled from continuous league UI' : 'restored from continuous league UI',
  ))
}
async function forceChallenge() {
  await runAction(() => api.forceLeagueChallenge(challengeSideA.value, challengeSideB.value))
}
async function runAction(action: () => Promise<unknown>) {
  try {
    await action()
    notice.value = 'coordinator 已接收并验证操作。'
    error.value = ''
    await refresh()
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : String(reason)
  }
}
</script>

<template>
  <section class="page-stack league-page">
    <div class="league-hero">
      <div>
        <span class="page-eyebrow">PERSISTENT COMPONENT TRUESKILL</span>
        <h1>持续联赛</h1>
        <p>Checkpoint 权重与精确牌组独立更新；Bundle 分数由两个后验实时合成。</p>
      </div>
      <div class="league-refresh">
        <small v-if="refreshedAt">更新于 {{ refreshedAt.toLocaleTimeString() }}</small>
        <n-button :loading="loading" @click="refresh">刷新</n-button>
      </div>
    </div>

    <div v-if="error" class="league-message is-error">{{ error }}</div>
    <div v-else-if="summary && !summary.available" class="league-message is-warning">
      coordinator 数据库尚未创建；页面会继续自动重试。
    </div>
    <div v-if="notice" class="league-message">{{ notice }}</div>

    <div class="league-signals">
      <article><span>已决事件</span><strong>{{ summary?.rating_events ?? 0 }}</strong><small>{{ decidedRate.toFixed(1) }}% 可计分完成局</small></article>
      <article><span>未决</span><strong>{{ summary?.unresolved ?? 0 }}</strong><small>基础设施 / 双方错误 / 超步数</small></article>
      <article><span>队列</span><strong>{{ queueDepth }}</strong><small>{{ summary?.leased ?? 0 }} leased · {{ summary?.queued ?? 0 }} waiting</small></article>
      <article><span>活跃榜</span><strong>{{ summary?.bundles ?? 0 }}</strong><small>{{ summary?.incumbents ?? 0 }} checkpoint · {{ rankedDecks }} decks</small></article>
      <article><span>候选</span><strong>{{ summary?.candidates ?? 0 }}</strong><small>32 / 256 局门禁</small></article>
      <article><span>Worker</span><strong>{{ summary?.workers ?? 0 }}</strong><small>{{ workerGroups.length }} 个资源池</small></article>
    </div>

    <n-card class="work-card league-standings" :bordered="false">
      <n-tabs type="line" animated>
        <n-tab-pane name="bundles" tab="Bundle 总榜">
          <league-standings-table :rows="bundles" view="bundle" />
        </n-tab-pane>
        <n-tab-pane name="checkpoints" tab="Checkpoint">
          <league-standings-table :rows="checkpoints" view="checkpoint" />
        </n-tab-pane>
        <n-tab-pane name="decks" tab="精确牌组">
          <league-standings-table :rows="decks" view="deck" />
        </n-tab-pane>
        <n-tab-pane name="candidates" tab="候选门禁">
          <league-standings-table :rows="candidates" view="candidate" />
          <div v-if="!candidates.length" class="league-empty">目前没有自动候选；人工 checkpoint 为永久 incumbent。</div>
          <div v-for="row in candidates" :key="`gate-${row.identity}`" class="league-gate">
            <div>
              <strong>{{ row.label }}</strong>
              <small>{{ Math.min(row.decided_games ?? row.games, 256) }}/256 · P(top20) {{ row.p_top20 == null ? '—' : `${(row.p_top20 * 100).toFixed(1)}%` }} · {{ row.decision_reason ?? '评测中' }}</small>
            </div>
            <n-button size="small" @click="setControllerActive(row)">{{ row.active ? '停用' : '恢复' }}</n-button>
          </div>
        </n-tab-pane>
      </n-tabs>
    </n-card>

    <div class="league-lower-grid">
      <n-card class="work-card" :bordered="false" title="Worker pools">
        <div v-if="!workerGroups.length" class="league-empty">当前没有 worker heartbeat。</div>
        <article v-for="group in workerGroups" :key="group.key" class="worker-pool">
          <div class="worker-pool__head">
            <div><span class="gpu-mark">GPU {{ group.gpuIndex ?? 'CPU' }}</span><strong>{{ group.hostname }}</strong></div>
            <span>{{ group.active }}/{{ group.slots }} active</span>
          </div>
          <div class="worker-pool__metrics">
            <div><span>吞吐</span><strong>{{ group.gamesPerHour.toFixed(0) }}</strong><small>games/hour</small></div>
            <div><span>完成</span><strong>{{ group.games }}</strong><small>worker reports</small></div>
            <div><span>GPU</span><strong>{{ group.utilization == null ? '—' : `${group.utilization}%` }}</strong><small>{{ group.freeGiB == null ? '—' : `${group.freeGiB.toFixed(1)} GiB free` }}</small></div>
            <div><span>错误</span><strong :class="{ danger: group.errors > 0 }">{{ group.errors }}</strong><small>{{ group.sourceCommit.slice(0, 10) }}</small></div>
          </div>
        </article>
      </n-card>

      <n-card class="work-card" :bordered="false" title="近期 Matchup">
        <div class="matchup-list">
          <div v-for="row in matchups.slice(0, 16)" :key="`${row.side_a_bundle_id}:${row.side_b_bundle_id}`" class="matchup-row">
            <div><strong>{{ matchupLabel(row.side_a_bundle_id) }}</strong><span>vs</span><strong>{{ matchupLabel(row.side_b_bundle_id) }}</strong></div>
            <div><b>{{ row.wins }}</b><span>/ {{ row.draws }} /</span><b>{{ row.losses }}</b><em v-if="row.unresolved">+{{ row.unresolved }} unresolved</em></div>
          </div>
          <div v-if="!matchups.length" class="league-empty">暂无对局结果。</div>
        </div>
      </n-card>
    </div>

    <details class="league-operations">
      <summary>受信网络操作面板</summary>
      <div class="league-form">
        <n-input v-model:value="checkpointPath" placeholder="checkpoint_pair_v*.json" />
        <n-input v-model:value="checkpointLabel" placeholder="Checkpoint label（可选）" />
        <n-button @click="addCheckpoint">加入 checkpoint</n-button>
        <n-input v-model:value="deckPath" placeholder="60 卡 CSV" />
        <n-input v-model:value="deckLabel" placeholder="牌组 label（可选）" />
        <n-button @click="addDeck">加入牌组</n-button>
        <n-input v-model:value="releasePath" placeholder="bundle_manifest.json" />
        <n-input v-model:value="releaseAlias" placeholder="submission alias" />
        <n-button @click="addRelease">加入 release</n-button>
        <n-input v-model:value="challengeSideA" placeholder="Side A bundle ID" />
        <n-input v-model:value="challengeSideB" placeholder="Side B bundle ID" />
        <n-button @click="forceChallenge">强制挑战</n-button>
      </div>
    </details>
  </section>
</template>

<style scoped>
.league-page { --league-border: rgba(148, 163, 184, .13); }
.league-hero { display: flex; align-items: flex-end; justify-content: space-between; gap: 18px; }
.league-hero h1 { margin: 4px 0 6px; }
.league-hero p { margin: 0; color: #94a3b8; }
.league-refresh { display: flex; align-items: center; gap: 10px; color: #64748b; }
.league-message { padding: 10px 14px; border: 1px solid rgba(52, 211, 153, .18); border-radius: 9px; background: rgba(52, 211, 153, .08); color: #a7f3d0; }
.league-message.is-error { border-color: rgba(248, 113, 113, .2); background: rgba(248, 113, 113, .1); color: #fecaca; }
.league-message.is-warning { border-color: rgba(245, 158, 11, .2); background: rgba(245, 158, 11, .1); color: #fde68a; }
.league-signals { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; }
.league-signals article { min-height: 92px; padding: 14px; border: 1px solid var(--league-border); border-radius: 12px; background: linear-gradient(145deg, rgba(30, 41, 59, .76), rgba(15, 23, 42, .9)); }
.league-signals span, .league-signals small { display: block; color: #94a3b8; }
.league-signals strong { display: block; margin: 5px 0 2px; color: #f8fafc; font-size: 25px; font-variant-numeric: tabular-nums; }
.league-standings { min-height: 420px; }
.league-lower-grid { display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(0, .95fr); gap: 14px; }
.worker-pool { padding: 13px 0; border-bottom: 1px solid var(--league-border); }
.worker-pool:last-child { border-bottom: 0; }
.worker-pool__head { display: flex; align-items: center; justify-content: space-between; color: #94a3b8; }
.worker-pool__head > div { display: flex; align-items: center; gap: 9px; }
.gpu-mark { padding: 3px 7px; border-radius: 6px; background: rgba(99, 230, 190, .12); color: #6ee7b7; font: 700 11px ui-monospace, monospace; }
.worker-pool__metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }
.worker-pool__metrics div { padding: 9px; border-radius: 8px; background: rgba(15, 23, 42, .56); }
.worker-pool__metrics span, .worker-pool__metrics small { display: block; color: #64748b; }
.worker-pool__metrics strong { display: block; margin: 2px 0; color: #e2e8f0; font-size: 17px; }
.worker-pool__metrics strong.danger { color: #fca5a5; }
.matchup-list { max-height: 390px; overflow: auto; }
.matchup-row { padding: 10px 0; border-bottom: 1px solid var(--league-border); }
.matchup-row > div { display: flex; align-items: center; gap: 7px; min-width: 0; }
.matchup-row strong { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.matchup-row > div:last-child { margin-top: 5px; color: #94a3b8; font-variant-numeric: tabular-nums; }
.matchup-row b { color: #e2e8f0; }
.matchup-row em { margin-left: auto; color: #fbbf24; font-size: 11px; font-style: normal; }
.league-gate { display: flex; justify-content: space-between; gap: 12px; padding: 10px 0; border-bottom: 1px solid var(--league-border); }
.league-gate small { display: block; margin-top: 3px; color: #94a3b8; }
.league-empty { padding: 28px 12px; color: #64748b; text-align: center; }
.league-operations { padding: 14px; border: 1px solid var(--league-border); border-radius: 12px; background: rgba(15, 23, 42, .6); }
.league-operations summary { color: #94a3b8; cursor: pointer; }
.league-form { display: grid; grid-template-columns: 1fr 1fr auto; gap: 8px; margin-top: 14px; }
@media (max-width: 1180px) { .league-signals { grid-template-columns: repeat(3, 1fr); } .league-lower-grid { grid-template-columns: 1fr; } }
@media (max-width: 720px) { .league-hero { align-items: flex-start; flex-direction: column; } .league-signals { grid-template-columns: repeat(2, 1fr); } .worker-pool__metrics { grid-template-columns: repeat(2, 1fr); } .league-form { grid-template-columns: 1fr; } }
</style>

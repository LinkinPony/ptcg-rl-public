<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { NAlert, NButton, NCard, NSelect, NTag } from 'naive-ui'
import { api } from '../api'
import { decimal, percent } from '../format'
import type {
  AllocationPortfolio,
  AllocationRole,
  AllocationSort,
  OpponentAllocationMatchupPage,
  OpponentAllocationSummary,
} from '../types'

const props = defineProps<{
  run: string
  summary: OpponentAllocationSummary | null
}>()

const pageSize = 100
const offset = ref(0)
const candidate = ref('')
const artifact = ref('')
const objective = ref<AllocationPortfolio | AllocationRole | ''>('')
const seat = ref<-1 | 0 | 1>(-1)
const sort = ref<AllocationSort>(
  props.summary?.allocation_mode === 'adaptive' ? 'debt' : 'planned_games',
)
const page = ref<OpponentAllocationMatchupPage | null>(null)
const loading = ref(false)
const error = ref('')
let generation = 0
let observedMode = props.summary?.allocation_mode ?? null

const portfolioLabels: Record<AllocationPortfolio, string> = {
  counter: '补强弱点',
  frontier: '能力前沿',
  probe: '信息探测',
  rehearsal: '防遗忘复习',
  staleness: '陈旧覆盖',
}
const roleLabels: Record<AllocationRole, string> = {
  protected: '保护锚点',
  recent: '近期策略',
  counter: '弱点对抗',
  frontier: '能力前沿',
  recovery: '恢复覆盖',
}
const isRoleBudget = computed(() => props.summary?.allocation_mode === 'role_budget')

const candidateOptions = computed(() => [
  { label: '全部候选牌组', value: '' },
  ...(props.summary?.candidates ?? []).map((item) => ({
    label: `${item.candidate_display_name} · ${item.candidate_deck_hash ?? '编号不可用'}`,
    value: item.candidate_deck_digest,
  })),
])
const artifactOptions = computed(() => [
  { label: '全部 active artifacts', value: '' },
  ...(props.summary?.artifacts ?? []).map((item) => ({
    label: `v${item.source_policy_version} · ${item.stratum} · ${short(item.source_fingerprint)}`,
    value: item.artifact_id,
  })),
])
const objectiveOptions = computed(() => [
  { label: isRoleBudget.value ? '全部预算角色' : '全部目标组合', value: '' },
  ...Object.entries(isRoleBudget.value ? roleLabels : portfolioLabels).map(
    ([value, label]) => ({ value, label }),
  ),
])
const seatOptions = [
  { label: '全部先后手', value: -1 },
  { label: '候选先手', value: 0 },
  { label: '候选后手', value: 1 },
]
const commonSortOptions: { label: string; value: AllocationSort }[] = [
  { label: '对局得分后验最低', value: 'weakness' },
  { label: '不确定性最高', value: 'uncertainty' },
  { label: '目标份额最高', value: 'target' },
  { label: '本窗计划局数最多', value: 'planned_games' },
]
const sortOptions = computed(() =>
  isRoleBudget.value
    ? commonSortOptions
    : [
        { label: '决策欠账最高', value: 'debt' as const },
        ...commonSortOptions,
        { label: '目标变化最大', value: 'change' as const },
      ],
)
const pageEnd = computed(() => Math.min(offset.value + pageSize, page.value?.total ?? 0))

watch(
  [
    () => props.run,
    () => props.summary?.target_fingerprint,
    () => props.summary?.allocation_mode,
    candidate,
    artifact,
    objective,
    seat,
    sort,
  ],
  () => {
    const nextMode = props.summary?.allocation_mode ?? null
    if (nextMode !== observedMode) {
      observedMode = nextMode
      objective.value = ''
      sort.value = nextMode === 'adaptive' ? 'debt' : 'planned_games'
    }
    if (isRoleBudget.value && (sort.value === 'debt' || sort.value === 'change')) {
      sort.value = 'planned_games'
    }
    offset.value = 0
    void loadPage()
  },
  { immediate: true },
)

async function loadPage() {
  if (!props.run || !props.summary?.available) {
    page.value = null
    return
  }
  const requestGeneration = ++generation
  loading.value = true
  error.value = ''
  try {
    const next = await api.opponentAllocationMatchups(props.run, {
      offset: offset.value,
      limit: pageSize,
      sort: sort.value,
      candidateDeckDigest: candidate.value || null,
      artifactId: artifact.value || null,
      portfolio: isRoleBudget.value
        ? null
        : (objective.value as AllocationPortfolio) || null,
      role: isRoleBudget.value
        ? (objective.value as AllocationRole) || null
        : null,
      candidateSeat: seat.value === 0 || seat.value === 1 ? seat.value : null,
    })
    if (requestGeneration === generation) page.value = next
  } catch (reason) {
    if (requestGeneration === generation) {
      error.value = reason instanceof Error ? reason.message : String(reason)
    }
  } finally {
    if (requestGeneration === generation) loading.value = false
  }
}

function movePage(nextOffset: number) {
  offset.value = Math.max(0, nextOffset)
  void loadPage()
}

function short(value: string | null | undefined, width = 10) {
  return value ? value.slice(0, width) : '—'
}

function signedPercent(value: number | null) {
  if (value == null) return '首次进入'
  const sign = value > 0 ? '+' : ''
  return `${sign}${(value * 100).toFixed(2)}pp`
}
</script>

<template>
  <section class="page-stack allocation-page">
    <div class="page-heading">
      <div>
        <span class="page-eyebrow">{{ isRoleBudget ? 'FIXED ROLE-BUDGET OPPONENT ALLOCATION' : 'JOINT ADAPTIVE OPPONENT ALLOCATION' }}</span>
        <h1>对手分配</h1>
        <p>{{ isRoleBudget ? '观察 collect 窗口如何把固定角色预算落实到候选、历史 artifact 与精确 route。' : '直接观察 collect 窗口为何选择某个候选、历史 artifact 与精确 route，而不把原始胜率当作唯一信号。' }}</p>
      </div>
      <n-tag :type="summary?.available ? 'success' : 'warning'" round>
        {{ summary?.available ? `SETTLED WINDOW ${summary.window_sequence}` : 'WAITING FOR ALLOCATION' }}
      </n-tag>
    </div>

    <n-alert v-if="!summary?.available" type="info" title="尚无对手分配窗口快照">
      {{ summary?.detail ?? '结算首个 collect 窗口后，这里会自动出现。' }}
    </n-alert>
    <n-card v-if="summary?.execution" class="work-card" :bordered="false">
      <template #header>
        <div class="section-title">
          <span>当前 quota 执行</span>
          <small>{{ summary.execution.window_id }} · {{ summary.execution.window_state }}</small>
        </div>
      </template>
      <div class="allocation-kpis execution-kpis">
        <div><span>Exposure shard</span><strong>{{ summary.execution.exposure_cohort_games.toLocaleString() }} games</strong></div>
        <div><span>首波 workers</span><strong>{{ summary.execution.initial_wave_workers_issued }} / {{ summary.execution.initial_wave_workers_total }}</strong></div>
        <div><span>Accepted</span><strong>{{ summary.execution.accepted_decisions.toLocaleString() }}</strong></div>
        <div><span>Provisional</span><strong>{{ summary.execution.provisional_decisions.toLocaleString() }}</strong></div>
        <div><span>Inflight credit</span><strong>{{ summary.execution.inflight_decision_credit.toLocaleString() }}</strong></div>
        <div><span>Shards</span><strong>{{ summary.execution.shards_completed }} / {{ summary.execution.shards_issued }}</strong></div>
      </div>
    </n-card>
    <template v-if="summary?.available">
      <div class="allocation-kpis">
        <div><span>精确 matchup cells</span><strong>{{ summary.matchup_count.toLocaleString() }}</strong></div>
        <div><span>有证据 cells</span><strong>{{ summary.evidence_cells.toLocaleString() }}</strong></div>
        <div><span>低证据 cells</span><strong>{{ summary.low_evidence_cells.toLocaleString() }}</strong></div>
        <div><span>Active artifacts</span><strong>{{ summary.artifacts.length }}</strong></div>
        <div><span>目标指纹</span><strong class="mono">{{ short(summary.target_fingerprint, 12) }}</strong></div>
      </div>

      <n-card class="work-card" :bordered="false">
        <template #header>
          <div class="section-title">
            <span>{{ isRoleBudget ? '固定角色预算' : '学习价值组合' }}</span>
            <small>{{ isRoleBudget ? '固定目标、artifact 数量与本窗实际可训练决策' : '目标决策质量 vs 本窗实际可训练决策' }}</small>
          </div>
        </template>
        <div v-if="!isRoleBudget" class="portfolio-grid">
          <div v-for="item in summary.portfolios" :key="item.portfolio" class="portfolio-card">
            <span>{{ portfolioLabels[item.portfolio] }}</span>
            <strong>{{ percent(item.target_share) }}</strong>
            <div class="share-track"><i :style="{ width: `${item.target_share * 100}%` }" /></div>
            <small>实际 {{ percent(item.actual_decision_share) }} · {{ item.actual_decisions.toLocaleString() }} decisions</small>
            <small>{{ item.planned_games.toLocaleString() }} planned games</small>
          </div>
        </div>
        <div v-else class="portfolio-grid">
          <div v-for="item in summary.roles" :key="item.role" class="portfolio-card">
            <span>{{ roleLabels[item.role] }}</span>
            <strong>{{ percent(item.target_share) }}</strong>
            <div class="share-track"><i :style="{ width: `${item.target_share * 100}%` }" /></div>
            <small>实际 {{ percent(item.actual_decision_share) }} · {{ item.actual_decisions.toLocaleString() }} decisions</small>
            <small>{{ item.artifact_count }} artifacts · {{ item.planned_games.toLocaleString() }} planned games</small>
          </div>
        </div>
      </n-card>

      <n-card class="work-card" :bordered="false">
        <template #header>
          <div class="section-title"><span>候选牌组目标</span><small>base 约束、联合弱点需求与实际窗口质量</small></div>
        </template>
        <div class="allocation-table-wrap">
          <table class="allocation-table candidate-table">
            <thead><tr><th>候选</th><th>Base</th><th>本窗目标</th><th v-if="!isRoleBudget">较上窗</th><th>目标加权得分后验</th><th>最弱 cell</th><th>证据覆盖</th><th>实际决策</th></tr></thead>
            <tbody>
              <tr v-for="item in summary.candidates" :key="item.candidate_deck_digest">
                <td><strong>{{ item.candidate_display_name }}</strong><small>{{ item.candidate_deck_hash ?? '牌组编号不可用' }}</small></td>
                <td>{{ percent(item.base_share) }}</td>
                <td><b>{{ percent(item.target_share) }}</b></td>
                <td v-if="!isRoleBudget" :class="{ positive: (item.previous_target_share ?? item.target_share) < item.target_share }">
                  {{ signedPercent(item.previous_target_share == null ? null : item.target_share - item.previous_target_share) }}
                </td>
                <td>{{ percent(item.posterior_score) }}</td>
                <td>{{ percent(item.worst_posterior_score) }}</td>
                <td>{{ item.evidence_cells.toLocaleString() }} / {{ item.matchup_cells.toLocaleString() }}</td>
                <td>{{ item.actual_decisions.toLocaleString() }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </n-card>

      <n-card class="work-card" :bordered="false">
        <template #header>
          <div class="section-title"><span>Active 历史 artifacts</span><small>{{ isRoleBudget ? '固定角色、route 覆盖与执行结果' : '动态保留、重入与覆盖结果' }}</small></div>
        </template>
        <div class="allocation-table-wrap">
          <table class="allocation-table artifact-table">
            <thead><tr><th>冻结策略</th><th>角色</th><th>Routes</th><th>全局目标</th><th v-if="!isRoleBudget">后验</th><th v-if="!isRoleBudget">最弱 cell</th><th>计划 / 实际局</th><th>实际决策</th></tr></thead>
            <tbody>
              <tr v-for="item in summary.artifacts" :key="item.artifact_id">
                <td><strong>policy v{{ item.source_policy_version }}</strong><small>{{ short(item.source_fingerprint, 14) }}</small></td>
                <td><n-tag size="small" :bordered="false">{{ item.role ? roleLabels[item.role] : item.stratum }}</n-tag></td>
                <td>{{ item.route_count }}</td>
                <td>{{ percent(item.target_share) }}</td>
                <td v-if="!isRoleBudget">{{ percent(item.posterior_score) }}</td>
                <td v-if="!isRoleBudget">{{ percent(item.worst_posterior_score) }}</td>
                <td>{{ item.planned_games.toLocaleString() }} / {{ item.actual_games.toLocaleString() }}</td>
                <td>{{ item.actual_decisions.toLocaleString() }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </n-card>

      <n-card class="work-card" :bordered="false">
        <template #header>
          <div class="section-title"><span>精确 matchup 诊断</span><small>{{ isRoleBudget ? '服务端筛选与分页；展示角色目标、后验证据与执行量' : '服务端筛选与分页；正 debt 表示历史决策量不足' }}</small></div>
        </template>
        <div class="allocation-filters">
          <n-select v-model:value="candidate" :options="candidateOptions" placeholder="候选牌组" />
          <n-select v-model:value="artifact" :options="artifactOptions" placeholder="历史 artifact" />
          <n-select v-model:value="objective" :options="objectiveOptions" :placeholder="isRoleBudget ? '预算角色' : '目标组合'" />
          <n-select v-model:value="seat" :options="seatOptions" placeholder="先后手" />
          <n-select v-model:value="sort" :options="sortOptions" placeholder="排序" />
        </div>
        <n-alert v-if="error" type="error" title="读取 matchup 明细失败">{{ error }}</n-alert>
        <div class="allocation-table-wrap matchup-wrap" :class="{ loading }">
          <table class="allocation-table matchup-table">
            <thead><tr><th>候选 → 对手</th><th>席位 / 目标</th><th>Artifact</th><th>快 / 慢后验</th><th>σ / 证据</th><th>{{ isRoleBudget ? '目标份额' : '目标 / Δ' }}</th><th>{{ isRoleBudget ? '历史决策量' : '决策 debt' }}</th><th>计划局 / 实际决策</th></tr></thead>
            <tbody>
              <tr v-for="row in page?.rows ?? []" :key="`${row.candidate_deck_digest}/${row.route_id}/${row.candidate_seat}`">
                <td>
                  <strong>{{ row.candidate_display_name }} <em>{{ row.candidate_deck_hash ?? '编号不可用' }}</em></strong>
                  <small>→ {{ row.opponent_display_name }} · {{ row.opponent_deck_hash ?? '编号不可用' }}</small>
                </td>
                <td><b>seat {{ row.candidate_seat }}</b><small>{{ row.role ? roleLabels[row.role] : row.portfolio ? portfolioLabels[row.portfolio] : '—' }}</small></td>
                <td><b>v{{ row.source_policy_version }} · {{ row.stratum }}</b><small>{{ short(row.source_fingerprint, 12) }}</small></td>
                <td>{{ percent(row.posterior_score) }} / {{ percent(row.slow_score) }}</td>
                <td>{{ decimal(row.posterior_stddev, 4) }} / {{ decimal(row.effective_evidence, 1) }}</td>
                <td>{{ percent(row.target_share) }}<small v-if="!isRoleBudget">{{ signedPercent(row.target_delta) }}</small></td>
                <td v-if="isRoleBudget">
                  {{ decimal(row.decision_mass, 1) }}
                  <small>expected/game {{ decimal(row.expected_decisions_per_game, 1) }}</small>
                </td>
                <td v-else :class="{ positive: (row.decision_debt_before ?? 0) > 0, negative: (row.decision_debt_before ?? 0) < 0 }">
                  {{ decimal(row.decision_debt_before, 1) }}
                  <small>planned → {{ decimal(row.projected_decision_debt, 1) }}</small>
                </td>
                <td>{{ row.planned_games.toLocaleString() }} / {{ row.actual_decisions.toLocaleString() }}</td>
              </tr>
            </tbody>
          </table>
          <div v-if="!loading && !(page?.rows.length)" class="empty-state"><span>当前过滤条件没有 matchup cell。</span></div>
        </div>
        <div class="allocation-pagination">
          <span>{{ page?.total ? `${offset + 1}–${pageEnd} / ${page.total.toLocaleString()}` : '0 rows' }}</span>
          <div>
            <n-button size="small" :disabled="offset === 0 || loading" @click="movePage(offset - pageSize)">上一页</n-button>
            <n-button size="small" :disabled="pageEnd >= (page?.total ?? 0) || loading" @click="movePage(offset + pageSize)">下一页</n-button>
          </div>
        </div>
      </n-card>

      <div class="allocation-provenance">
        <span>plan <b>{{ short(summary.plan_id, 16) }}</b></span>
        <span>revision <b>{{ short(summary.revision_fingerprint, 16) }}</b></span>
        <span>committed state <b>{{ short(summary.committed_state_fingerprint, 16) }}</b></span>
        <span>recorded <b>{{ summary.recorded_at_utc }}</b></span>
      </div>
    </template>
  </section>
</template>

<style scoped>
.allocation-kpis { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 9px; }
.execution-kpis { grid-template-columns: repeat(6, minmax(0, 1fr)); }
.allocation-kpis > div { padding: 13px; border: 1px solid rgba(148, 163, 184, .1); border-radius: 10px; background: rgba(10, 18, 29, .75); }
.allocation-kpis span, .allocation-kpis strong { display: block; }
.allocation-kpis span { color: #738399; font-size: 8px; letter-spacing: .05em; text-transform: uppercase; }
.allocation-kpis strong { margin-top: 6px; color: #e5eef6; font-size: 18px; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px !important; }
.portfolio-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 9px; }
.portfolio-card { padding: 12px; border-radius: 9px; background: rgba(5, 12, 20, .45); border: 1px solid rgba(99, 230, 190, .1); }
.portfolio-card > span, .portfolio-card > strong, .portfolio-card > small { display: block; }
.portfolio-card > span { color: #88a0b5; font-size: 9px; }
.portfolio-card > strong { margin: 5px 0; color: #74e3c2; font-size: 18px; }
.portfolio-card > small { margin-top: 4px; color: #66798c; font-size: 8px; }
.share-track { height: 4px; overflow: hidden; border-radius: 999px; background: rgba(148, 163, 184, .1); }
.share-track i { display: block; height: 100%; background: linear-gradient(90deg, #2dd4bf, #74c0fc); }
.allocation-table-wrap { width: 100%; overflow: auto; transition: opacity .15s ease; }
.allocation-table-wrap.loading { opacity: .5; }
.allocation-table { width: 100%; border-collapse: collapse; font-size: 9px; font-variant-numeric: tabular-nums; }
.candidate-table { min-width: 1050px; }
.artifact-table { min-width: 980px; }
.matchup-table { min-width: 1480px; }
.allocation-table th { position: sticky; top: 0; z-index: 1; padding: 9px; color: #7d8da1; text-align: left; background: #0e1622; white-space: nowrap; }
.allocation-table td { padding: 9px; color: #b4c1ce; border-top: 1px solid rgba(148, 163, 184, .07); white-space: nowrap; }
.allocation-table td strong, .allocation-table td small { display: block; }
.allocation-table td strong { color: #dfeaf2; }
.allocation-table td small { margin-top: 3px; color: #697b8e; }
.allocation-table td b { color: #c4d3df; font-weight: 650; }
.allocation-table td em { margin-left: 5px; color: #62d6ba; font-size: 8px; font-style: normal; }
.allocation-table .positive { color: #63e6be; }
.allocation-table .negative { color: #ff9f9f; }
.allocation-filters { display: grid; grid-template-columns: 1.4fr 1.2fr 1fr .8fr 1.1fr; gap: 8px; margin-bottom: 12px; }
.matchup-wrap { min-height: 320px; }
.allocation-pagination { display: flex; align-items: center; justify-content: space-between; margin-top: 10px; color: #718298; font-size: 9px; }
.allocation-pagination > div { display: flex; gap: 7px; }
.allocation-provenance { display: flex; flex-wrap: wrap; gap: 8px 18px; color: #65778b; font-size: 8px; }
.allocation-provenance b { color: #91a3b5; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
@media (max-width: 1100px) {
  .allocation-kpis, .portfolio-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .allocation-filters { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 680px) {
  .allocation-kpis, .portfolio-grid, .allocation-filters { grid-template-columns: 1fr; }
}
</style>

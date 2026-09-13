<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import {
  NAlert,
  NCard,
  NFlex,
  NSelect,
  NSpin,
  NTag,
} from 'naive-ui'
import { api } from '../api'
import { decimal, percent } from '../format'
import type {
  CheckpointInfo,
  EnvironmentWindow,
  TrainingEvidenceRange,
  TwoDeckPair,
  TwoDeckRecommendationPayload,
} from '../types'

const props = defineProps<{
  run: string
  checkpoint: CheckpointInfo | null
  refreshVersion?: number
}>()

const trainingRange = ref<TrainingEvidenceRange>('checkpoint')
const windowDays = ref<EnvironmentWindow>(7)
const payload = ref<TwoDeckRecommendationPayload | null>(null)
const loading = ref(false)
const error = ref('')
let requestGeneration = 0

const rangeOptions: Array<{ label: string; value: TrainingEvidenceRange }> = [
  { label: '所选 Checkpoint', value: 'checkpoint' },
  { label: '近 15 分钟', value: 'recent_15m' },
  { label: '近 60 分钟', value: 'recent_60m' },
  { label: 'Run 累计', value: 'cumulative' },
]
const windowOptions: Array<{ label: string; value: EnvironmentWindow }> = [
  { label: '最近 1 个完整日', value: 1 },
  { label: '最近 2 个完整日', value: 2 },
  { label: '最近 7 个完整日', value: 7 },
  { label: '最近 14 个完整日', value: 14 },
]
const warnings = computed(() => [...new Set([
  ...(payload.value?.quality.warnings ?? []),
  ...(payload.value?.warnings ?? []),
])])
const topPairs = computed(() => payload.value?.pairs.slice(0, 10) ?? [])
const recommendedCandidates = computed(() => {
  const recommendation = payload.value?.recommendation
  if (!recommendation) return []
  const selected = new Set([
    recommendation.deck_a_digest,
    recommendation.deck_b_digest,
  ])
  return (payload.value?.candidates ?? []).filter(
    (candidate) => selected.has(candidate.deck_digest),
  )
})

onMounted(load)
watch(
  [
    () => props.run,
    () => props.checkpoint?.version,
    () => props.refreshVersion,
    trainingRange,
    windowDays,
  ],
  load,
)

async function load() {
  const generation = ++requestGeneration
  if (!props.run || !props.checkpoint) {
    payload.value = null
    loading.value = false
    error.value = ''
    return
  }
  loading.value = true
  error.value = ''
  payload.value = null
  try {
    const next = await api.twoDeckRecommendation(
      props.run,
      trainingRange.value,
      windowDays.value,
      props.checkpoint.version,
    )
    if (generation === requestGeneration) payload.value = next
  } catch (reason) {
    if (generation === requestGeneration) {
      error.value = reason instanceof Error ? reason.message : String(reason)
    }
  } finally {
    if (generation === requestGeneration) loading.value = false
  }
}

function pairLabel(pair: TwoDeckPair) {
  return `${pair.deck_a_display_name} + ${pair.deck_b_display_name}`
}

function correlation(value: number | null) {
  return value == null ? '—' : decimal(value, 2)
}
</script>

<template>
  <section class="page-stack two-deck-recommendation-panel">
    <div class="page-heading">
      <div>
        <span class="page-eyebrow">TWO-SUBMISSION PORTFOLIO PROXY</span>
        <h1>双席卡组推荐</h1>
        <p>融合训练 matchup 与 Kaggle Daily meta，寻找未来环境场景下最高一席更有潜力的组合。</p>
      </div>
      <n-flex align="center" wrap>
        <n-select
          v-model:value="trainingRange"
          :options="rangeOptions"
          class="portfolio-filter-select"
          aria-label="训练证据范围"
        />
        <n-select
          v-model:value="windowDays"
          :options="windowOptions"
          class="portfolio-filter-select"
          aria-label="Kaggle Daily 窗口"
        />
      </n-flex>
    </div>

    <n-alert
      class="portfolio-disclaimer"
      type="warning"
      title="代理指标，不是提交结论"
    >
      这是训练证据与 Kaggle Daily meta 的组合代理，不是 Kaggle μ/rank 预测，也不是提交授权。
      它不会创建 profile、上传 submission，或替代固定 deployment bundle 门禁。
    </n-alert>
    <n-alert v-if="error" type="error" closable @close="error = ''">
      {{ error }}
    </n-alert>

    <n-spin :show="loading">
      <template v-if="payload">
        <div class="portfolio-quality-strip">
          <n-tag :type="payload.quality.ready ? 'success' : 'warning'" round>
            {{ payload.quality.ready ? '证据可计算' : '证据未就绪' }}
          </n-tag>
          <span>候选 {{ payload.quality.eligible_candidates }}/{{ payload.quality.active_candidates }}</span>
          <span>组合 {{ payload.quality.candidate_pairs }}</span>
          <span>已展开 exact {{ percent(payload.quality.known_meta_mass) }}</span>
          <span>未展开 exact {{ percent(payload.quality.unexpanded_explicit_meta_mass) }}</span>
          <span>Rare / Unknown {{ percent(payload.quality.rare_unknown_meta_mass) }}</span>
          <span>{{ payload.public_as_of_date ?? '公开环境日期不可用' }}</span>
        </div>

        <n-alert v-for="warning in warnings" :key="warning" type="warning">
          {{ warning }}
        </n-alert>

        <n-card
          v-if="payload.available && payload.recommendation"
          class="work-card portfolio-recommendation-card"
          :bordered="false"
        >
          <template #header>
            <div class="section-title">
              <span>推荐双席</span>
              <small>目标：两个固定 submission 的最高 meta-weighted score proxy</small>
            </div>
          </template>
          <template #header-extra>
            <n-tag type="success" size="small">#{{ payload.recommendation.rank }}</n-tag>
          </template>

          <div class="portfolio-deck-pair">
            <div class="portfolio-deck-card">
              <span>SLOT 1</span>
              <strong>{{ payload.recommendation.deck_a_display_name }}</strong>
              <small class="mono">deck_hash · {{ payload.recommendation.deck_a_hash }}</small>
            </div>
            <div class="portfolio-pair-mark">+</div>
            <div class="portfolio-deck-card">
              <span>SLOT 2</span>
              <strong>{{ payload.recommendation.deck_b_display_name }}</strong>
              <small class="mono">deck_hash · {{ payload.recommendation.deck_b_hash }}</small>
            </div>
          </div>

          <div class="portfolio-metrics">
            <div>
              <span>Expected max proxy</span>
              <strong>{{ percent(payload.recommendation.expected_best_score) }}</strong>
              <small>两个固定席位中的较高场景分</small>
            </div>
            <div>
              <span>95% proxy CI</span>
              <strong>
                {{ percent(payload.recommendation.credible_low) }}–{{ percent(payload.recommendation.credible_high) }}
              </strong>
              <small>代理后验区间</small>
            </div>
            <div>
              <span>Best-score LCB</span>
              <strong>{{ percent(payload.recommendation.best_score_lcb) }}</strong>
              <small>组合保守下界</small>
            </div>
            <div>
              <span>多样化收益</span>
              <strong>{{ percent(payload.recommendation.diversification_gain) }}</strong>
              <small>相对较强单席的期望增益</small>
            </div>
            <div>
              <span>训练共同弱面</span>
              <strong>{{ percent(payload.recommendation.common_weak_meta_mass) }}</strong>
              <small>两席同时偏弱的已识别 meta 质量</small>
            </div>
            <div>
              <span>P(至少一席 &gt; 50%)</span>
              <strong>{{ percent(payload.recommendation.probability_at_least_one_above_even) }}</strong>
              <small>不是 Kaggle rating 达标概率</small>
            </div>
            <div>
              <span>共同下行概率</span>
              <strong>{{ percent(payload.recommendation.joint_downside_probability) }}</strong>
              <small>两个代理分同时不高于 50%</small>
            </div>
            <div>
              <span>场景相关性</span>
              <strong>{{ correlation(payload.recommendation.score_correlation) }}</strong>
              <small>缺失时不作推断</small>
            </div>
          </div>
        </n-card>

        <div v-else class="empty-state portfolio-empty">
          <strong>暂时无法生成双席推荐</strong>
          <span>需要至少两个 route-compatible 候选，并同时具备训练与完整 Kaggle Daily 环境证据。</span>
        </div>

        <n-card
          v-if="recommendedCandidates.length"
          class="work-card"
          :bordered="false"
        >
          <template #header>
            <div class="section-title">
              <span>两席证据拆解</span>
              <small>Daily 牌表后验仅作独立诊断，不参与 v1 排名；proxy 用双 seat 训练 matchup 重加权到近期 meta</small>
            </div>
          </template>
          <div class="portfolio-table-wrap">
            <table class="portfolio-table portfolio-evidence-table">
              <thead>
                <tr>
                  <th>卡组</th><th>训练表现</th><th>Kaggle Daily 牌表表现（诊断）</th>
                  <th>近期 meta proxy</th><th>已观测 / Prior-only meta</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="candidate in recommendedCandidates" :key="candidate.deck_digest">
                  <td class="portfolio-pair-cell">
                    <strong>{{ candidate.display_name }}</strong>
                    <small>deck_hash · {{ candidate.deck_hash }}</small>
                  </td>
                  <td>
                    {{ percent(candidate.training_mean) }}
                    <small>
                      {{ candidate.training_games }} 局 ·
                      {{ percent(candidate.training_credible_low) }}–{{ percent(candidate.training_credible_high) }}
                    </small>
                  </td>
                  <td>
                    {{ percent(candidate.public_deploy_mean) }}
                    <small>{{ candidate.public_games }} 局 · LCB {{ percent(candidate.public_deploy_lcb) }}</small>
                  </td>
                  <td>
                    <strong>#{{ candidate.proxy_rank }} · {{ percent(candidate.proxy_mean) }}</strong>
                    <small>LCB {{ percent(candidate.proxy_lcb) }}</small>
                  </td>
                  <td>
                    {{ percent(candidate.observed_meta_mass) }} /
                    {{ percent(candidate.prior_only_meta_mass) }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </n-card>

        <n-card
          v-if="topPairs.length"
          class="work-card"
          :bordered="false"
        >
          <template #header>
            <div class="section-title">
              <span>Top 组合</span>
              <small>组合排序而非单卡综合分前二；每一行均为两个固定 exact deck</small>
            </div>
          </template>
          <div class="portfolio-table-wrap">
            <table class="portfolio-table">
              <thead>
                <tr>
                  <th>排名</th><th>两席</th><th>Expected max</th><th>95% CI / LCB</th>
                  <th>多样化收益</th><th>P(至少一席 &gt;50%)</th><th>共同下行</th>
                  <th>共同弱面</th><th>相关性</th><th>Regret</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="pair in topPairs" :key="`${pair.deck_a_digest}:${pair.deck_b_digest}`">
                  <td><strong>#{{ pair.rank }}</strong></td>
                  <td class="portfolio-pair-cell" :title="pairLabel(pair)">
                    <strong>{{ pair.deck_a_display_name }}</strong>
                    <small>deck_hash · {{ pair.deck_a_hash }}</small>
                    <b>+</b>
                    <strong>{{ pair.deck_b_display_name }}</strong>
                    <small>deck_hash · {{ pair.deck_b_hash }}</small>
                  </td>
                  <td><strong>{{ percent(pair.expected_best_score) }}</strong></td>
                  <td>
                    {{ percent(pair.credible_low) }}–{{ percent(pair.credible_high) }}
                    <small>LCB {{ percent(pair.best_score_lcb) }}</small>
                  </td>
                  <td>{{ percent(pair.diversification_gain) }}</td>
                  <td>{{ percent(pair.probability_at_least_one_above_even) }}</td>
                  <td>{{ percent(pair.joint_downside_probability) }}</td>
                  <td>{{ percent(pair.common_weak_meta_mass) }}</td>
                  <td>{{ correlation(pair.score_correlation) }}</td>
                  <td>{{ percent(pair.expected_regret) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </n-card>

        <div class="portfolio-provenance">
          <span>Run <b>{{ payload.run_id }}</b></span>
          <span>Checkpoint <b>{{ payload.checkpoint_version == null ? '未绑定' : `v${payload.checkpoint_version}` }}</b></span>
          <span>训练范围 <b>{{ payload.training_range }}</b></span>
          <span>Daily <b>{{ payload.public_window_days }} 日</b></span>
          <span>Snapshot <b class="mono">{{ payload.public_snapshot_fingerprint ?? '不可用' }}</b></span>
        </div>
      </template>

      <div v-else class="empty-state portfolio-empty">
        <strong>{{ error ? '双席证据读取失败' : checkpoint ? '正在读取双席证据' : '请先选择 immutable checkpoint' }}</strong>
        <span>本面板只读取已有统计，不会启动评测或提交。</span>
      </div>
    </n-spin>
  </section>
</template>

<style scoped>
.portfolio-filter-select { width: 180px; }
.portfolio-disclaimer { border: 1px solid rgba(255, 193, 92, .28); }
.portfolio-quality-strip {
  display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; padding: 9px 11px;
  color: #7c8da2; border: 1px solid rgba(99, 230, 190, .11); border-radius: 8px;
  background: rgba(99, 230, 190, .035); font-size: 8px;
}
.portfolio-recommendation-card { border-color: rgba(99, 230, 190, .22) !important; }
.portfolio-deck-pair { display: grid; grid-template-columns: minmax(0, 1fr) 32px minmax(0, 1fr); align-items: stretch; gap: 10px; }
.portfolio-deck-card {
  min-width: 0; padding: 14px; border: 1px solid rgba(99, 230, 190, .13); border-radius: 9px;
  background: linear-gradient(145deg, rgba(17, 39, 37, .72), rgba(7, 14, 22, .58));
}
.portfolio-deck-card > span { color: #64d8ba; font-size: 7px; font-weight: 800; letter-spacing: .13em; }
.portfolio-deck-card > strong { display: block; margin: 7px 0 5px; color: #edf6f3; font-size: 16px; line-height: 1.35; }
.portfolio-deck-card > small { color: #63d6b8; font-size: 8px; overflow-wrap: anywhere; }
.portfolio-pair-mark { display: grid; place-items: center; color: #63e6be; font-size: 20px; }
.portfolio-metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }
.portfolio-metrics > div { min-width: 0; padding: 10px; border-radius: 8px; background: rgba(5, 10, 18, .35); border: 1px solid rgba(148, 163, 184, .07); }
.portfolio-metrics span, .portfolio-metrics small { display: block; }
.portfolio-metrics span { color: #6c7e93; font-size: 7px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
.portfolio-metrics strong { display: block; margin: 5px 0 3px; color: #dcebe6; font-size: 15px; font-variant-numeric: tabular-nums; }
.portfolio-metrics small { color: #63758a; font-size: 7px; line-height: 1.45; }
.portfolio-empty { min-height: 180px; }
.portfolio-table-wrap { max-width: 100%; overflow: auto; }
.portfolio-table { width: 100%; min-width: 1180px; border-collapse: collapse; font-size: 8px; font-variant-numeric: tabular-nums; }
.portfolio-evidence-table { min-width: 820px; }
.portfolio-table th { position: sticky; top: 0; z-index: 1; padding: 8px; color: #8190a3; text-align: left; background: #0e1520; }
.portfolio-table td { padding: 8px; color: #aebdca; border-top: 1px solid rgba(148, 163, 184, .07); white-space: nowrap; }
.portfolio-table td > strong { color: #dce7ef; }
.portfolio-table td > small { display: block; margin-top: 3px; color: #64768b; }
.portfolio-pair-cell { min-width: 270px; }
.portfolio-pair-cell strong, .portfolio-pair-cell small { display: inline; }
.portfolio-pair-cell small { margin-left: 5px; color: #63d6b8; }
.portfolio-pair-cell b { margin: 0 7px; color: #64768b; }
.portfolio-provenance { display: flex; flex-wrap: wrap; gap: 7px 18px; color: #627388; font-size: 7px; }
.portfolio-provenance b { color: #91a0b1; font-weight: 550; overflow-wrap: anywhere; }
@media (max-width: 900px) {
  .portfolio-deck-pair { grid-template-columns: 1fr; }
  .portfolio-pair-mark { min-height: 20px; }
  .portfolio-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 560px) {
  .portfolio-filter-select { width: 100%; }
  .portfolio-metrics { grid-template-columns: 1fr; }
}
</style>

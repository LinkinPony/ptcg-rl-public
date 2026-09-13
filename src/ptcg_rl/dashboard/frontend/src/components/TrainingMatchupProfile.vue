<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { NSelect } from 'naive-ui'
import { percent } from '../format'
import {
  intervalPosition,
  matchupConclusion,
  sortedMatchups,
  stratumLabel,
} from '../trainingMatchupProfile'
import type {
  MatchupConclusion,
  MatchupSort,
} from '../trainingMatchupProfile'
import type {
  TrainingDeckMatchupsPayload,
  TrainingOpponentMatchupSummary,
  TrainingPosterior,
} from '../types'
import TrainingMatchupBreakdown from './TrainingMatchupBreakdown.vue'

const props = defineProps<{
  detail: TrainingDeckMatchupsPayload | null
  loading: boolean
}>()

type ConclusionFilter = MatchupConclusion | 'all'

const activeConclusion = ref<ConclusionFilter>('disadvantage')
const sortMode = ref<MatchupSort>('weakest')
const expandedOpponent = ref<string | null>(null)

const conclusionLabels: Record<MatchupConclusion, string> = {
  disadvantage: '明确劣势',
  uncertain: '尚不确定',
  advantage: '明确优势',
}
const sortOptions: Array<{ label: string; value: MatchupSort }> = [
  { label: '最弱优先', value: 'weakest' },
  { label: '最强优先', value: 'strongest' },
  { label: '局数最多', value: 'games' },
  { label: '最不确定', value: 'uncertainty' },
]

const conclusionCounts = computed(() => {
  const counts: Record<MatchupConclusion, number> = {
    disadvantage: 0,
    uncertain: 0,
    advantage: 0,
  }
  for (const row of props.detail?.opponents ?? []) {
    counts[matchupConclusion(row.posterior)] += 1
  }
  return counts
})

const visibleOpponents = computed(() => sortedMatchups(
  (props.detail?.opponents ?? []).filter((row) => (
    activeConclusion.value === 'all'
    || matchupConclusion(row.posterior) === activeConclusion.value
  )),
  sortMode.value,
))

const strongest = computed(() => sortedMatchups(
  (props.detail?.opponents ?? []).filter(
    (row) => matchupConclusion(row.posterior) === 'advantage',
  ),
  'strongest',
)[0] ?? null)

const weakest = computed(() => sortedMatchups(
  (props.detail?.opponents ?? []).filter(
    (row) => matchupConclusion(row.posterior) === 'disadvantage',
  ),
  'weakest',
)[0] ?? null)

const insights = computed(() => {
  if (!props.detail || props.detail.overall.evidence_state !== 'ready') return []
  const output = [
    `当前筛选覆盖 ${props.detail.opponent_deck_count} 套 exact opponent deck：`
      + `${conclusionCounts.value.advantage} 个明确优势、`
      + `${conclusionCounts.value.disadvantage} 个明确劣势、`
      + `${conclusionCounts.value.uncertain} 个尚不能判断。`,
  ]
  if (weakest.value) {
    output.push(
      `最大可信弱点是 ${weakest.value.opponent_display_name}：`
        + `${posteriorInterval(weakest.value.posterior)}，`
        + `${weakest.value.posterior.observed.games.toLocaleString()} 局。`,
    )
  }
  if (strongest.value) {
    output.push(
      `最高后验优势是 ${strongest.value.opponent_display_name}：`
        + `${posteriorInterval(strongest.value.posterior)}，`
        + `${strongest.value.posterior.observed.games.toLocaleString()} 局。`,
    )
  }
  return output
})

watch(() => props.detail, () => {
  expandedOpponent.value = null
})

function toggleOpponent(row: TrainingOpponentMatchupSummary) {
  expandedOpponent.value = expandedOpponent.value === row.opponent_deck_label
    ? null
    : row.opponent_deck_label
}

function intervalStyle(posterior: TrainingPosterior) {
  const low = intervalPosition(posterior.credible_low)
  const high = intervalPosition(posterior.credible_high)
  return { left: `${low}%`, width: `${Math.max(0, high - low)}%` }
}

function meanStyle(posterior: TrainingPosterior) {
  return { left: `${intervalPosition(posterior.posterior_mean)}%` }
}

function posteriorInterval(posterior: TrainingPosterior, digits = 1) {
  return `${percent(posterior.posterior_mean, digits)}`
    + ` [${percent(posterior.credible_low, digits)}–`
    + `${percent(posterior.credible_high, digits)}]`
}

function seatSummary() {
  const seats = props.detail?.seat_breakdown ?? []
  if (!seats.length) return '—'
  return seats
    .map((row) => `S${row.candidate_seat} ${percent(row.posterior.posterior_mean, 1)}`)
    .join(' / ')
}

</script>

<template>
  <div v-if="detail" class="matchup-profile" :class="{ loading }">
    <div class="matchup-profile-notice">
      以下结论只聚合当前筛选下的训练观测，不代表公开 meta 权重或部署 bundle 强度。
    </div>

    <div class="matchup-profile-summary">
      <div>
        <span>整体后验</span>
        <strong>{{ percent(detail.overall.posterior_mean, 1) }}</strong>
        <small>
          95% CI {{ percent(detail.overall.credible_low, 1) }}–{{ percent(detail.overall.credible_high, 1) }}
        </small>
      </div>
      <div>
        <span>观测证据</span>
        <strong>{{ detail.overall.observed.games.toLocaleString() }} 局</strong>
        <small>原始得分率 {{ percent(detail.overall.observed.score_rate, 1) }}</small>
      </div>
      <div>
        <span>Matchup 画像</span>
        <strong>
          <b class="profile-count--bad">{{ conclusionCounts.disadvantage }} 弱</b>
          · <b class="profile-count--good">{{ conclusionCounts.advantage }} 强</b>
        </strong>
        <small>{{ conclusionCounts.uncertain }} 未定 · {{ detail.pilot_count.toLocaleString() }} pilots</small>
      </div>
      <div>
        <span>Seat 对照</span>
        <strong>{{ seatSummary() }}</strong>
        <small>描述性观测；仅在对应 seat 有证据时显示</small>
      </div>
    </div>

    <div v-if="insights.length" class="matchup-profile-insights">
      <strong>诊断摘要</strong>
      <ul>
        <li v-for="insight in insights" :key="insight">{{ insight }}</li>
      </ul>
    </div>

    <div v-if="detail.stratum_breakdown.length" class="matchup-profile-strata">
      <span>按对手分层</span>
      <div
        v-for="stratumRow in detail.stratum_breakdown"
        :key="`${stratumRow.opponent_kind}:${stratumRow.opponent_stratum}`"
      >
        <b>{{ stratumLabel(stratumRow.opponent_stratum) }}</b>
        <strong>{{ percent(stratumRow.posterior.posterior_mean, 1) }}</strong>
        <small>
          {{ stratumRow.posterior.observed.games.toLocaleString() }} 局 ·
          {{ stratumRow.pilot_count }} pilots
        </small>
      </div>
    </div>

    <div class="matchup-profile-toolbar">
      <div class="matchup-profile-tabs" role="tablist" aria-label="Matchup 结论">
        <button
          :class="{ active: activeConclusion === 'disadvantage' }"
          type="button"
          role="tab"
          :aria-selected="activeConclusion === 'disadvantage'"
          @click="activeConclusion = 'disadvantage'"
        >
          明确劣势 <b>{{ conclusionCounts.disadvantage }}</b>
        </button>
        <button
          :class="{ active: activeConclusion === 'uncertain' }"
          type="button"
          role="tab"
          :aria-selected="activeConclusion === 'uncertain'"
          @click="activeConclusion = 'uncertain'"
        >
          尚不确定 <b>{{ conclusionCounts.uncertain }}</b>
        </button>
        <button
          :class="{ active: activeConclusion === 'advantage' }"
          type="button"
          role="tab"
          :aria-selected="activeConclusion === 'advantage'"
          @click="activeConclusion = 'advantage'"
        >
          明确优势 <b>{{ conclusionCounts.advantage }}</b>
        </button>
        <button
          :class="{ active: activeConclusion === 'all' }"
          type="button"
          role="tab"
          :aria-selected="activeConclusion === 'all'"
          @click="activeConclusion = 'all'"
        >
          全部 <b>{{ detail.opponent_deck_count }}</b>
        </button>
      </div>
      <n-select v-model:value="sortMode" :options="sortOptions" class="matchup-sort-select" />
    </div>

    <div class="matchup-spectrum-heading" aria-hidden="true">
      <span>Exact opponent deck</span>
      <span class="matchup-spectrum-axis-labels"><i>0%</i><i>50%</i><i>100%</i></span>
      <span>后验 / 95% CI</span>
      <span>证据</span>
      <span>结论</span>
    </div>

    <div v-if="visibleOpponents.length" class="matchup-spectrum">
      <article
        v-for="row in visibleOpponents"
        :key="row.opponent_deck_label"
        class="matchup-spectrum-item"
        :class="`matchup-spectrum-item--${matchupConclusion(row.posterior)}`"
      >
        <button
          type="button"
          class="matchup-spectrum-row"
          :aria-expanded="expandedOpponent === row.opponent_deck_label"
          @click="toggleOpponent(row)"
        >
          <span class="matchup-spectrum-deck">
            <strong>{{ row.opponent_display_name }}</strong>
            <small>
              旧编号 · {{ row.opponent_deck_hash ?? '不可用' }} · {{ row.pilot_count }} pilots
            </small>
          </span>
          <span
            class="matchup-interval"
            :aria-label="posteriorInterval(row.posterior)"
          >
            <i class="matchup-interval__midline" />
            <i
              class="matchup-interval__range"
              :style="intervalStyle(row.posterior)"
            />
            <i
              class="matchup-interval__mean"
              :style="meanStyle(row.posterior)"
            />
          </span>
          <span class="matchup-spectrum-score">
            <strong>{{ percent(row.posterior.posterior_mean, 1) }}</strong>
            <small>
              {{ percent(row.posterior.credible_low, 1) }}–{{ percent(row.posterior.credible_high, 1) }}
            </small>
          </span>
          <span class="matchup-spectrum-evidence">
            <strong>{{ row.posterior.observed.games.toLocaleString() }} 局</strong>
            <small>观测 {{ percent(row.posterior.observed.score_rate, 1) }}</small>
          </span>
          <span class="matchup-conclusion">
            {{ conclusionLabels[matchupConclusion(row.posterior)] }}
          </span>
          <span class="matchup-spectrum-chevron">
            {{ expandedOpponent === row.opponent_deck_label ? '−' : '+' }}
          </span>
        </button>

        <training-matchup-breakdown
          v-if="expandedOpponent === row.opponent_deck_label"
          :opponent="row"
          :matchups="detail.matchups"
        />
      </article>
    </div>
    <div v-else class="matchup-profile-empty">
      当前分类没有 opponent deck；可切换到“全部”或调整筛选条件。
    </div>
  </div>
  <div v-else class="matchup-profile-empty">
    {{ loading ? '正在聚合 matchup 画像…' : '当前筛选没有可用 matchup 观测。' }}
  </div>
</template>

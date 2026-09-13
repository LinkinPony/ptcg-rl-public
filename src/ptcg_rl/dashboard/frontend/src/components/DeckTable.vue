<script setup lang="ts">
import { computed } from 'vue'
import { percent } from '../format'
import type {
  TrainingController,
  TrainingDeckStanding,
  TrainingDeckStrengthPayload,
  TrainingEvidenceRange,
} from '../types'

const props = defineProps<{
  evidence: TrainingDeckStrengthPayload
  ranges: TrainingEvidenceRange[]
  sortRange: TrainingEvidenceRange
  sortDirection: 'asc' | 'desc'
  selectedDeckLabel: string | null
  familyFilter: string | null
  searchQuery: string
}>()
const emit = defineEmits<{
  sort: [range: TrainingEvidenceRange]
  select: [deckLabel: string]
}>()

interface TableRow {
  deck_label: string
  deck_hash: string
  display_name: string
  family_id: string | null
  family_display_name: string | null
  ranges: Partial<Record<TrainingEvidenceRange, TrainingDeckStanding>>
}

interface TableGroup {
  key: string
  family_id: string | null
  display_name: string
  rows: TableRow[]
}

const labels: Record<TrainingEvidenceRange, string> = {
  checkpoint: '所选 Checkpoint',
  recent_15m: '近 15 分钟',
  recent_60m: '近 60 分钟',
  cumulative: 'Run 累计',
}

const rows = computed<TableRow[]>(() => {
  const byLabel = new Map<string, TableRow>()
  for (const range of props.ranges) {
    for (const standing of props.evidence.ranges[range].standings) {
      const row = byLabel.get(standing.deck_label) ?? {
        deck_label: standing.deck_label,
        deck_hash: standing.deck_hash,
        display_name: standing.display_name,
        family_id: standing.family_id,
        family_display_name: standing.family_display_name,
        ranges: {},
      }
      row.ranges[range] = standing
      byLabel.set(standing.deck_label, row)
    }
  }
  const direction = props.sortDirection === 'desc' ? -1 : 1
  return [...byLabel.values()].sort((left, right) => {
    const leftValue = left.ranges[props.sortRange]?.posterior.posterior_mean
    const rightValue = right.ranges[props.sortRange]?.posterior.posterior_mean
    if (leftValue == null && rightValue == null) {
      return left.display_name.localeCompare(right.display_name)
    }
    if (leftValue == null) return 1
    if (rightValue == null) return -1
    return direction * (leftValue - rightValue)
  })
})

const groups = computed<TableGroup[]>(() => {
  const query = props.searchQuery.trim().toLocaleLowerCase()
  const visible = rows.value.filter((row) => {
    if (props.familyFilter != null && row.family_id !== props.familyFilter) {
      return false
    }
    if (!query) return true
    return [
      row.display_name,
      row.deck_label,
      row.deck_hash,
      row.family_display_name ?? '',
      row.family_id ?? '',
    ].some((value) => value.toLocaleLowerCase().includes(query))
  })
  const grouped = new Map<string, TableGroup>()
  for (const row of visible) {
    const key = row.family_id ?? '__unclassified__'
    const group = grouped.get(key) ?? {
      key,
      family_id: row.family_id,
      display_name: row.family_display_name ?? '未分类',
      rows: [],
    }
    group.rows.push(row)
    grouped.set(key, group)
  }
  return [...grouped.values()]
})

function score(
  standing: TrainingDeckStanding | undefined,
  controller: TrainingController,
) {
  return percent(standing?.controller_scores[controller]?.score_rate ?? null, 0)
}
</script>

<template>
  <div class="training-deck-table-wrap">
    <table class="training-deck-table">
      <thead>
        <tr>
          <th class="training-deck-table__deck">卡组</th>
          <th v-for="range in ranges" :key="range">
            <button
              class="range-sort-button"
              :class="{ active: sortRange === range }"
              @click="emit('sort', range)"
            >
              <span>{{ labels[range] }}</span>
              <small>{{ sortRange === range ? (sortDirection === 'desc' ? '↓' : '↑') : '点击排序' }}</small>
            </button>
          </th>
        </tr>
      </thead>
      <tbody v-for="group in groups" :key="group.key">
        <tr class="training-deck-family-row">
          <th :colspan="ranges.length + 1">
            <span>{{ group.display_name }}</span>
            <b>{{ group.rows.length }} 个 exact</b>
          </th>
        </tr>
        <tr
          v-for="row in group.rows"
          :key="row.deck_label"
          :class="{ selected: selectedDeckLabel === row.deck_label }"
          :aria-selected="selectedDeckLabel === row.deck_label"
          tabindex="0"
          @click="emit('select', row.deck_label)"
          @keydown.enter="emit('select', row.deck_label)"
          @keydown.space.prevent="emit('select', row.deck_label)"
        >
          <th class="training-deck-table__deck">
            <strong>{{ row.display_name }}</strong>
            <small class="mono">旧编号 · {{ row.deck_hash }}</small>
            <span>{{ row.deck_label }}</span>
          </th>
          <td v-for="range in ranges" :key="range">
            <template v-if="row.ranges[range]?.posterior.evidence_state === 'ready'">
              <div class="range-cell__headline">
                <span>#{{ row.ranges[range]?.rank }}</span>
                <strong>{{ percent(row.ranges[range]?.posterior.posterior_mean ?? null, 1) }}</strong>
                <b>{{ row.ranges[range]?.posterior.observed.games.toLocaleString() }} 局</b>
              </div>
              <div class="range-cell__stats">
                <span>
                  W-D-L
                  <b>
                    {{ row.ranges[range]?.posterior.observed.wins }}-{{ row.ranges[range]?.posterior.observed.draws }}-{{ row.ranges[range]?.posterior.observed.losses }}
                  </b>
                </span>
                <span>观测 <b>{{ percent(row.ranges[range]?.posterior.observed.score_rate ?? null, 1) }}</b></span>
                <span>
                  95% CI
                  <b>{{ percent(row.ranges[range]?.posterior.credible_low ?? null, 0) }}–{{ percent(row.ranges[range]?.posterior.credible_high ?? null, 0) }}</b>
                </span>
                <span>P(&gt;50%) <b>{{ percent(row.ranges[range]?.posterior.probability_above_half ?? null, 0) }}</b></span>
                <span>
                  Seat 0 / 1
                  <b>
                    {{ percent(row.ranges[range]?.seat_scores['0']?.score_rate ?? null, 0) }}
                    /
                    {{ percent(row.ranges[range]?.seat_scores['1']?.score_rate ?? null, 0) }}
                  </b>
                </span>
              </div>
              <div class="range-cell__controllers">
                <span>Self {{ score(row.ranges[range], 'self_play') }}</span>
                <span>Sentinel {{ score(row.ranges[range], 'sentinel') }}</span>
                <span>Adaptive {{ score(row.ranges[range], 'adaptive_history') }}</span>
                <span>Scripted {{ score(row.ranges[range], 'scripted') }}</span>
              </div>
            </template>
            <div v-else class="range-cell__unavailable">
              <strong>不可用</strong>
              <span>{{ evidence.ranges[range].metadata.unavailable_reason ?? '该卡组没有观测对局' }}</span>
            </div>
          </td>
        </tr>
      </tbody>
      <tbody v-if="!groups.length">
        <tr class="training-deck-table__empty">
          <td :colspan="ranges.length + 1">没有匹配当前 family 或搜索条件的卡组。</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

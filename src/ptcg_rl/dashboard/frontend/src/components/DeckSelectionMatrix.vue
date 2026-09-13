<script setup lang="ts">
import { computed } from 'vue'
import { percent } from '../format'
import type {
  DeckSelectionCell,
  DeckSelectionStanding,
} from '../types'

const props = defineProps<{
  standings: DeckSelectionStanding[]
  cells: DeckSelectionCell[]
  highlightedDeckId: string | null
}>()
const emit = defineEmits<{ select: [deckId: string] }>()

const cellMap = computed(() => new Map(
  props.cells.map((cell) => [`${cell.candidate_id}\u0000${cell.opponent_id}`, cell]),
))

function cell(candidateId: string, opponentId: string) {
  return cellMap.value.get(`${candidateId}\u0000${opponentId}`)
}

function background(value: number | null | undefined) {
  if (value == null) return 'rgba(148, 163, 184, .04)'
  const strength = Math.min(1, Math.abs(value - 0.5) * 2)
  return value >= 0.5
    ? `rgba(52, 211, 153, ${0.08 + strength * 0.34})`
    : `rgba(248, 113, 113, ${0.08 + strength * 0.32})`
}

function tooltip(row: DeckSelectionCell | undefined) {
  if (!row) return '没有证据'
  return [
    `${row.candidate_label} vs ${row.opponent_label}`,
    `得分 ${percent(row.score_rate, 1)}`,
    `W-D-L ${row.wins}-${row.draws}-${row.losses}`,
    `95% CI ${percent(row.credible_low, 1)}–${percent(row.credible_high, 1)}`,
    `Seat 0 / 1: ${row.seat_0_games} / ${row.seat_1_games}`,
  ].join('\n')
}
</script>

<template>
  <div class="deck-matrix-wrap">
    <table class="deck-matrix">
      <thead>
        <tr>
          <th class="deck-matrix__corner">候选卡组 ↓ / 对手 →</th>
          <th
            v-for="opponent in standings"
            :key="opponent.deck_id"
            :class="{ highlighted: opponent.deck_id === highlightedDeckId }"
            :title="opponent.deck_label"
          >
            <span>{{ opponent.rank }}</span>
            <small>{{ opponent.deck_label }}</small>
          </th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="candidate in standings"
          :key="candidate.deck_id"
          :class="{ highlighted: candidate.deck_id === highlightedDeckId }"
          @click="emit('select', candidate.deck_id)"
        >
          <th :title="candidate.deck_label">
            <span>#{{ candidate.rank }}</span>
            <strong>{{ candidate.deck_label }}</strong>
          </th>
          <td
            v-for="opponent in standings"
            :key="opponent.deck_id"
            :class="{
              diagonal: candidate.deck_id === opponent.deck_id,
              incomplete: cell(candidate.deck_id, opponent.deck_id)?.evidence_state !== 'ready'
                && candidate.deck_id !== opponent.deck_id,
              highlighted: opponent.deck_id === highlightedDeckId,
            }"
            :style="{
              background: candidate.deck_id === opponent.deck_id
                ? undefined
                : background(cell(candidate.deck_id, opponent.deck_id)?.score_rate),
            }"
            :title="candidate.deck_id === opponent.deck_id
              ? '同一卡组'
              : tooltip(cell(candidate.deck_id, opponent.deck_id))"
          >
            {{ candidate.deck_id === opponent.deck_id
              ? '—'
              : percent(cell(candidate.deck_id, opponent.deck_id)?.score_rate, 0) }}
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

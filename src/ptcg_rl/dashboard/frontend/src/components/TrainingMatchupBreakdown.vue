<script setup lang="ts">
import { computed } from 'vue'
import { percent } from '../format'
import { stratumLabel } from '../trainingMatchupProfile'
import type {
  TrainingMatchupDetail,
  TrainingOpponentMatchupSummary,
  TrainingPosterior,
} from '../types'

const props = defineProps<{
  opponent: TrainingOpponentMatchupSummary
  matchups: TrainingMatchupDetail[]
}>()

const pilotRows = computed(() => props.matchups
  .filter((row) => (
    row.opponent_deck_label === props.opponent.opponent_deck_label
  ))
  .sort((left, right) => (
    left.opponent_stratum.localeCompare(right.opponent_stratum)
    || left.opponent_id.localeCompare(right.opponent_id)
    || left.candidate_seat - right.candidate_seat
  )))

function posteriorInterval(posterior: TrainingPosterior) {
  return `${percent(posterior.posterior_mean, 1)}`
    + ` [${percent(posterior.credible_low, 1)}–`
    + `${percent(posterior.credible_high, 1)}]`
}
</script>

<template>
  <div class="matchup-breakdown">
    <div class="matchup-breakdown-groups">
      <section>
        <strong>Seat 分解</strong>
        <div class="matchup-breakdown-cards">
          <div v-for="seatRow in opponent.seat_breakdown" :key="seatRow.candidate_seat">
            <span>Seat {{ seatRow.candidate_seat }}</span>
            <b>{{ percent(seatRow.posterior.posterior_mean, 1) }}</b>
            <small>
              {{ seatRow.posterior.observed.games.toLocaleString() }} 局 ·
              {{ percent(seatRow.posterior.credible_low, 0) }}–{{ percent(seatRow.posterior.credible_high, 0) }}
            </small>
          </div>
        </div>
      </section>
      <section>
        <strong>Opponent stratum 分解</strong>
        <div class="matchup-breakdown-cards">
          <div
            v-for="stratumRow in opponent.stratum_breakdown"
            :key="`${stratumRow.opponent_kind}:${stratumRow.opponent_stratum}`"
          >
            <span>{{ stratumLabel(stratumRow.opponent_stratum) }}</span>
            <b>{{ percent(stratumRow.posterior.posterior_mean, 1) }}</b>
            <small>
              {{ stratumRow.posterior.observed.games.toLocaleString() }} 局 ·
              {{ stratumRow.pilot_count }} pilots
            </small>
          </div>
        </div>
      </section>
    </div>

    <div class="matchup-pilot-table">
      <table>
        <thead>
          <tr>
            <th>Pilot</th>
            <th>Stratum</th>
            <th>Seat</th>
            <th>W-D-L</th>
            <th>观测</th>
            <th>后验 / 95% CI</th>
            <th>P(&gt;50%)</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="pilotRow in pilotRows"
            :key="`${pilotRow.opponent_stratum}:${pilotRow.opponent_id}:${pilotRow.candidate_seat}`"
          >
            <td class="mono">{{ pilotRow.opponent_id || '不可用' }}</td>
            <td>{{ stratumLabel(pilotRow.opponent_stratum) }}</td>
            <td>Seat {{ pilotRow.candidate_seat }}</td>
            <td>
              {{ pilotRow.posterior.observed.wins }}-{{ pilotRow.posterior.observed.draws }}-{{ pilotRow.posterior.observed.losses }}
            </td>
            <td>{{ percent(pilotRow.posterior.observed.score_rate, 1) }}</td>
            <td>{{ posteriorInterval(pilotRow.posterior) }}</td>
            <td>{{ percent(pilotRow.posterior.probability_above_half, 0) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

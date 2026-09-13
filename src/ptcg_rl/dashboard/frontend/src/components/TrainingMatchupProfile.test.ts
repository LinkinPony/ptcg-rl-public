import { mount } from '@vue/test-utils'
import type {
  OutcomeStats,
  TrainingDeckMatchupsPayload,
  TrainingOpponentMatchupSummary,
  TrainingPosterior,
} from '../types'
import { matchupConclusion } from '../trainingMatchupProfile'
import TrainingMatchupProfile from './TrainingMatchupProfile.vue'

describe('TrainingMatchupProfile', () => {
  it('starts from credible disadvantages and drills into retained pilot cells', async () => {
    const wrapper = mount(TrainingMatchupProfile, {
      props: { detail: detailFixture(), loading: false },
      global: { stubs: { NSelect: true } },
    })

    expect(wrapper.findAll('.matchup-spectrum-item')).toHaveLength(1)
    expect(wrapper.find('.matchup-spectrum-item').text()).toContain('Opponent Weak')
    expect(wrapper.text()).toContain('1 个明确优势、1 个明确劣势、1 个尚不能判断')
    expect(wrapper.text()).toContain('旧编号 · weak12345678')

    await wrapper.find('.matchup-spectrum-row').trigger('click')
    expect(wrapper.find('.matchup-breakdown').exists()).toBe(true)
    expect(wrapper.find('.matchup-pilot-table').text()).toContain('pilot-weak')
    expect(wrapper.find('.matchup-pilot-table').text()).toContain('Seat 0')

    await wrapper.findAll('.matchup-profile-tabs button')[2].trigger('click')
    expect(wrapper.findAll('.matchup-spectrum-item')).toHaveLength(1)
    expect(wrapper.find('.matchup-spectrum-item').text()).toContain('Opponent Strong')
  })

  it('uses the complete 95% interval rather than the mean for conclusions', () => {
    expect(matchupConclusion(posterior(0.7, 0.55, 0.82))).toBe('advantage')
    expect(matchupConclusion(posterior(0.3, 0.18, 0.45))).toBe('disadvantage')
    expect(matchupConclusion(posterior(0.7, 0.5, 0.82))).toBe('uncertain')
    expect(matchupConclusion(posterior(0.3, 0.18, 0.5))).toBe('uncertain')
  })
})

function detailFixture(): TrainingDeckMatchupsPayload {
  const weak = opponent('weak', 'Opponent Weak', posterior(0.3, 0.2, 0.4))
  const uncertain = opponent(
    'uncertain',
    'Opponent Uncertain',
    posterior(0.52, 0.42, 0.62),
  )
  const strong = opponent(
    'strong',
    'Opponent Strong',
    posterior(0.75, 0.65, 0.85),
  )
  return {
    run_id: 'synthetic',
    candidate_deck_label: 'candidate',
    range: 'cumulative',
    checkpoint_version: 1,
    controller: 'all',
    candidate_seat: null,
    opponent_set: 'all',
    available: true,
    unavailable_reason: null,
    overall: posterior(0.55, 0.51, 0.59, 300),
    opponent_deck_count: 3,
    pilot_count: 3,
    seat_breakdown: [
      { candidate_seat: 0, posterior: posterior(0.53, 0.48, 0.58, 150) },
      { candidate_seat: 1, posterior: posterior(0.57, 0.52, 0.62, 150) },
    ],
    stratum_breakdown: [],
    opponents: [weak, uncertain, strong],
    matchups: [{
      opponent_kind: 'frozen',
      opponent_stratum: 'adaptive_history',
      opponent_deck_label: weak.opponent_deck_label,
      opponent_deck_hash: weak.opponent_deck_hash,
      opponent_display_name: weak.opponent_display_name,
      opponent_id: 'pilot-weak',
      candidate_seat: 0,
      posterior: weak.posterior,
    }],
  }
}

function opponent(
  suffix: string,
  displayName: string,
  value: TrainingPosterior,
): TrainingOpponentMatchupSummary {
  return {
    opponent_deck_label: `opponent-${suffix}`,
    opponent_deck_hash: `${suffix}12345678`,
    opponent_display_name: displayName,
    pilot_count: 1,
    posterior: value,
    seat_breakdown: [{ candidate_seat: 0, posterior: value }],
    stratum_breakdown: [{
      opponent_kind: 'frozen',
      opponent_stratum: 'adaptive_history',
      pilot_count: 1,
      posterior: value,
    }],
  }
}

function posterior(
  mean: number,
  low: number,
  high: number,
  games = 100,
): TrainingPosterior {
  return {
    observed: outcome(games, mean),
    posterior_mean: mean,
    credible_low: low,
    credible_high: high,
    probability_above_half: mean >= 0.5 ? 0.9 : 0.1,
    evidence_state: 'ready',
  }
}

function outcome(games: number, scoreRate: number): OutcomeStats {
  const wins = Math.round(games * scoreRate)
  return {
    games,
    wins,
    draws: 0,
    losses: games - wins,
    win_rate: scoreRate,
    score_rate: scoreRate,
  }
}

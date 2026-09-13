import { mount } from '@vue/test-utils'
import type {
  DeckSelectionCell,
  DeckSelectionStanding,
} from '../types'
import DeckSelectionMatrix from './DeckSelectionMatrix.vue'

describe('DeckSelectionMatrix', () => {
  it('renders directed scores and emits the selected candidate row', async () => {
    const standings: DeckSelectionStanding[] = [
      standing('deck-a', 'Deck A', 1),
      standing('deck-b', 'Deck B', 2),
    ]
    const cells: DeckSelectionCell[] = [
      cell('deck-a', 'Deck A', 'deck-b', 'Deck B', 0.75),
      cell('deck-b', 'Deck B', 'deck-a', 'Deck A', 0.25),
    ]
    const wrapper = mount(DeckSelectionMatrix, {
      props: { standings, cells, highlightedDeckId: null },
    })

    expect(wrapper.text()).toContain('75%')
    expect(wrapper.text()).toContain('25%')
    await wrapper.find('tbody tr').trigger('click')
    expect(wrapper.emitted('select')).toEqual([['deck-a']])
  })
})

function standing(
  deckId: string,
  deckLabel: string,
  rank: number,
): DeckSelectionStanding {
  return {
    rank,
    deck_id: deckId,
    deck_hash: `${deckId}-hash`,
    deck_label: deckLabel,
    games: 2,
    wins: 1,
    draws: 0,
    losses: 1,
    truncated: 0,
    opponent_count: 1,
    equal_score_rate: 0.5,
    posterior_mean: 0.5,
    credible_low: 0.1,
    credible_high: 0.9,
    worst_opponent_id: null,
    worst_opponent_label: null,
    worst_matchup_score: null,
    best_opponent_id: null,
    best_opponent_label: null,
    best_matchup_score: null,
    evidence_state: 'ready',
  }
}

function cell(
  candidateId: string,
  candidateLabel: string,
  opponentId: string,
  opponentLabel: string,
  scoreRate: number,
): DeckSelectionCell {
  return {
    candidate_id: candidateId,
    candidate_hash: `${candidateId}-hash`,
    opponent_id: opponentId,
    opponent_hash: `${opponentId}-hash`,
    candidate_label: candidateLabel,
    opponent_label: opponentLabel,
    games: 2,
    wins: scoreRate > 0.5 ? 2 : 0,
    draws: 0,
    losses: scoreRate > 0.5 ? 0 : 2,
    truncated: 0,
    agent_error_games: 0,
    seat_0_games: 1,
    seat_1_games: 1,
    score_rate: scoreRate,
    posterior_mean: scoreRate,
    credible_low: 0.1,
    credible_high: 0.9,
    evidence_state: 'ready',
  }
}

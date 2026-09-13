import { mount } from '@vue/test-utils'
import type {
  OutcomeStats,
  TrainingDeckStanding,
  TrainingDeckStrengthPayload,
  TrainingEvidenceRange,
} from '../types'
import DeckTable from './DeckTable.vue'

describe('DeckTable', () => {
  it('renders all selected evidence ranges and emits range sorting', async () => {
    const evidence = payload()
    const ranges: TrainingEvidenceRange[] = [
      'checkpoint',
      'recent_15m',
      'recent_60m',
      'cumulative',
    ]
    const wrapper = mount(DeckTable, {
      props: {
        evidence,
        ranges,
        sortRange: 'checkpoint',
        sortDirection: 'desc',
        selectedDeckLabel: null,
        familyFilter: null,
        searchQuery: '',
      },
    })

    expect(wrapper.text()).toContain('Deck A')
    expect(wrapper.text()).toContain('所选 Checkpoint')
    expect(wrapper.text()).toContain('近 15 分钟')
    expect(wrapper.text()).toContain('Run 累计')
    expect(wrapper.text()).toContain('Sentinel 60%')
    expect(wrapper.text()).toContain('Adaptive 60%')
    expect(wrapper.text()).not.toContain('Frozen')
    expect(wrapper.text()).not.toContain('Stationary')
    expect(wrapper.text()).toContain('60.0%')
    expect(wrapper.text()).toContain('Family A')
    expect(wrapper.text()).toContain('旧编号 · abcdef123456')
    expect(wrapper.text()).not.toContain('aaaaaaaaaaaa')
    expect(wrapper.find('.training-deck-table__shortlist').exists()).toBe(false)
    await wrapper.findAll('.range-sort-button')[1].trigger('click')
    expect(wrapper.emitted('sort')).toEqual([['recent_15m']])
  })
})

function payload(): TrainingDeckStrengthPayload {
  const rangeNames: TrainingEvidenceRange[] = [
    'checkpoint',
    'recent_15m',
    'recent_60m',
    'cumulative',
  ]
  const ranges = Object.fromEntries(
    rangeNames.map((range) => [
      range,
      {
        metadata: {
          range,
          available: true,
          unavailable_reason: null,
          started_at_utc: '2026-01-01T00:00:00Z',
          ended_at_utc: '2026-01-01T00:01:00Z',
          windows_considered: 1,
          windows_selected: 1,
          mixed_version_windows_excluded: 0,
          unknown_version_windows_excluded: 0,
        },
        standings: [standing()],
      },
    ]),
  ) as TrainingDeckStrengthPayload['ranges']
  return {
    run_id: 'synthetic',
    checkpoint_version: 1,
    checkpoint_pair_manifest_sha256: 'artifact',
    semantics: 'observed_training_distribution_wdl_posterior_v1',
    source_warning: 'Observed training evidence only.',
    exact_dimensions_available: true,
    active_deck_labels: ['main_abcdef123456'],
    families: [{
      family_id: 'f'.repeat(64),
      display_name: 'Family A',
      deck_labels: ['main_abcdef123456'],
    }],
    stationary_opponent_kinds: [],
    ranges,
    warnings: [],
  }
}

function standing(): TrainingDeckStanding {
  const outcome: OutcomeStats = {
    games: 10,
    wins: 6,
    draws: 0,
    losses: 4,
    win_rate: 0.6,
    score_rate: 0.6,
  }
  return {
    rank: 1,
    deck_label: 'main_abcdef123456',
    deck_hash: 'abcdef123456',
    deck_digest: 'a'.repeat(64),
    display_name: 'Deck A',
    family_id: 'f'.repeat(64),
    family_display_name: 'Family A',
    route_compatible: true,
    posterior: {
      observed: outcome,
      posterior_mean: 0.6,
      credible_low: 0.3,
      credible_high: 0.8,
      probability_above_half: 0.7,
      evidence_state: 'ready',
    },
    controller_scores: {
      all: outcome,
      self_play: outcome,
      sentinel: outcome,
      adaptive_history: outcome,
      scripted: outcome,
    },
    seat_scores: { 0: outcome, 1: outcome },
    opponent_count: 1,
    strongest_matchup: null,
    weakest_matchup: null,
  }
}

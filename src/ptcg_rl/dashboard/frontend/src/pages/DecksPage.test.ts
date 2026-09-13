import { flushPromises, mount } from '@vue/test-utils'
import { NSelect } from 'naive-ui'
import { nextTick } from 'vue'
import type {
  CheckpointInfo,
  OutcomeStats,
  TrainingDeckStanding,
  TrainingDeckStrengthPayload,
  TrainingEvidenceRange,
} from '../types'
import DeckTable from '../components/DeckTable.vue'
import TrainingControllerPanel from '../components/TrainingControllerPanel.vue'
import DecksPage from './DecksPage.vue'

const apiMocks = vi.hoisted(() => ({
  series: vi.fn(),
  matrix: vi.fn(),
  matchups: vi.fn(),
  startTask: vi.fn(),
}))

vi.mock('../api', () => ({
  api: {
    trainingDeckSeries: apiMocks.series,
    trainingDeckMatrix: apiMocks.matrix,
    trainingDeckMatchups: apiMocks.matchups,
    startTask: apiMocks.startTask,
  },
}))

describe('DecksPage', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.clearAllMocks()
    apiMocks.series.mockResolvedValue({
      available: true,
      series: [],
    })
    apiMocks.matrix.mockResolvedValue({
      available: true,
      cells: [],
    })
    apiMocks.matchups.mockResolvedValue({
      available: true,
      overall: {
        observed: {
          games: 0,
          wins: 0,
          draws: 0,
          losses: 0,
          win_rate: null,
          score_rate: null,
        },
        posterior_mean: null,
        credible_low: null,
        credible_high: null,
        probability_above_half: null,
        evidence_state: 'unavailable',
      },
      opponent_deck_count: 0,
      pilot_count: 0,
      seat_breakdown: [],
      stratum_breakdown: [],
      opponents: [],
      matchups: [],
    })
  })

  it('loads existing evidence without creating a task and persists the decision', async () => {
    const checkpoint = checkpointFixture()
    const wrapper = mount(DecksPage, {
      props: {
        run: 'synthetic',
        checkpoint,
        evidence: payload(),
      },
      global: {
        stubs: {
          TrendChart: true,
          MatchupHeatmap: true,
          TwoDeckRecommendationPanel: true,
          teleport: true,
        },
      },
    })
    await nextTick()

    expect(wrapper.findComponent(DeckTable).props('ranges')).toEqual([
      'checkpoint',
      'recent_15m',
      'recent_60m',
      'cumulative',
    ])
    expect(apiMocks.startTask).not.toHaveBeenCalled()
    expect(wrapper.findComponent(TrainingControllerPanel).props('range')).toBe(
      'checkpoint',
    )
    expect(wrapper.text()).toContain('Sentinel')
    expect(wrapper.text()).toContain('Adaptive history')
    expect(wrapper.text()).not.toContain('Frozen')
    expect(wrapper.text()).not.toContain('Stationary')
    expect(
      wrapper.findAllComponents(NSelect).map((select) => select.props('options')),
    ).toContainEqual([
      { label: '全部对手分层', value: 'all' },
      { label: 'Self-play', value: 'self_play' },
      { label: 'Sentinel', value: 'sentinel' },
      { label: 'Adaptive history', value: 'adaptive_history' },
      { label: 'Scripted', value: 'scripted' },
    ])
    expect(apiMocks.matrix).toHaveBeenCalledWith(
      'synthetic',
      'cumulative',
      checkpoint.version,
      'all',
      null,
      'active',
    )

    wrapper.findComponent(TrainingControllerPanel).vm.$emit(
      'update:range',
      'cumulative',
    )
    await nextTick()
    const viewState = JSON.parse(
      window.localStorage.getItem(
        'ptcg-training-deck-views:synthetic',
      ) ?? '{}',
    ) as Record<string, unknown>
    expect(viewState.controllerRange).toBe('cumulative')

    wrapper.findComponent(DeckTable).vm.$emit('select', 'main_abcdef123456')
    await flushPromises()
    await wrapper.find('.decision-primary-action').trigger('click')

    const stored = JSON.parse(
      window.localStorage.getItem(
        `ptcg-training-deck-decision:synthetic:${checkpoint.pair_manifest_sha256}`,
      ) ?? '{}',
    ) as Record<string, unknown>
    expect(stored).not.toHaveProperty('shortlist')
    expect(stored.finalDeckLabel).toBe('main_abcdef123456')
    expect(wrapper.find('.training-deck-table__shortlist').exists()).toBe(false)
    expect(wrapper.text()).toContain('Family A')
    expect(apiMocks.matchups).toHaveBeenCalledWith(
      'synthetic',
      'main_abcdef123456',
      'checkpoint',
      checkpoint.version,
      'all',
      null,
      'all',
    )

    const matrixCalls = apiMocks.matrix.mock.calls.length
    const seriesCalls = apiMocks.series.mock.calls.length
    await wrapper.setProps({ refreshVersion: 1 })
    await flushPromises()

    expect(apiMocks.series).toHaveBeenCalledTimes(seriesCalls + 1)
    expect(apiMocks.matrix).toHaveBeenCalledTimes(matrixCalls)

    await wrapper.find('.matrix-refresh-button').trigger('click')
    await flushPromises()
    expect(apiMocks.matrix).toHaveBeenCalledTimes(matrixCalls + 1)
  })
})

function payload(): TrainingDeckStrengthPayload {
  const rangeNames: TrainingEvidenceRange[] = [
    'checkpoint',
    'recent_15m',
    'recent_60m',
    'cumulative',
  ]
  return {
    run_id: 'synthetic',
    checkpoint_version: 1,
    checkpoint_pair_manifest_sha256: 'pair',
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
    ranges: Object.fromEntries(rangeNames.map((range) => [
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
    ])) as TrainingDeckStrengthPayload['ranges'],
    warnings: [],
  }
}

function standing(): TrainingDeckStanding {
  const outcome: OutcomeStats = {
    games: 4,
    wins: 3,
    draws: 0,
    losses: 1,
    win_rate: 0.75,
    score_rate: 0.75,
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
      posterior_mean: 0.7,
      credible_low: 0.3,
      credible_high: 0.9,
      probability_above_half: 0.8,
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

function checkpointFixture(): CheckpointInfo {
  return {
    run_id: 'synthetic',
    version: 1,
    pair_manifest_sha256: 'pair',
    policy_sha256: 'policy',
    learner_state_sha256: 'learner',
    model_config_fingerprint: null,
    training_roster_fingerprint: null,
    exact_registry_fingerprint: null,
    active_exact_deck_digests: ['abcdef123456' + '0'.repeat(52)],
    updated_at_utc: '2026-01-01T00:00:00Z',
    metric_available: false,
    current: true,
  }
}

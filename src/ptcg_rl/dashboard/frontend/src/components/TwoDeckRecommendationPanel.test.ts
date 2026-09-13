import { flushPromises, mount } from '@vue/test-utils'
import type {
  CheckpointInfo,
  TwoDeckPair,
  TwoDeckRecommendationPayload,
} from '../types'
import TwoDeckRecommendationPanel from './TwoDeckRecommendationPanel.vue'

const apiMocks = vi.hoisted(() => ({
  recommendation: vi.fn(),
}))

vi.mock('../api', () => ({
  api: {
    twoDeckRecommendation: apiMocks.recommendation,
  },
}))

describe('TwoDeckRecommendationPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMocks.recommendation.mockResolvedValue(payload())
  })

  it('loads the selected checkpoint defaults and preserves both display hashes', async () => {
    const wrapper = mount(TwoDeckRecommendationPanel, {
      props: {
        run: 'synthetic-run',
        checkpoint: checkpoint(),
      },
    })
    await flushPromises()

    expect(apiMocks.recommendation).toHaveBeenCalledWith(
      'synthetic-run',
      'checkpoint',
      7,
      17,
    )
    expect(wrapper.text()).toContain('Alpha deck')
    expect(wrapper.text()).toContain('a1a2a3a4a5a6')
    expect(wrapper.text()).toContain('Beta deck')
    expect(wrapper.text()).toContain('b1b2b3b4b5b6')
    expect(wrapper.text()).toContain('不是 Kaggle μ/rank 预测')
    expect(wrapper.text()).toContain('也不是提交授权')
    expect(wrapper.text()).not.toContain('暂时无法生成双席推荐')

    await wrapper.setProps({ checkpoint: null })
    await flushPromises()
    expect(apiMocks.recommendation).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain('请先选择 immutable checkpoint')
  })
})

function payload(): TwoDeckRecommendationPayload {
  const recommendation = pair()
  return {
    schema_version: 1,
    available: true,
    semantics: 'training_matchups_kaggle_daily_meta_portfolio_proxy_v1',
    run_id: 'synthetic-run',
    checkpoint_version: 17,
    checkpoint_pair_manifest_sha256: 'pair-sha',
    training_range: 'checkpoint',
    training_started_at_utc: '2026-01-01T00:00:00Z',
    training_ended_at_utc: '2026-01-01T01:00:00Z',
    public_window_days: 7,
    public_as_of_date: '2026-01-02',
    public_snapshot_fingerprint: 'snapshot-sha',
    objective: 'expected_max_meta_weighted_score_proxy',
    quality: {
      ready: true,
      training_available: true,
      public_environment_available: true,
      active_candidates: 3,
      eligible_candidates: 2,
      candidate_pairs: 1,
      known_meta_mass: 0.9,
      unknown_meta_mass: 0.1,
      unexpanded_explicit_meta_mass: 0.08,
      rare_unknown_meta_mass: 0.02,
      warnings: [],
    },
    recommendation,
    pairs: [recommendation],
    candidates: [],
    warnings: [],
  }
}

function pair(): TwoDeckPair {
  return {
    rank: 1,
    deck_a_label: 'main_a1a2a3a4a5a6',
    deck_a_hash: 'a1a2a3a4a5a6',
    deck_a_digest: 'a'.repeat(64),
    deck_a_display_name: 'Alpha deck',
    deck_b_label: 'main_b1b2b3b4b5b6',
    deck_b_hash: 'b1b2b3b4b5b6',
    deck_b_digest: 'b'.repeat(64),
    deck_b_display_name: 'Beta deck',
    expected_best_score: 0.64,
    credible_low: 0.55,
    credible_high: 0.72,
    best_score_lcb: 0.56,
    probability_at_least_one_above_even: 0.82,
    joint_downside_probability: 0.18,
    diversification_gain: 0.04,
    score_correlation: -0.2,
    expected_regret: 0.01,
    common_weak_meta_mass: 0.08,
    shared_observed_meta_mass: 0.74,
  }
}

function checkpoint(): CheckpointInfo {
  return {
    run_id: 'synthetic-run',
    version: 17,
    pair_manifest_sha256: 'pair-sha',
    policy_sha256: 'policy-sha',
    learner_state_sha256: 'learner-sha',
    model_config_fingerprint: null,
    training_roster_fingerprint: null,
    exact_registry_fingerprint: null,
    active_exact_deck_digests: ['a'.repeat(64), 'b'.repeat(64)],
    updated_at_utc: '2026-01-01T00:00:00Z',
    metric_available: true,
    current: true,
  }
}

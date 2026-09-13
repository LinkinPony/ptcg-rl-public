import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import NowPage from './NowPage.vue'
import type { WorkbenchSummary } from '../types'

describe('NowPage', () => {
  it('renders posterior evidence separately from learner progress', () => {
    const wrapper = mount(NowPage, {
      props: { summary: summary(), loading: false },
    })

    expect(wrapper.text()).toContain('62.50%')
    expect(wrapper.text()).toContain('95% CI 50.0%–75.0%')
    expect(wrapper.text()).toContain('≥ 12,345')
    expect(wrapper.text()).toContain('当前段 345')
    expect(wrapper.text()).toContain('v7')
    expect(wrapper.text()).toContain('未发现需要介入的异常')
  })
})

function summary(): WorkbenchSummary {
  return {
    run: {
      run_id: 'synthetic',
      display_name: 'Synthetic',
      current: true,
      lineage_id: null,
      schema_version: 2,
      updated_at_utc: '2026-01-01T00:00:00Z',
      minute_index: 1,
      data_state: 'ready',
      detail: null,
    },
    data_age_seconds: 3,
    data_state: 'ready',
    warnings: [],
    checkpoint: {
      run_id: 'synthetic',
      version: 6,
      pair_manifest_sha256: 'a'.repeat(64),
      policy_sha256: 'b'.repeat(64),
      learner_state_sha256: 'c'.repeat(64),
    model_config_fingerprint: null,
    training_roster_fingerprint: null,
    exact_registry_fingerprint: null,
    active_exact_deck_digests: [],
    updated_at_utc: '2026-01-01T00:00:00Z',
      metric_available: true,
      current: true,
    },
    training_games: {
      semantics: 'terminal_outcome_telemetry_sum_v1',
      total_games: 12345,
      selected_run_games: 345,
      other_runs_games: 12000,
      counted_run_count: 4,
      unreported_run_count: 2,
      invalid_run_count: 0,
      is_lower_bound: true,
      count_stages: ['scored_terminal_single_writer'],
      updated_at_utc: '2026-01-01T00:00:00Z',
    },
    progress: {
      update_index: 7,
      target_updates: 10,
      optimizer_step_index: 14,
      decisions_seen: 100,
      target_decisions: 200,
      kept_decisions_per_second: 25,
    },
    latest_learner: {
      loss: 0.1,
      policy_loss: 0.01,
      value_loss: 0.2,
      belief_loss: 0.03,
      entropy: 0.7,
      approximate_kl: 0.01,
      clip_fraction: 0.1,
      gradient_norm: 0.5,
      learning_rate: 0.0003,
      collection_seconds: 1,
      learner_seconds: 2,
      checkpoint_seconds: 3,
      cuda_peak_allocated_bytes: 1024,
      cuda_peak_reserved_bytes: 2048,
      fragments_stale: 0,
    },
    evidence: {
      observed: {
        games: 8,
        wins: 5,
        draws: 0,
        losses: 3,
        win_rate: 0.625,
        score_rate: 0.625,
      },
      seat_games: { 0: 4, 1: 4 },
      seat_balanced: true,
      posterior_mean: 0.625,
      credible_low: 0.5,
      credible_high: 0.75,
      probability_above_half: 0.8,
      evidence_state: 'ready',
    },
    alerts: [{
      code: 'healthy',
      severity: 'info',
      title: '未发现需要介入的异常',
      detail: 'Synthetic healthy state.',
    }],
  }
}

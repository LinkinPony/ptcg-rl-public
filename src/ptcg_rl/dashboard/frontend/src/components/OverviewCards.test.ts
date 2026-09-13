import { mount } from '@vue/test-utils'
import { NConfigProvider } from 'naive-ui'
import { defineComponent, h } from 'vue'
import type { HealthPayload, OutcomeStats, PerformanceTable } from '../types'
import OverviewCards from './OverviewCards.vue'

describe('OverviewCards', () => {
  it('renders the canonical opponent-strata score cards', () => {
    const outcome = (
      games: number,
      scoreRate: number,
    ): OutcomeStats => ({
      games,
      wins: games,
      draws: 0,
      losses: 0,
      win_rate: scoreRate,
      score_rate: scoreRate,
    })
    const table: PerformanceTable = {
      run_id: 'strata',
      scope: 'segment',
      window: 'cumulative',
      started_at_utc: null,
      ended_at_utc: null,
      source_segments: ['strata'],
      overall: {
        all: outcome(10, 0.5),
        self_play: outcome(2, 0.4),
        sentinel: outcome(2, 0.6),
        adaptive_history: outcome(3, 0.7),
        scripted: outcome(3, 0.8),
      },
      decks: [],
      exact_dimensions_available: true,
    }
    const host = defineComponent({
      setup: () => () => h(NConfigProvider, null, {
        default: () => h(OverviewCards, { table, health: null }),
      }),
    })

    const text = mount(host).text()

    expect(text).toContain('综合得分率50.00%')
    expect(text).toContain('Self-play40.00%')
    expect(text).toContain('Sentinel60.00%')
    expect(text).toContain('Adaptive history70.00%')
    expect(text).toContain('Scripted80.00%')
  })

  it('renders recurrent inference snapshot admission state', () => {
    const health: HealthPayload = {
      run_id: 'rolling-gap',
      data_age_seconds: 3,
      data_state: 'ready',
      warnings: [],
      status: {
        inference: {
          decisions_per_second: 742.35,
          last_sync: {
            current_weight_version: 31863,
            deferred_weight_version: 31870,
            deferred_weight_reason: 'version_gap',
            snapshot_min_version_gap: 8,
            snapshot_pool: {
              resident_versions: [31855, 31863],
              leases_by_version: {
                '31855': 121,
                '31863': 4321,
              },
            },
          },
        },
      },
    }
    const host = defineComponent({
      setup: () => () => h(NConfigProvider, null, {
        default: () => h(OverviewCards, { table: null, health }),
      }),
    })

    const text = mount(host).text()

    expect(text).toContain('742.4 dec/s')
    expect(text).toContain('v31863')
    expect(text).toContain('v31870 · version_gap')
    expect(text).toContain('v31855, v31863')
    expect(text).toContain('v31855 · 121')
    expect(text).toContain('v31863 · 4321')
  })

  it('renders recurrent game recycling and actor watchdog telemetry', () => {
    const health: HealthPayload = {
      run_id: 'rolling-recycle',
      data_age_seconds: 1,
      data_state: 'ready',
      warnings: [],
      status: {
        actors: {
          rollout_features: {
            live_game_count: 3072,
            live_game_steps_p50: 38,
            live_game_steps_p90: 112,
            live_game_steps_max: 181,
            max_live_game_steps_seen: 244,
            stale_recurrent_games_recycled: 17,
            stale_recurrent_games_deferred_pending_evidence: 3,
            stale_recurrent_sequences_released: 34,
            stale_recurrent_oldest_candidate_policy_version: 31863,
            stale_recurrent_max_candidate_version_age: 18,
          },
        },
        runtime: {
          actor_recycle_requests: 2,
          actor_recycles: 1,
        },
      },
    }
    const host = defineComponent({
      setup: () => () => h(NConfigProvider, null, {
        default: () => h(OverviewCards, { table: null, health }),
      }),
    })

    const text = mount(host).text()

    expect(text).toContain('循环序列存活性')
    expect(text).toContain('3,072 live · p50 38 · p90 112 · max 181 · peak 244')
    expect(text).toContain('18 versions · oldest v31863')
    expect(text).toContain('17 games · 34 seq')
    expect(text).toContain('3 observations')
    expect(text).toContain('deferred observed')
    expect(text).toContain('req 2 · done 1')
  })

  it('renders learner staleness refill telemetry without actor status', () => {
    const health: HealthPayload = {
      run_id: 'rolling-refill',
      data_age_seconds: 1,
      data_state: 'ready',
      warnings: [],
      status: {
        learner_status: {
          staleness_refill: {
            accumulated_raw_decisions: 27_000,
            stale_decisions: 2_424,
            retained_budget_decisions: 24_576,
            remaining_decisions: 0,
            topup_chunks: 1,
          },
        },
      },
    }
    const host = defineComponent({
      setup: () => () => h(NConfigProvider, null, {
        default: () => h(OverviewCards, { table: null, health }),
      }),
    })

    const text = mount(host).text()

    expect(text).toContain('循环序列存活性')
    expect(text).toContain('Learner refill')
    expect(text).toContain(
      'raw 27,000 · stale 2,424 · retained 24,576 · remaining 0 · topups 1',
    )
  })

  it('renders simple-stateless learner progress and allocator telemetry', () => {
    const health: HealthPayload = {
      run_id: 'stateless',
      data_age_seconds: 2,
      data_state: 'ready',
      warnings: [],
      status: {
        learner_status: {
          format: 'simple-stateless-training-status-v1',
          update_index: 3,
          target_updates: 100,
          throughput: {
            windows: 3,
            kept_decisions: 12_345,
            kept_decisions_per_second: 678.9,
          },
          latest_collection: {
            games_started: 96,
            games_finished: 96,
            candidate_decisions: 8_000,
            mirror_opponent_decisions: 4_000,
            decisions_per_second: 700.5,
          },
          latest_timing: {
            cuda_checkpoint_peak_allocated_bytes: 16 * 1024 ** 3,
            cuda_checkpoint_peak_reserved_bytes: 17 * 1024 ** 3,
            cuda_checkpoint_allocation_retries: 0,
            cuda_checkpoint_ooms: 0,
          },
        },
      },
    }
    const host = defineComponent({
      setup: () => () => h(NConfigProvider, null, {
        default: () => h(OverviewCards, { table: null, health }),
      }),
    })

    const text = mount(host).text()

    expect(text).toContain('Simple-stateless learner')
    expect(text).toContain('3 / 100')
    expect(text).toContain('3 windows · 12,345 kept · 678.9 kept/s')
    expect(text).toContain('96 / 96 games · 8,000 candidate · 4,000 mirror-opponent')
    expect(text).toContain('active 16.0 GiB · reserved 17.0 GiB · retries 0 · OOM 0')
    expect(text).toContain('healthy')
  })

  it('does not render recurrent telemetry for legacy health payloads', () => {
    const health: HealthPayload = {
      run_id: 'legacy',
      data_age_seconds: null,
      data_state: 'stale',
      warnings: [],
      status: {},
    }
    const host = defineComponent({
      setup: () => () => h(NConfigProvider, null, {
        default: () => h(OverviewCards, { table: null, health }),
      }),
    })

    const text = mount(host).text()

    expect(text).not.toContain('循环序列存活性')
    expect(text).toContain('— dec/s')
  })
})

import { buildTrainingDeckDecision } from './deckSelectionExport'
import type {
  CheckpointInfo,
  TrainingDeckStanding,
  TrainingDeckStrengthPayload,
} from './types'

describe('training deck decision export', () => {
  it('binds the checkpoint, selected evidence ranges, family, and final deck', () => {
    const standing = standingFixture()
    const evidence = evidenceFixture(standing)
    const checkpoint = checkpointFixture()

    const output = buildTrainingDeckDecision({
      format: 'json',
      runId: 'synthetic',
      checkpoint,
      evidence,
      selectedRanges: ['checkpoint', 'cumulative'],
      finalDeckLabel: standing.deck_label,
    })

    expect(output?.checkpoint.pair_manifest_sha256).toBe('pair')
    expect(output?.checkpoint.active_exact_deck_digests).toEqual([])
    expect(output?.selected_ranges).toEqual(['checkpoint', 'cumulative'])
    expect(output?.final_deck.deck_hash).toBe(standing.deck_hash)
    expect(output?.final_deck.family_id).toBe('f'.repeat(64))
    expect(output?.final_deck.evidence.checkpoint?.posterior.observed.games).toBe(2)
  })
})

function evidenceFixture(
  standing: TrainingDeckStanding,
): TrainingDeckStrengthPayload {
  const metadata = {
    available: true,
    unavailable_reason: null,
    started_at_utc: null,
    ended_at_utc: null,
    windows_considered: 1,
    windows_selected: 1,
    mixed_version_windows_excluded: 0,
    unknown_version_windows_excluded: 0,
  }
  return {
    run_id: 'synthetic',
    checkpoint_version: 1,
    checkpoint_pair_manifest_sha256: 'pair',
    semantics: 'observed_training_distribution_wdl_posterior_v1',
    source_warning: 'Observed training evidence only.',
    exact_dimensions_available: true,
    active_deck_labels: [standing.deck_label],
    families: [{
      family_id: 'f'.repeat(64),
      display_name: 'Family A',
      deck_labels: [standing.deck_label],
    }],
    stationary_opponent_kinds: [],
    ranges: {
      checkpoint: {
        metadata: { ...metadata, range: 'checkpoint' },
        standings: [standing],
      },
      recent_15m: {
        metadata: { ...metadata, range: 'recent_15m' },
        standings: [standing],
      },
      recent_60m: {
        metadata: { ...metadata, range: 'recent_60m' },
        standings: [standing],
      },
      cumulative: {
        metadata: { ...metadata, range: 'cumulative' },
        standings: [standing],
      },
    },
    warnings: [],
  }
}

function standingFixture(): TrainingDeckStanding {
  const observed = {
    games: 2,
    wins: 1,
    draws: 0,
    losses: 1,
    win_rate: 0.5,
    score_rate: 0.5,
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
      observed,
      posterior_mean: 0.5,
      credible_low: 0.1,
      credible_high: 0.9,
      probability_above_half: 0.5,
      evidence_state: 'ready',
    },
    controller_scores: { all: observed },
    seat_scores: { 0: observed, 1: observed },
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
    active_exact_deck_digests: [],
    updated_at_utc: '2026-01-01T00:00:00Z',
    metric_available: false,
    current: true,
  }
}

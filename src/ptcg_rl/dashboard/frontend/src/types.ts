export type WindowName = 'cumulative' | '15m' | '60m'
export type ScopeName = 'segment' | 'lineage'
export type HeatmapPaletteName = 'semantic' | 'viridis' | 'icefire' | 'teal'
export type TrainingEvidenceRange =
  | 'checkpoint'
  | 'recent_15m'
  | 'recent_60m'
  | 'cumulative'
export type TrainingController =
  | 'all'
  | 'self_play'
  | 'sentinel'
  | 'adaptive_history'
  | 'scripted'
export type OpponentSet = 'active' | 'all'

export interface OutcomeStats {
  games: number
  wins: number
  draws: number
  losses: number
  win_rate: number | null
  score_rate: number | null
}

export interface RunInfo {
  run_id: string
  display_name: string
  current: boolean
  lineage_id: string | null
  schema_version: number
  updated_at_utc: string | null
  minute_index: number
  data_state: 'ready' | 'stale' | 'invalid'
  detail: string | null
}

export interface DeckRow {
  deck_label: string
  deck_hash: string
  display_name: string
  slices: Record<string, OutcomeStats>
}

export interface PerformanceTable {
  run_id: string
  scope: ScopeName
  window: WindowName
  started_at_utc: string | null
  ended_at_utc: string | null
  source_segments: string[]
  overall: Record<string, OutcomeStats>
  decks: DeckRow[]
  exact_dimensions_available: boolean
}

export interface SeriesPoint {
  minute_index: number
  ended_at_utc: string
  games: number
  win_rate: number | null
  score_rate: number | null
}

export interface DeckSeries {
  deck_label: string
  deck_hash: string
  display_name: string
  points: SeriesPoint[]
}

export interface MatchupRow {
  candidate_deck_label: string
  candidate_deck_hash: string | null
  candidate_display_name: string
  opponent_kind: string
  opponent_deck_label: string
  opponent_deck_hash: string | null
  opponent_display_name: string
  opponent_id: string
  candidate_seat: number
  outcomes: OutcomeStats
}

export interface InferenceSnapshotPoolStatus {
  current_version?: number | null
  resident_versions?: number[]
  leases_by_version?: Record<string, number>
}

export interface InferenceLastSyncStatus {
  current_weight_version?: number | null
  deferred_weight_version?: number | null
  deferred_weight_reason?: string | null
  snapshot_min_version_gap?: number | null
  snapshot_pool?: InferenceSnapshotPoolStatus | null
}

export interface InferenceHealthStatus {
  decisions_per_second?: number | null
  last_sync?: InferenceLastSyncStatus | null
}

export interface LearnerStalenessRefillStatus {
  chunks?: number | null
  accumulated_raw_decisions?: number | null
  stale_decisions?: number | null
  retained_budget_decisions?: number | null
  remaining_decisions?: number | null
  topup_chunks?: number | null
}

export interface LearnerHealthStatus {
  format?: string | null
  update_index?: number | null
  target_updates?: number | null
  elapsed_seconds?: number | null
  throughput?: {
    windows?: number | null
    kept_decisions?: number | null
    kept_decisions_per_second?: number | null
  } | null
  latest_collection?: {
    games_started?: number | null
    games_finished?: number | null
    candidate_decisions?: number | null
    mirror_opponent_decisions?: number | null
    decisions_per_second?: number | null
  } | null
  latest_timing?: {
    cuda_checkpoint_peak_allocated_bytes?: number | null
    cuda_checkpoint_peak_reserved_bytes?: number | null
    cuda_checkpoint_allocation_retries?: number | null
    cuda_checkpoint_ooms?: number | null
  } | null
  staleness_refill?: LearnerStalenessRefillStatus | null
}

export interface OldestLiveGameStatus {
  actor_index?: number | null
  game_id?: string | null
  steps?: number | null
}

export interface ActorRolloutFeaturesStatus {
  live_game_count?: number | null
  live_game_steps_min?: number | null
  live_game_steps_mean?: number | null
  live_game_steps_p50?: number | null
  live_game_steps_p90?: number | null
  live_game_steps_p99?: number | null
  live_game_steps_max?: number | null
  max_live_game_steps_seen?: number | null
  oldest_live_games?: OldestLiveGameStatus[]
  stale_recurrent_recycle_polls?: number | null
  stale_recurrent_candidate_sequences_examined?: number | null
  stale_recurrent_games_recycled?: number | null
  stale_recurrent_games_deferred_pending_evidence?: number | null
  stale_recurrent_sequences_released?: number | null
  stale_recurrent_last_learner_version?: number | null
  stale_recurrent_last_served_version?: number | null
  stale_recurrent_oldest_candidate_policy_version?: number | null
  stale_recurrent_max_candidate_version_age?: number | null
}

export interface ActorHealthStatus {
  actors?: number | null
  rollout_features?: ActorRolloutFeaturesStatus | null
}

export interface RuntimeHealthStatus {
  actor_recycle_requests?: number | null
  actor_recycle_requests_by_actor?: Record<string, number> | null
  actor_recycles?: number | null
  actor_recycles_by_actor?: number[] | null
}

export interface SupervisorHealthStatus {
  actor_recycles?: number | null
  actor_recycles_by_actor?: number[] | null
}

export interface RunSummaryHealthStatus {
  supervisor?: SupervisorHealthStatus | null
}

export interface OperationalHealthStatus {
  inference?: InferenceHealthStatus | null
  learner_status?: LearnerHealthStatus | null
  actors?: ActorHealthStatus | null
  runtime?: RuntimeHealthStatus | null
  summary?: RunSummaryHealthStatus | null
  [key: string]: unknown
}

export interface HealthPayload {
  run_id: string
  data_age_seconds: number | null
  data_state: 'ready' | 'stale' | 'invalid'
  status: OperationalHealthStatus
  warnings: string[]
}

export type WorkbenchView =
  | 'now'
  | 'learning'
  | 'allocation'
  | 'decks'
  | 'compare'
  | 'tasks'
  | 'league'

export type AllocationPortfolio =
  | 'counter'
  | 'frontier'
  | 'probe'
  | 'rehearsal'
  | 'staleness'

export type AllocationRole =
  | 'protected'
  | 'recent'
  | 'counter'
  | 'frontier'
  | 'recovery'

export type AllocationMode = 'adaptive' | 'role_budget'

export type AllocationSort =
  | 'debt'
  | 'weakness'
  | 'uncertainty'
  | 'target'
  | 'change'
  | 'planned_games'

export interface OpponentAllocationCandidate {
  candidate_deck_digest: string
  candidate_deck_label: string | null
  candidate_deck_hash: string | null
  candidate_display_name: string
  base_share: number
  target_share: number
  previous_target_share: number | null
  posterior_score: number
  worst_posterior_score: number
  matchup_cells: number
  evidence_cells: number
  planned_games: number
  actual_games: number
  actual_decisions: number
}

export interface OpponentAllocationPortfolio {
  portfolio: AllocationPortfolio
  target_share: number
  planned_games: number
  planned_expected_decisions: number
  actual_games: number
  actual_decisions: number
  actual_decision_share: number
}

export interface OpponentAllocationArtifact {
  artifact_id: string
  source_fingerprint: string
  source_policy_version: number
  stratum: string
  role: AllocationRole | null
  route_count: number
  target_share: number
  posterior_score: number | null
  worst_posterior_score: number | null
  evidence_cells: number | null
  planned_games: number
  planned_expected_decisions: number | null
  actual_games: number
  actual_decisions: number
}

export interface OpponentAllocationRole {
  role: AllocationRole
  target_share: number
  artifact_count: number
  planned_games: number
  planned_expected_decisions: number
  actual_games: number
  actual_decisions: number
  actual_decision_share: number
}

export interface OpponentAllocationExecution {
  window_id: string
  window_sequence: number
  window_state: 'idle' | 'collecting' | 'committed' | 'aborted'
  exposure_cohort_games: number
  initial_wave_workers_issued: number
  initial_wave_workers_total: number
  assignment_pool_issued: number
  assignment_pool_total: number
  target_decisions: number
  accepted_decisions: number
  provisional_decisions: number
  inflight_decision_credit: number
  shards_issued: number
  shards_completed: number
}

export interface OpponentAllocationSummary {
  available: boolean
  run_id: string
  allocation_mode: AllocationMode | null
  recorded_at_utc: string | null
  detail: string | null
  window_sequence: number | null
  plan_id: string | null
  target_fingerprint: string | null
  predecessor_state_fingerprint: string | null
  committed_state_fingerprint: string | null
  revision_fingerprint: string | null
  evidence_cells: number
  low_evidence_cells: number
  matchup_count: number
  execution: OpponentAllocationExecution | null
  candidates: OpponentAllocationCandidate[]
  portfolios: OpponentAllocationPortfolio[]
  roles: OpponentAllocationRole[]
  artifacts: OpponentAllocationArtifact[]
}

export interface OpponentAllocationMatchup {
  candidate_deck_digest: string
  candidate_deck_label: string | null
  candidate_deck_hash: string | null
  candidate_display_name: string
  artifact_id: string
  source_fingerprint: string
  source_policy_version: number
  stratum: string
  route_id: string
  opponent_deck_digest: string
  opponent_deck_label: string | null
  opponent_deck_hash: string | null
  opponent_display_name: string
  candidate_seat: 0 | 1
  portfolio: AllocationPortfolio | null
  role: AllocationRole | null
  target_share: number
  global_target_share: number
  previous_target_share: number | null
  target_delta: number | null
  posterior_score: number
  slow_score: number
  posterior_stddev: number
  effective_evidence: number
  utility: number | null
  components: Record<AllocationPortfolio, number>
  expected_decisions_per_game: number
  decision_mass: number
  normalized_decision_mass: number | null
  decision_debt_before: number | null
  projected_decision_debt: number | null
  planned_games: number
  planned_expected_decisions: number
  actual_games: number
  actual_decisions: number
  actual_score_sum: number | null
}

export interface OpponentAllocationMatchupPage {
  available: boolean
  run_id: string
  allocation_mode: AllocationMode | null
  window_sequence: number | null
  target_fingerprint: string | null
  total: number
  offset: number
  limit: number
  sort: AllocationSort
  candidate_deck_digest: string | null
  artifact_id: string | null
  portfolio: AllocationPortfolio | null
  role: AllocationRole | null
  candidate_seat: 0 | 1 | null
  rows: OpponentAllocationMatchup[]
}

export interface LeagueSummary {
  available: boolean
  controllers: number
  decks: number
  bundles: number
  candidates: number
  incumbents: number
  queued: number
  leased: number
  completed: number
  unresolved: number
  rating_events: number
  workers: number
}

export interface LeagueStanding {
  identity: string
  label: string
  kind: 'checkpoint' | 'script' | 'deck' | 'bundle'
  controller_id: string | null
  controller_label: string | null
  deck_digest: string | null
  deck_hash: string | null
  deck_label: string | null
  mu: number
  sigma: number
  conservative: number
  percentile: number
  p_top20: number | null
  games: number
  wins: number
  draws: number
  losses: number
  unresolved: number
  bundle_count: number
  rank_eligible: boolean
  active: boolean
  candidate_kind: string | null
  candidate_state: string | null
  decided_games: number | null
  decision_reason: string | null
  aliases: string[]
}

export interface LeagueWorker {
  worker_id: string
  hostname: string
  source_commit: string
  runtime_fingerprint: string
  belief_fingerprint: string
  current_match_id: string | null
  games_completed: number
  games_per_hour: number
  errors: number
  first_seen_at: string
  last_seen_at: string
  resources: Record<string, unknown>
}

export interface LeagueMatchup {
  side_a_bundle_id: string
  side_b_bundle_id: string
  games: number
  wins: number
  draws: number
  losses: number
  unresolved: number
}

export interface CheckpointInfo {
  run_id: string
  version: number
  pair_manifest_sha256: string
  policy_sha256: string
  learner_state_sha256: string
  model_config_fingerprint: string | null
  training_roster_fingerprint: string | null
  exact_registry_fingerprint: string | null
  active_exact_deck_digests: string[]
  updated_at_utc: string
  metric_available: boolean
  current: boolean
}

export interface PosteriorStats {
  observed: OutcomeStats
  seat_games: Record<string, number>
  seat_balanced: boolean
  posterior_mean: number | null
  credible_low: number | null
  credible_high: number | null
  probability_above_half: number | null
  evidence_state: 'ready' | 'missing_seat' | 'unavailable'
}

export interface DeckEvidence {
  deck_label: string
  deck_hash: string
  display_name: string
  posterior: PosteriorStats
  controller_scores: Record<string, OutcomeStats>
  opponent_count: number
}

export interface DeckEvidencePayload {
  run_id: string
  window: WindowName
  semantics: 'training_pool_wdl_posterior_v1'
  overall: PosteriorStats
  decks: DeckEvidence[]
  exact_dimensions_available: boolean
}

export interface MatchupEvidence {
  opponent_kind: string
  opponent_deck_label: string
  opponent_display_name: string
  opponent_id: string
  posterior: PosteriorStats
}

export interface DeckMatchupPayload {
  run_id: string
  candidate_deck_label: string
  window: WindowName
  matchups: MatchupEvidence[]
}

export interface TrainingPosterior {
  observed: OutcomeStats
  posterior_mean: number | null
  credible_low: number | null
  credible_high: number | null
  probability_above_half: number | null
  evidence_state: 'ready' | 'unavailable'
}

export interface TrainingMatchupSummary {
  opponent_deck_label: string
  opponent_deck_hash: string | null
  opponent_display_name: string
  outcomes: OutcomeStats
}

export interface TrainingDeckStanding {
  rank: number | null
  deck_label: string
  deck_hash: string
  deck_digest: string | null
  display_name: string
  family_id: string | null
  family_display_name: string | null
  route_compatible: boolean | null
  posterior: TrainingPosterior
  controller_scores: Partial<Record<TrainingController, OutcomeStats>>
  seat_scores: Record<string, OutcomeStats>
  opponent_count: number
  strongest_matchup: TrainingMatchupSummary | null
  weakest_matchup: TrainingMatchupSummary | null
}

export interface TrainingDeckFamily {
  family_id: string
  display_name: string
  deck_labels: string[]
}

export interface TrainingEvidenceRangeMetadata {
  range: TrainingEvidenceRange
  available: boolean
  unavailable_reason: string | null
  started_at_utc: string | null
  ended_at_utc: string | null
  windows_considered: number
  windows_selected: number
  mixed_version_windows_excluded: number
  unknown_version_windows_excluded: number
}

export interface TrainingEvidenceRangeStandings {
  metadata: TrainingEvidenceRangeMetadata
  standings: TrainingDeckStanding[]
}

export interface TrainingDeckStrengthPayload {
  run_id: string
  checkpoint_version: number | null
  checkpoint_pair_manifest_sha256: string | null
  semantics: 'observed_training_distribution_wdl_posterior_v1'
  source_warning: string
  exact_dimensions_available: boolean
  active_deck_labels: string[]
  families: TrainingDeckFamily[]
  stationary_opponent_kinds: string[]
  ranges: Record<TrainingEvidenceRange, TrainingEvidenceRangeStandings>
  warnings: string[]
}

export interface TrainingSeriesPoint {
  minute_index: number
  ended_at_utc: string
  games: number
  score_rate: number | null
}

export interface TrainingDeckSeries {
  deck_label: string
  deck_hash: string
  display_name: string
  points: TrainingSeriesPoint[]
}

export interface TrainingDeckSeriesPayload {
  run_id: string
  range: TrainingEvidenceRange
  checkpoint_version: number | null
  controller: TrainingController
  available: boolean
  unavailable_reason: string | null
  series: TrainingDeckSeries[]
}

export interface TrainingMatrixCell {
  candidate_deck_label: string
  candidate_deck_hash: string
  candidate_display_name: string
  opponent_deck_label: string
  opponent_deck_hash: string | null
  opponent_display_name: string
  pilot_count: number
  posterior: TrainingPosterior
}

export interface TrainingDeckMatrixPayload {
  run_id: string
  range: TrainingEvidenceRange
  checkpoint_version: number | null
  controller: TrainingController
  candidate_seat: number | null
  opponent_set: OpponentSet
  available: boolean
  unavailable_reason: string | null
  cells: TrainingMatrixCell[]
}

export interface TrainingMatchupDetail {
  opponent_kind: string
  opponent_stratum: string
  opponent_deck_label: string
  opponent_deck_hash: string | null
  opponent_display_name: string
  opponent_id: string
  candidate_seat: number
  posterior: TrainingPosterior
}

export interface TrainingMatchupSeatSummary {
  candidate_seat: number
  posterior: TrainingPosterior
}

export interface TrainingMatchupStratumSummary {
  opponent_kind: string
  opponent_stratum: string
  pilot_count: number
  posterior: TrainingPosterior
}

export interface TrainingOpponentMatchupSummary {
  opponent_deck_label: string
  opponent_deck_hash: string | null
  opponent_display_name: string
  pilot_count: number
  posterior: TrainingPosterior
  seat_breakdown: TrainingMatchupSeatSummary[]
  stratum_breakdown: TrainingMatchupStratumSummary[]
}

export interface TrainingDeckMatchupsPayload {
  run_id: string
  candidate_deck_label: string
  range: TrainingEvidenceRange
  checkpoint_version: number | null
  controller: TrainingController
  candidate_seat: number | null
  opponent_set: OpponentSet
  available: boolean
  unavailable_reason: string | null
  overall: TrainingPosterior
  opponent_deck_count: number
  pilot_count: number
  seat_breakdown: TrainingMatchupSeatSummary[]
  stratum_breakdown: TrainingMatchupStratumSummary[]
  opponents: TrainingOpponentMatchupSummary[]
  matchups: TrainingMatchupDetail[]
}

export type EnvironmentWindow = 1 | 2 | 7 | 14
export type EnvironmentCoverageState =
  | 'verified_responder'
  | 'evidence_blind_spot'
  | 'unresolved'

export interface EnvironmentMetaDeck {
  deck_hash: string | null
  deck_digest: string
  deck_label: string | null
  display_name: string | null
  active_roster: boolean
  sides: number
  share: number
  early_share: number
  late_share: number
  share_delta: number
  new_in_late_half: boolean
  unique_pilots: number
  pilot_hhi: number
  effective_pilots: number
}

export interface EnvironmentRosterStanding {
  deck_label: string
  deck_hash: string
  deck_digest: string
  display_name: string | null
  family_id: string | null
  rank: number | null
  evidence_status: 'eligible' | 'sparse'
  environment_sides: number
  valid_games: number
  first_games: number
  second_games: number
  observed_score: number | null
  deploy_mean: number
  deploy_credible_low: number
  deploy_credible_high: number
  deploy_lcb: number
  matchup_cvar_mean: number
  probability_above_even: number
  observed_meta_mass: number
  prior_only_meta_mass: number
}

export interface EnvironmentCoverageRow {
  opponent_deck_hash: string | null
  opponent_deck_digest: string
  opponent_deck_label: string | null
  meta_share: number
  state: EnvironmentCoverageState
  best_candidate_deck_hash: string | null
  best_lcb: number | null
  eligible_candidates: number
}

export interface EnvironmentMatchupCell {
  candidate_deck_hash: string
  candidate_deck_digest: string
  opponent_deck_hash: string | null
  opponent_deck_digest: string
  opponent_deck_label: string | null
  games: number
  first_games: number
  second_games: number
  evidence_eligible: boolean
  posterior_mean: number
  credible_low: number
  credible_high: number
  lcb: number
}

export interface PublicEnvironmentPayload {
  schema_version: 1
  available: boolean
  run_id: string
  checkpoint_version: number | null
  window_days: EnvironmentWindow
  as_of_date: string | null
  required_dates: string[]
  missing_dates: string[]
  generated_at_utc: string | null
  snapshot_fingerprint: string | null
  source_scope: string
  unavailable_reason: string | null
  score_semantics: {
    submission_data_used: boolean
    primary?: string | null
    meta_weighting?: string | null
    draw_score?: number | null
    non_done?: string | null
    seat_weights?: Record<string, number>
  }
  quality: {
    episodes: number
    sides: number
    valid_episodes: number
    unresolved_episodes: number
    source_missing_episodes: number
    source_missing_bytes: number
    explicit_meta_mass: number
    unknown_tail_mass: number
    eligible_roster_decks: number
    roster_decks: number
  }
  overview: {
    exact_decks: number
    pilot_keys: number
    meta_hhi: number
    top_10_share: number
  }
  meta_decks: EnvironmentMetaDeck[]
  roster_standings: EnvironmentRosterStanding[]
  coverage: EnvironmentCoverageRow[]
  matrix: EnvironmentMatchupCell[]
}

export interface PublicEnvironmentMatrixPayload {
  run_id: string
  checkpoint_version: number | null
  window_days: EnvironmentWindow
  available: boolean
  snapshot_fingerprint: string | null
  cells: EnvironmentMatchupCell[]
}

export interface PublicEnvironmentMatchupsPayload extends PublicEnvironmentMatrixPayload {
  deck_hash: string
}

export interface TwoDeckRecommendationQuality {
  ready: boolean
  training_available: boolean
  public_environment_available: boolean
  active_candidates: number
  eligible_candidates: number
  candidate_pairs: number
  known_meta_mass: number
  unknown_meta_mass: number
  unexpanded_explicit_meta_mass: number
  rare_unknown_meta_mass: number
  warnings: string[]
}

export interface TwoDeckCandidate {
  deck_label: string
  deck_hash: string
  deck_digest: string
  display_name: string
  route_compatible: boolean
  training_games: number
  training_first_games: number
  training_second_games: number
  training_mean: number
  training_credible_low: number
  training_credible_high: number
  public_evidence_status: 'eligible' | 'sparse'
  public_games: number
  public_deploy_mean: number
  public_deploy_lcb: number
  proxy_rank: number
  proxy_mean: number
  proxy_credible_low: number
  proxy_credible_high: number
  proxy_lcb: number
  observed_meta_mass: number
  prior_only_meta_mass: number
}

export interface TwoDeckPair {
  rank: number
  deck_a_label: string
  deck_a_hash: string
  deck_a_digest: string
  deck_a_display_name: string
  deck_b_label: string
  deck_b_hash: string
  deck_b_digest: string
  deck_b_display_name: string
  expected_best_score: number
  credible_low: number
  credible_high: number
  best_score_lcb: number
  probability_at_least_one_above_even: number
  joint_downside_probability: number
  diversification_gain: number
  score_correlation: number | null
  expected_regret: number
  common_weak_meta_mass: number
  shared_observed_meta_mass: number
}

export interface TwoDeckRecommendationPayload {
  schema_version: 1
  available: boolean
  semantics: 'training_matchups_kaggle_daily_meta_portfolio_proxy_v1'
  run_id: string
  checkpoint_version: number | null
  checkpoint_pair_manifest_sha256: string | null
  training_range: TrainingEvidenceRange
  training_started_at_utc: string | null
  training_ended_at_utc: string | null
  public_window_days: EnvironmentWindow
  public_as_of_date: string | null
  public_snapshot_fingerprint: string | null
  objective: 'expected_max_meta_weighted_score_proxy'
  quality: TwoDeckRecommendationQuality
  recommendation: TwoDeckPair | null
  pairs: TwoDeckPair[]
  candidates: TwoDeckCandidate[]
  warnings: string[]
}

export interface WorkbenchAlert {
  code: string
  severity: 'info' | 'warning' | 'critical'
  title: string
  detail: string
}

export interface ProgressSummary {
  update_index: number | null
  target_updates: number | null
  optimizer_step_index: number | null
  decisions_seen: number | null
  target_decisions: number | null
  kept_decisions_per_second: number | null
}

export interface CurrentLearnerSnapshot {
  loss: number | null
  policy_loss: number | null
  value_loss: number | null
  belief_loss: number | null
  entropy: number | null
  approximate_kl: number | null
  clip_fraction: number | null
  gradient_norm: number | null
  learning_rate: number | null
  collection_seconds: number | null
  learner_seconds: number | null
  checkpoint_seconds: number | null
  cuda_peak_allocated_bytes: number | null
  cuda_peak_reserved_bytes: number | null
  fragments_stale: number | null
}

export interface TrainingGameStatistics {
  semantics: 'terminal_outcome_telemetry_sum_v1'
  total_games: number
  selected_run_games: number | null
  other_runs_games: number
  counted_run_count: number
  unreported_run_count: number
  invalid_run_count: number
  is_lower_bound: boolean
  count_stages: string[]
  updated_at_utc: string | null
}

export interface WorkbenchSummary {
  run: RunInfo
  data_age_seconds: number | null
  data_state: 'ready' | 'stale' | 'invalid'
  warnings: string[]
  checkpoint: CheckpointInfo | null
  training_games: TrainingGameStatistics
  progress: ProgressSummary
  latest_learner: CurrentLearnerSnapshot
  evidence: PosteriorStats
  alerts: WorkbenchAlert[]
}

export interface LearnerMetricRecord {
  format: 'checkpoint-learner-metric-v1'
  recorded_at_utc: string
  run_version: string
  update_index: number
  optimizer_step_index: number
  checkpoint_version: number
  pair_manifest_sha256: string
  policy_sha256: string
  learner_state_sha256: string
  decisions: number
  fragments_seen: number
  fragments_stale: number
  loss: number
  policy_loss: number
  value_loss: number
  belief_loss: number
  entropy: number
  ratio_mean: number
  approximate_kl: number
  clip_fraction: number
  gradient_norm: number
  learning_rate: number
  kept_decisions_per_second: number
  collection_seconds: number
  learner_seconds: number
  checkpoint_seconds: number
  total_seconds: number
  cuda_peak_allocated_bytes: number | null
  cuda_peak_reserved_bytes: number | null
  cuda_allocation_retries: number | null
  cuda_ooms: number | null
}

export interface LearnerSeriesPayload {
  run_id: string
  records: LearnerMetricRecord[]
  complete: boolean
  warning: string | null
}

export interface ComparisonReference {
  run_id: string
  checkpoint_version: number | null
}

export interface ComparisonRow {
  reference: ComparisonReference
  checkpoint: CheckpointInfo | null
  learner_metric: LearnerMetricRecord | null
  training_pool: PosteriorStats
  compatible_group: string | null
  warnings: string[]
}

export interface ComparisonPayload {
  rows: ComparisonRow[]
  all_compatible: boolean
}

export type JobState =
  | 'starting'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelling'
  | 'cancelled'
  | 'unknown'

export interface JobTemplateField {
  name: string
  label: string
  kind: string
  required: boolean
  options?: string[]
}

export interface JobTemplate {
  template_id:
    | 'bundle_evaluation'
    | 'package_validation'
    | 'config_dry_run'
    | 'public_environment_refresh'
  label: string
  description: string
  fields: JobTemplateField[]
}

export interface JobReceipt {
  schema_version: 1
  job_id: string
  template_id: JobTemplate['template_id']
  state: JobState
  created_at_utc: string
  updated_at_utc: string
  parameters: Record<string, unknown>
  argv: string[]
  cwd: string
  log_path: string
  worker_pid: number | null
  exit_code: number | null
  detail: string | null
}

export interface JobProgressPayload {
  schema_version: 1
  job_id: string
  template_id: JobTemplate['template_id']
  state: JobState
  phase: string
  message: string
  completed: number
  total: number
  percent: number
  current_date: string | null
  updated_at_utc: string
  detail: string | null
}

export interface DashboardSession {
  actions_enabled: boolean
  action_scope: 'disabled' | 'loopback'
  request_token: string | null
  refresh_seconds: number
}

export interface JobLogPayload {
  job_id: string
  text: string
  truncated: boolean
}

export type TaskKind =
  | 'bundle_strength'
  | 'release_h2h'
  | 'runtime_elo'
  | 'package_validation'
  | 'config_dry_run'

export type TaskMode = 'formal' | 'diagnostic' | 'utility'
export type TaskExecution = 'local' | 'distributed' | 'remote'
export type TaskState =
  | 'queued'
  | 'starting'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelling'
  | 'cancelled'
  | 'unknown'

export interface TaskWorkflow {
  kind: TaskKind
  label: string
  description: string
  modes: TaskMode[]
  result_views: string[]
}

export type TaskCatalogItemKind =
  | 'release_bundle'
  | 'checkpoint'
  | 'deck'
  | 'registered_opponent'
  | 'public_catalog'
  | 'side_observations'
  | 'evaluation_profile'
  | 'submission_profile'
  | 'training_profile'
  | 'runtime_template'

export interface TaskCatalogItem {
  artifact_id: string
  kind: TaskCatalogItemKind
  label: string
  detail: string | null
  fingerprint: string | null
  available: boolean
  unavailable_reason: string | null
  metadata: Record<string, unknown>
}

export interface TaskResourceSnapshot {
  training_active: boolean
  local_gpu_busy: boolean
  running_heavy_tasks: number
  running_light_tasks: number
  detail: string
}

export interface TaskCatalog {
  workflows: TaskWorkflow[]
  releases: TaskCatalogItem[]
  checkpoints: TaskCatalogItem[]
  decks: TaskCatalogItem[]
  registered_opponents: TaskCatalogItem[]
  public_catalogs: TaskCatalogItem[]
  side_observations: TaskCatalogItem[]
  evaluation_profiles: TaskCatalogItem[]
  submission_profiles: TaskCatalogItem[]
  training_profiles: TaskCatalogItem[]
  runtime_templates: TaskCatalogItem[]
  resources: TaskResourceSnapshot
}

export interface TaskParticipantRef {
  source: 'release_bundle' | 'registered_opponent' | 'checkpoint'
  artifact_id: string
  label?: string | null
  deck_id?: string | null
  public_catalog_id?: string | null
  runtime_template_id?: string | null
}

export type TaskCreateRequest =
  | {
      kind: 'bundle_strength'
      mode: 'formal' | 'diagnostic'
      label: string
      candidates: TaskParticipantRef[]
      opponents: TaskParticipantRef[]
      games_per_matchup: number
      execution: TaskExecution
      side_observations_id: string
      output_label: string
    }
  | {
      kind: 'release_h2h'
      mode: 'formal'
      label: string
      candidate_id: string
      opponent_id: string
      games: number
      execution: TaskExecution
      output_label: string
    }
  | {
      kind: 'runtime_elo'
      mode: 'diagnostic'
      label: string
      checkpoint_id: string
      deck_ids: string[]
      public_catalog_id: string
      runtime_template_id: string
      games_per_pair: number
      execution: 'local'
      output_label: string
    }
  | {
      kind: 'package_validation'
      mode: 'utility'
      label: string
      submission_profile_id: string
      output_label: string
    }
  | {
      kind: 'config_dry_run'
      mode: 'utility'
      label: string
      training_profile_id: string
    }

export interface TaskProgress {
  phase: string
  games_total: number | null
  games_committed: number | null
  games_finished: number | null
  percent: number | null
  rate_per_second: number | null
  eta_seconds: number | null
  quality_warnings: string[]
  detail: string | null
}

export interface TaskReceipt {
  schema_version: 2
  task_id: string
  kind: TaskKind
  mode: TaskMode
  label: string
  state: TaskState
  resource_class: 'light' | 'local_heavy' | 'local_cuda' | 'remote_heavy'
  queue_reason: string | null
  created_at_utc: string
  updated_at_utc: string
  started_at_utc: string | null
  finished_at_utc: string | null
  spec_fingerprint: string
  spec_path: string
  output_dir: string | null
  status_path: string | null
  log_path: string
  argv: string[]
  cwd: string
  attempt: number
  retry_of: string | null
  worker_pid: number | null
  worker_start_ticks: number | null
  exit_code: number | null
  detail: string | null
  progress: TaskProgress | null
}

export interface TaskListPayload {
  tasks: TaskReceipt[]
  total: number
  offset: number
  limit: number
}

export interface DeckSelectionQuality {
  ready: boolean
  expected_decks: number
  observed_decks: number
  expected_matchups: number
  observed_matchups: number
  expected_games_per_matchup: number
  expected_games: number
  observed_games: number
  seat_balanced: boolean
  agent_error_games: number
  truncated_games: number
  warnings: string[]
}

export interface DeckSelectionCell {
  candidate_id: string
  candidate_hash: string
  opponent_id: string
  opponent_hash: string
  candidate_label: string
  opponent_label: string
  games: number
  wins: number
  draws: number
  losses: number
  truncated: number
  agent_error_games: number
  seat_0_games: number
  seat_1_games: number
  score_rate: number | null
  posterior_mean: number | null
  credible_low: number | null
  credible_high: number | null
  evidence_state: 'ready' | 'incomplete' | 'missing'
}

export interface DeckSelectionStanding {
  rank: number
  deck_id: string
  deck_hash: string
  deck_label: string
  games: number
  wins: number
  draws: number
  losses: number
  truncated: number
  opponent_count: number
  equal_score_rate: number | null
  posterior_mean: number | null
  credible_low: number | null
  credible_high: number | null
  worst_opponent_id: string | null
  worst_opponent_label: string | null
  worst_matchup_score: number | null
  best_opponent_id: string | null
  best_opponent_label: string | null
  best_matchup_score: number | null
  evidence_state: 'ready' | 'incomplete' | 'missing'
}

export interface DeckSelectionPayload {
  task_id: string
  task_state: TaskState
  semantics: 'same_checkpoint_equal_opponent_deck_selection_v1'
  checkpoint_id: string
  checkpoint_fingerprint: string | null
  spec_fingerprint: string
  result_fingerprint: string | null
  quality: DeckSelectionQuality
  standings: DeckSelectionStanding[]
  cells: DeckSelectionCell[]
}

export interface TaskResultSummary {
  task_id: string
  kind: TaskKind
  state: TaskState
  semantics: string
  headline: string
  metrics: Record<string, unknown>
  quality_warnings: string[]
  tables: string[]
  report_artifact_id: string | null
}

export interface TaskTablePayload {
  task_id: string
  table: string
  columns: string[]
  rows: Record<string, unknown>[]
  offset: number
  limit: number
  returned: number
  has_more: boolean
}

export interface TaskArtifact {
  artifact_id: string
  label: string
  media_type: string
  size_bytes: number
  sha256: string
  previewable: boolean
  downloadable: boolean
}

export interface TaskArtifactList {
  task_id: string
  artifacts: TaskArtifact[]
}

export interface TaskArtifactContent {
  task_id: string
  artifact_id: string
  media_type: string
  text: string
  truncated: boolean
}

export interface TaskLogPayloadV2 {
  task_id: string
  text: string
  truncated: boolean
}

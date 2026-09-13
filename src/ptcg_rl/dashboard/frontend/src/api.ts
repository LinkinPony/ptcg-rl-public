import type {
  CheckpointInfo,
  ComparisonPayload,
  ComparisonReference,
  DashboardSession,
  DeckSelectionPayload,
  DeckEvidencePayload,
  DeckMatchupPayload,
  DeckSeries,
  JobLogPayload,
  JobProgressPayload,
  JobReceipt,
  JobTemplate,
  LearnerSeriesPayload,
  LeagueMatchup,
  LeagueStanding,
  LeagueSummary,
  LeagueWorker,
  MatchupRow,
  PerformanceTable,
  RunInfo,
  ScopeName,
  SeriesPoint,
  TaskArtifactContent,
  TaskArtifactList,
  TaskCatalog,
  TaskCreateRequest,
  TaskListPayload,
  TaskLogPayloadV2,
  TaskReceipt,
  TaskResultSummary,
  TaskTablePayload,
  TrainingController,
  TrainingDeckMatchupsPayload,
  TrainingDeckMatrixPayload,
  TrainingDeckSeriesPayload,
  TrainingDeckStrengthPayload,
  TrainingEvidenceRange,
  TrainingGameStatistics,
  TwoDeckRecommendationPayload,
  OpponentSet,
  AllocationPortfolio,
  AllocationRole,
  AllocationSort,
  OpponentAllocationMatchupPage,
  OpponentAllocationSummary,
  EnvironmentWindow,
  PublicEnvironmentMatchupsPayload,
  PublicEnvironmentMatrixPayload,
  PublicEnvironmentPayload,
  WindowName,
  WorkbenchSummary,
} from './types'

const inFlightGets = new Map<string, Promise<unknown>>()

function requestJson<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const method = (init?.method ?? 'GET').toUpperCase()
  if (method === 'GET') {
    const existing = inFlightGets.get(path)
    if (existing) return existing as Promise<T>
  }
  const pending = fetchJson<T>(path, init)
  if (method === 'GET') {
    inFlightGets.set(path, pending)
    const release = () => {
      if (inFlightGets.get(path) === pending) inFlightGets.delete(path)
    }
    void pending.then(release, release)
  }
  return pending
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  })
  if (!response.ok) {
    const message = await response.text()
    throw new Error(`${response.status}: ${message}`)
  }
  const contentType = response.headers.get('content-type')?.toLowerCase() ?? ''
  if (!contentType.includes('application/json')) {
    const preview = (await response.text()).slice(0, 120).replace(/\s+/g, ' ')
    throw new Error(
      `API ${path} returned ${contentType || 'an unknown content type'} instead of JSON. ` +
        `The dashboard backend may be stale or misconfigured. Response: ${preview}`,
    )
  }
  return (await response.json()) as T
}

const runPath = (run: string) => `/api/v2/runs/${encodeURIComponent(run)}`

export const api = {
  leagueSummary: () => requestJson<LeagueSummary>('/api/v3/league/summary'),
  leagueCheckpoints: () => requestJson<LeagueStanding[]>('/api/v3/league/checkpoints'),
  leagueDecks: () => requestJson<LeagueStanding[]>('/api/v3/league/decks'),
  leagueBundles: () => requestJson<LeagueStanding[]>('/api/v3/league/bundles'),
  leagueCandidates: () => requestJson<LeagueStanding[]>('/api/v3/league/candidates'),
  leagueWorkers: () => requestJson<LeagueWorker[]>('/api/v3/league/workers'),
  leagueMatchups: () => requestJson<LeagueMatchup[]>('/api/v3/league/matchups'),
  addLeagueCheckpoint: (pairManifestPath: string, label: string | null) =>
    requestJson<Record<string, string>>('/api/v3/league/assets/checkpoints', {
      method: 'POST',
      body: JSON.stringify({ pair_manifest_path: pairManifestPath, label }),
    }),
  addLeagueDeck: (deckPath: string, label: string | null) =>
    requestJson<Record<string, boolean>>('/api/v3/league/assets/decks', {
      method: 'POST',
      body: JSON.stringify({ deck_path: deckPath, label }),
    }),
  addLeagueRelease: (releaseManifestPath: string, alias: string) =>
    requestJson<Record<string, string>>('/api/v3/league/assets/releases', {
      method: 'POST',
      body: JSON.stringify({ release_manifest_path: releaseManifestPath, alias }),
    }),
  setLeagueControllerActive: (controllerId: string, active: boolean, reason: string) =>
    requestJson<Record<string, boolean>>(
      `/api/v3/league/controllers/${encodeURIComponent(controllerId)}/activation`,
      {
        method: 'POST',
        body: JSON.stringify({ active, reason }),
      },
    ),
  forceLeagueChallenge: (sideA: string, sideB: string) =>
    requestJson<Record<string, string>>('/api/v3/league/challenges', {
      method: 'POST',
      body: JSON.stringify({
        side_a_bundle_id: sideA,
        side_b_bundle_id: sideB,
      }),
    }),
  session: () => requestJson<DashboardSession>('/api/v2/session'),
  runs: () => requestJson<RunInfo[]>('/api/v2/runs'),
  summary: (run: string, window: WindowName) =>
    requestJson<WorkbenchSummary>(`${runPath(run)}/summary?window=${window}`),
  trainingGames: (run: string) =>
    requestJson<TrainingGameStatistics>(`${runPath(run)}/training-games`),
  checkpoints: (run: string) =>
    requestJson<CheckpointInfo[]>(`${runPath(run)}/checkpoints`),
  learnerSeries: (run: string) =>
    requestJson<LearnerSeriesPayload>(`${runPath(run)}/learner-series`),
  opponentAllocation: (run: string) =>
    requestJson<OpponentAllocationSummary>(`${runPath(run)}/opponent-allocation`),
  opponentAllocationMatchups: (
    run: string,
    options: {
      offset: number
      limit: number
      sort: AllocationSort
      candidateDeckDigest?: string | null
      artifactId?: string | null
      portfolio?: AllocationPortfolio | null
      role?: AllocationRole | null
      candidateSeat?: 0 | 1 | null
    },
  ) => {
    const params = new URLSearchParams({
      offset: `${options.offset}`,
      limit: `${options.limit}`,
      sort: options.sort,
    })
    if (options.candidateDeckDigest) {
      params.set('candidate_deck_digest', options.candidateDeckDigest)
    }
    if (options.artifactId) params.set('artifact_id', options.artifactId)
    if (options.portfolio) params.set('portfolio', options.portfolio)
    if (options.role) params.set('role', options.role)
    if (options.candidateSeat != null) {
      params.set('candidate_seat', `${options.candidateSeat}`)
    }
    return requestJson<OpponentAllocationMatchupPage>(
      `${runPath(run)}/opponent-allocation/matchups?${params}`,
    )
  },
  deckEvidence: (run: string, window: WindowName) =>
    requestJson<DeckEvidencePayload>(
      `${runPath(run)}/deck-evidence?window=${window}`,
    ),
  deckMatchups: (run: string, deck: string, window: WindowName) =>
    requestJson<DeckMatchupPayload>(
      `${runPath(run)}/decks/${encodeURIComponent(deck)}/matchups?window=${window}`,
    ),
  trainingDeckStrength: (run: string, checkpointVersion: number | null) => {
    const params = new URLSearchParams()
    if (checkpointVersion != null) {
      params.set('checkpoint_version', `${checkpointVersion}`)
    }
    return requestJson<TrainingDeckStrengthPayload>(
      `${runPath(run)}/training-deck-strength?${params}`,
    )
  },
  publicEnvironment: (
    run: string,
    windowDays: EnvironmentWindow,
    checkpointVersion: number | null,
  ) => {
    const params = environmentParams(windowDays, checkpointVersion)
    return requestJson<PublicEnvironmentPayload>(
      `${runPath(run)}/public-environment?${params}`,
    )
  },
  publicEnvironmentMatrix: (
    run: string,
    windowDays: EnvironmentWindow,
    checkpointVersion: number | null,
  ) => {
    const params = environmentParams(windowDays, checkpointVersion)
    return requestJson<PublicEnvironmentMatrixPayload>(
      `${runPath(run)}/public-environment/matrix?${params}`,
    )
  },
  publicEnvironmentMatchups: (
    run: string,
    deckHash: string,
    windowDays: EnvironmentWindow,
    checkpointVersion: number | null,
  ) => {
    const params = environmentParams(windowDays, checkpointVersion)
    return requestJson<PublicEnvironmentMatchupsPayload>(
      `${runPath(run)}/public-environment/decks/${encodeURIComponent(deckHash)}/matchups?${params}`,
    )
  },
  twoDeckRecommendation: (
    run: string,
    trainingRange: TrainingEvidenceRange,
    windowDays: EnvironmentWindow,
    checkpointVersion: number,
  ) => {
    const params = new URLSearchParams({
      training_range: trainingRange,
      window_days: `${windowDays}`,
      checkpoint_version: `${checkpointVersion}`,
    })
    return requestJson<TwoDeckRecommendationPayload>(
      `${runPath(run)}/two-deck-recommendation?${params}`,
    )
  },
  refreshPublicEnvironment: (
    run: string,
    checkpointVersion: number | null,
  ) => {
    const params = new URLSearchParams()
    if (checkpointVersion != null) {
      params.set('checkpoint_version', `${checkpointVersion}`)
    }
    return requestJson<JobReceipt>(
      `${runPath(run)}/public-environment/refresh?${params}`,
      { method: 'POST' },
    )
  },
  trainingDeckSeries: (
    run: string,
    range: TrainingEvidenceRange,
    checkpointVersion: number | null,
    controller: TrainingController,
  ) => {
    const params = trainingStrengthParams(range, checkpointVersion, controller)
    return requestJson<TrainingDeckSeriesPayload>(
      `${runPath(run)}/training-deck-strength/series?${params}`,
    )
  },
  trainingDeckMatrix: (
    run: string,
    range: TrainingEvidenceRange,
    checkpointVersion: number | null,
    controller: TrainingController,
    seat: number | null,
    opponentSet: OpponentSet,
  ) => {
    const params = trainingStrengthParams(range, checkpointVersion, controller)
    if (seat != null) params.set('candidate_seat', `${seat}`)
    params.set('opponent_set', opponentSet)
    return requestJson<TrainingDeckMatrixPayload>(
      `${runPath(run)}/training-deck-strength/matrix?${params}`,
    )
  },
  trainingDeckMatchups: (
    run: string,
    deck: string,
    range: TrainingEvidenceRange,
    checkpointVersion: number | null,
    controller: TrainingController,
    seat: number | null,
    opponentSet: OpponentSet,
  ) => {
    const params = trainingStrengthParams(range, checkpointVersion, controller)
    if (seat != null) params.set('candidate_seat', `${seat}`)
    params.set('opponent_set', opponentSet)
    return requestJson<TrainingDeckMatchupsPayload>(
      `${runPath(run)}/training-deck-strength/decks/${encodeURIComponent(deck)}/matchups?${params}`,
    )
  },
  compare: (references: ComparisonReference[]) =>
    requestJson<ComparisonPayload>('/api/v2/comparison', {
      method: 'POST',
      body: JSON.stringify({ references }),
    }),
  jobTemplates: () => requestJson<JobTemplate[]>('/api/v2/jobs/templates'),
  jobs: () => requestJson<JobReceipt[]>('/api/v2/jobs'),
  job: (jobId: string) =>
    requestJson<JobReceipt>(`/api/v2/jobs/${jobId}`),
  jobProgress: (jobId: string) =>
    requestJson<JobProgressPayload>(`/api/v2/jobs/${jobId}/progress`),
  jobLog: (jobId: string) =>
    requestJson<JobLogPayload>(`/api/v2/jobs/${jobId}/log`),
  startJob: (
    templateId: JobTemplate['template_id'],
    parameters: Record<string, unknown>,
    token: string,
  ) =>
    requestJson<JobReceipt>('/api/v2/jobs', {
      method: 'POST',
      headers: { 'X-Dashboard-Token': token },
      body: JSON.stringify({ template_id: templateId, parameters }),
    }),
  cancelJob: (jobId: string, token: string) =>
    requestJson<JobReceipt>(`/api/v2/jobs/${jobId}/cancel`, {
      method: 'POST',
      headers: { 'X-Dashboard-Token': token },
    }),
  taskCatalog: () => requestJson<TaskCatalog>('/api/v2/tasks/catalog'),
  tasks: (offset = 0, limit = 50) =>
    requestJson<TaskListPayload>(
      `/api/v2/tasks?offset=${offset}&limit=${limit}`,
    ),
  task: (taskId: string) =>
    requestJson<TaskReceipt>(`/api/v2/tasks/${taskId}`),
  runtimeLadderResult: (taskId: string) =>
    requestJson<DeckSelectionPayload>(
      `/api/v2/tasks/${taskId}/runtime-ladder`,
    ),
  taskResult: (taskId: string) =>
    requestJson<TaskResultSummary>(`/api/v2/tasks/${taskId}/result`),
  taskTable: (taskId: string, table: string, offset = 0, limit = 50) =>
    requestJson<TaskTablePayload>(
      `/api/v2/tasks/${taskId}/tables/${encodeURIComponent(table)}?offset=${offset}&limit=${limit}`,
    ),
  taskArtifacts: (taskId: string) =>
    requestJson<TaskArtifactList>(`/api/v2/tasks/${taskId}/artifacts`),
  taskArtifactContent: (taskId: string, artifactId: string) =>
    requestJson<TaskArtifactContent>(
      `/api/v2/tasks/${taskId}/artifacts/${artifactId}/content`,
    ),
  taskLog: (taskId: string) =>
    requestJson<TaskLogPayloadV2>(`/api/v2/tasks/${taskId}/log`),
  startTask: (payload: TaskCreateRequest, token: string) =>
    requestJson<TaskReceipt>('/api/v2/tasks', {
      method: 'POST',
      headers: { 'X-Dashboard-Token': token },
      body: JSON.stringify(payload),
    }),
  cancelTask: (taskId: string, token: string) =>
    requestJson<TaskReceipt>(`/api/v2/tasks/${taskId}/cancel`, {
      method: 'POST',
      headers: { 'X-Dashboard-Token': token },
    }),
  retryTask: (taskId: string, token: string) =>
    requestJson<TaskReceipt>(`/api/v2/tasks/${taskId}/retry`, {
      method: 'POST',
      headers: { 'X-Dashboard-Token': token },
    }),

  // v1 compatibility calls retained for legacy components and CLI parity.
  table: (run: string, window: WindowName, scope: ScopeName) =>
    requestJson<PerformanceTable>(
      `/api/v1/runs/${encodeURIComponent(run)}/decks?window=${window}&scope=${scope}`,
    ),
  series: (run: string, deck: string, slice: string, rolling: number) => {
    const params = new URLSearchParams({
      opponent_slice: slice,
      rolling_minutes: `${rolling}`,
    })
    if (deck) params.set('deck_label', deck)
    return requestJson<SeriesPoint[]>(
      `/api/v1/runs/${encodeURIComponent(run)}/series?${params}`,
    )
  },
  deckSeries: (run: string, slice: string, rolling: number, scope: ScopeName) => {
    const params = new URLSearchParams({
      opponent_slice: slice,
      rolling_minutes: `${rolling}`,
      scope,
    })
    return requestJson<DeckSeries[]>(
      `/api/v1/runs/${encodeURIComponent(run)}/deck-series?${params}`,
    )
  },
  matchups: (run: string, window: WindowName, kind: string, seat: string) => {
    const params = new URLSearchParams({ window })
    if (kind) params.set('opponent_kind', kind)
    if (seat) params.set('candidate_seat', seat)
    return requestJson<MatchupRow[]>(
      `/api/v1/runs/${encodeURIComponent(run)}/matchups?${params}`,
    )
  },
}

function trainingStrengthParams(
  range: TrainingEvidenceRange,
  checkpointVersion: number | null,
  controller: TrainingController,
) {
  const params = new URLSearchParams({ range, controller })
  if (checkpointVersion != null) {
    params.set('checkpoint_version', `${checkpointVersion}`)
  }
  return params
}

function environmentParams(
  windowDays: EnvironmentWindow,
  checkpointVersion: number | null,
) {
  const params = new URLSearchParams({ window_days: `${windowDays}` })
  if (checkpointVersion != null) {
    params.set('checkpoint_version', `${checkpointVersion}`)
  }
  return params
}

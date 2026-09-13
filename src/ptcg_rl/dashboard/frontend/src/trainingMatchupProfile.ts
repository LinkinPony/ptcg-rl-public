import type {
  TrainingOpponentMatchupSummary,
  TrainingPosterior,
} from './types'

export type MatchupConclusion = 'disadvantage' | 'uncertain' | 'advantage'
export type MatchupSort = 'weakest' | 'strongest' | 'games' | 'uncertainty'

const stratumLabels: Record<string, string> = {
  self_play: 'Self-play',
  sentinel: 'Sentinel',
  adaptive_history: 'Adaptive history',
  scripted: 'Scripted',
  legacy_unknown: 'Legacy unknown',
}

export const matchupConclusion = (
  posterior: TrainingPosterior,
): MatchupConclusion => {
  if (
    posterior.evidence_state === 'ready'
    && posterior.credible_low != null
    && posterior.credible_low > 0.5
  ) {
    return 'advantage'
  }
  if (
    posterior.evidence_state === 'ready'
    && posterior.credible_high != null
    && posterior.credible_high < 0.5
  ) {
    return 'disadvantage'
  }
  return 'uncertain'
}

export const credibleWidth = (posterior: TrainingPosterior) => {
  if (posterior.credible_low == null || posterior.credible_high == null) {
    return -1
  }
  return posterior.credible_high - posterior.credible_low
}

export const sortedMatchups = (
  rows: TrainingOpponentMatchupSummary[],
  sort: MatchupSort,
) => [...rows].sort((left, right) => {
  if (sort === 'games') {
    return right.posterior.observed.games - left.posterior.observed.games
      || displayOrder(left, right)
  }
  if (sort === 'uncertainty') {
    return credibleWidth(right.posterior) - credibleWidth(left.posterior)
      || displayOrder(left, right)
  }
  const leftMean = left.posterior.posterior_mean
  const rightMean = right.posterior.posterior_mean
  if (leftMean == null && rightMean == null) return displayOrder(left, right)
  if (leftMean == null) return 1
  if (rightMean == null) return -1
  const scoreOrder = sort === 'strongest'
    ? rightMean - leftMean
    : leftMean - rightMean
  return scoreOrder || displayOrder(left, right)
})

export const intervalPosition = (value: number | null) => (
  value == null ? 0 : Math.min(100, Math.max(0, value * 100))
)

export const stratumLabel = (value: string) => stratumLabels[value] ?? value

function displayOrder(
  left: TrainingOpponentMatchupSummary,
  right: TrainingOpponentMatchupSummary,
) {
  return left.opponent_display_name.localeCompare(right.opponent_display_name)
}

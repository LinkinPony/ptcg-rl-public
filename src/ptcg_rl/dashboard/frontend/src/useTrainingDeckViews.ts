import { onMounted, ref, watch, type Ref } from 'vue'
import { api } from './api'
import type {
  CheckpointInfo,
  OpponentSet,
  TrainingController,
  TrainingDeckMatchupsPayload,
  TrainingDeckMatrixPayload,
  TrainingDeckSeriesPayload,
  TrainingEvidenceRange,
} from './types'

interface TrainingDeckViewInputs {
  run: Readonly<Ref<string>>
  checkpoint: Readonly<Ref<CheckpointInfo | null>>
  selectedDeckLabel: Ref<string | null>
  refreshVersion?: Readonly<Ref<number | undefined>>
}

const ranges: TrainingEvidenceRange[] = [
  'checkpoint',
  'recent_15m',
  'recent_60m',
  'cumulative',
]
const controllers: TrainingController[] = [
  'all',
  'self_play',
  'sentinel',
  'adaptive_history',
  'scripted',
]

export function useTrainingDeckViews(inputs: TrainingDeckViewInputs) {
  const controllerRange = ref<TrainingEvidenceRange>('checkpoint')
  const trendRange = ref<TrainingEvidenceRange>('recent_60m')
  const trendController = ref<TrainingController>('all')
  const matrixRange = ref<TrainingEvidenceRange>('cumulative')
  const matrixController = ref<TrainingController>('all')
  const matrixSeat = ref('all')
  const matrixOpponentSet = ref<OpponentSet>('active')
  const detailRange = ref<TrainingEvidenceRange>('checkpoint')
  const detailController = ref<TrainingController>('all')
  const detailSeat = ref('all')
  const detailOpponentSet = ref<OpponentSet>('all')

  const trend = ref<TrainingDeckSeriesPayload | null>(null)
  const matrix = ref<TrainingDeckMatrixPayload | null>(null)
  const detail = ref<TrainingDeckMatchupsPayload | null>(null)
  const trendLoading = ref(false)
  const matrixLoading = ref(false)
  const detailLoading = ref(false)
  const error = ref('')
  let trendGeneration = 0
  let matrixGeneration = 0
  let detailGeneration = 0

  onMounted(() => {
    restoreViewState()
    void loadTrend()
    void loadMatrix()
  })
  watch(inputs.run, () => {
    error.value = ''
    restoreViewState()
    void loadTrend()
    void loadMatrix()
  })
  watch(() => inputs.checkpoint.value?.pair_manifest_sha256, () => {
    void loadTrend()
    void loadMatrix()
    void loadDetail()
  })
  watch([trendRange, trendController], () => {
    persistViewState()
    void loadTrend()
  })
  if (inputs.refreshVersion != null) {
    watch(inputs.refreshVersion, () => {
      void loadTrend()
    })
  }
  watch(controllerRange, persistViewState)
  watch(
    [matrixRange, matrixController, matrixSeat, matrixOpponentSet],
    () => {
      persistViewState()
      void loadMatrix()
    },
  )
  watch(
    [
      inputs.selectedDeckLabel,
      detailRange,
      detailController,
      detailSeat,
      detailOpponentSet,
    ],
    () => {
      persistViewState()
      void loadDetail()
    },
  )

  function restoreViewState() {
    try {
      const stored = JSON.parse(
        window.localStorage.getItem(
          `ptcg-training-deck-views:${inputs.run.value}`,
        ) ?? '{}',
      ) as Record<string, string>
      controllerRange.value = validRange(stored.controllerRange, 'checkpoint')
      trendRange.value = validRange(stored.trendRange, 'recent_60m')
      trendController.value = validController(stored.trendController)
      matrixRange.value = validRange(stored.matrixRange, 'cumulative')
      matrixController.value = validController(stored.matrixController)
      matrixSeat.value = validSeat(stored.matrixSeat)
      matrixOpponentSet.value = stored.matrixOpponentSet === 'all' ? 'all' : 'active'
      detailRange.value = validRange(stored.detailRange, 'checkpoint')
      detailController.value = validController(stored.detailController)
      detailSeat.value = validSeat(stored.detailSeat)
      detailOpponentSet.value = stored.detailOpponentSet === 'active' ? 'active' : 'all'
    } catch {
      // Defaults remain authoritative when local state is invalid.
    }
  }

  function persistViewState() {
    window.localStorage.setItem(
      `ptcg-training-deck-views:${inputs.run.value}`,
      JSON.stringify({
        controllerRange: controllerRange.value,
        trendRange: trendRange.value,
        trendController: trendController.value,
        matrixRange: matrixRange.value,
        matrixController: matrixController.value,
        matrixSeat: matrixSeat.value,
        matrixOpponentSet: matrixOpponentSet.value,
        detailRange: detailRange.value,
        detailController: detailController.value,
        detailSeat: detailSeat.value,
        detailOpponentSet: detailOpponentSet.value,
      }),
    )
  }

  async function loadTrend() {
    if (!inputs.run.value) return
    const generation = ++trendGeneration
    trendLoading.value = true
    try {
      const next = await api.trainingDeckSeries(
        inputs.run.value,
        trendRange.value,
        inputs.checkpoint.value?.version ?? null,
        trendController.value,
      )
      if (generation === trendGeneration) trend.value = next
    } catch (reason) {
      if (generation === trendGeneration) error.value = message(reason)
    } finally {
      if (generation === trendGeneration) trendLoading.value = false
    }
  }

  async function loadMatrix() {
    if (!inputs.run.value) return
    const generation = ++matrixGeneration
    matrixLoading.value = true
    try {
      const next = await api.trainingDeckMatrix(
        inputs.run.value,
        matrixRange.value,
        inputs.checkpoint.value?.version ?? null,
        matrixController.value,
        seatNumber(matrixSeat.value),
        matrixOpponentSet.value,
      )
      if (generation === matrixGeneration) matrix.value = next
    } catch (reason) {
      if (generation === matrixGeneration) error.value = message(reason)
    } finally {
      if (generation === matrixGeneration) matrixLoading.value = false
    }
  }

  async function loadDetail() {
    const deckLabel = inputs.selectedDeckLabel.value
    if (!inputs.run.value || !deckLabel) {
      detail.value = null
      return
    }
    const generation = ++detailGeneration
    detailLoading.value = true
    try {
      const next = await api.trainingDeckMatchups(
        inputs.run.value,
        deckLabel,
        detailRange.value,
        inputs.checkpoint.value?.version ?? null,
        detailController.value,
        seatNumber(detailSeat.value),
        detailOpponentSet.value,
      )
      if (generation === detailGeneration) detail.value = next
    } catch (reason) {
      if (generation === detailGeneration) error.value = message(reason)
    } finally {
      if (generation === detailGeneration) detailLoading.value = false
    }
  }

  return {
    controllerRange,
    trendRange,
    trendController,
    matrixRange,
    matrixController,
    matrixSeat,
    matrixOpponentSet,
    detailRange,
    detailController,
    detailSeat,
    detailOpponentSet,
    trend,
    matrix,
    detail,
    trendLoading,
    matrixLoading,
    detailLoading,
    error,
    refreshMatrix: loadMatrix,
  }
}

function validRange(
  value: string | undefined,
  fallback: TrainingEvidenceRange,
): TrainingEvidenceRange {
  return ranges.includes(value as TrainingEvidenceRange)
    ? value as TrainingEvidenceRange
    : fallback
}

function validController(value: string | undefined): TrainingController {
  return controllers.includes(value as TrainingController)
    ? value as TrainingController
    : 'all'
}

function validSeat(value: string | undefined) {
  return ['all', '0', '1'].includes(value ?? '') ? value as string : 'all'
}

function seatNumber(value: string) {
  return value === 'all' ? null : Number(value)
}

function message(reason: unknown) {
  return reason instanceof Error ? reason.message : String(reason)
}

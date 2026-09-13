<script setup lang="ts">
import { computed, onMounted, ref, toRef, watch } from 'vue'
import {
  NAlert,
  NButton,
  NCard,
  NCheckbox,
  NFlex,
  NInput,
  NSelect,
  NTag,
  NTabPane,
  NTabs,
} from 'naive-ui'
import DeckTable from '../components/DeckTable.vue'
import MatchupHeatmap from '../components/MatchupHeatmap.vue'
import PublicEnvironmentPanel from '../components/PublicEnvironmentPanel.vue'
import TrainingControllerPanel from '../components/TrainingControllerPanel.vue'
import TrainingDecisionRail from '../components/TrainingDecisionRail.vue'
import TrainingMatchupDrawer from '../components/TrainingMatchupDrawer.vue'
import TwoDeckRecommendationPanel from '../components/TwoDeckRecommendationPanel.vue'
import TrendChart from '../components/TrendChart.vue'
import { exportTrainingDeckDecision } from '../deckSelectionExport'
import { useTrainingDeckViews } from '../useTrainingDeckViews'
import type {
  CheckpointInfo,
  TrainingController,
  TrainingDeckStanding,
  TrainingDeckStrengthPayload,
  TrainingEvidenceRange,
} from '../types'

const props = defineProps<{
  run: string
  checkpoint: CheckpointInfo | null
  evidence: TrainingDeckStrengthPayload | null
  refreshVersion?: number
}>()

const sourceTab = ref<'training' | 'environment' | 'portfolio'>('training')

const allRanges: TrainingEvidenceRange[] = [
  'checkpoint',
  'recent_15m',
  'recent_60m',
  'cumulative',
]
const rangeOptions: Array<{
  label: string
  value: TrainingEvidenceRange
}> = [
  { label: '所选 Checkpoint', value: 'checkpoint' },
  { label: '近 15 分钟', value: 'recent_15m' },
  { label: '近 60 分钟', value: 'recent_60m' },
  { label: 'Run 累计', value: 'cumulative' },
]
const controllerOptions: Array<{
  label: string
  value: TrainingController
}> = [
  { label: '全部对手分层', value: 'all' },
  { label: 'Self-play', value: 'self_play' },
  { label: 'Sentinel', value: 'sentinel' },
  { label: 'Adaptive history', value: 'adaptive_history' },
  { label: 'Scripted', value: 'scripted' },
]
const seatOptions = [
  { label: '全部先后手', value: 'all' },
  { label: 'Seat 0', value: '0' },
  { label: 'Seat 1', value: '1' },
]

const selectedRanges = ref<TrainingEvidenceRange[]>([...allRanges])
const sortRange = ref<TrainingEvidenceRange>('checkpoint')
const sortDirection = ref<'asc' | 'desc'>('desc')
const finalDeckLabel = ref<string | null>(null)
const selectedDeckLabel = ref<string | null>(null)
const familyFilter = ref<string | null>(null)
const searchQuery = ref('')

const {
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
  refreshMatrix,
} = useTrainingDeckViews({
  run: toRef(props, 'run'),
  checkpoint: toRef(props, 'checkpoint'),
  selectedDeckLabel,
  refreshVersion: toRef(props, 'refreshVersion'),
})

const finalStandings = computed(() => {
  const output: Partial<Record<TrainingEvidenceRange, TrainingDeckStanding>> = {}
  for (const range of selectedRanges.value) {
    const row = props.evidence?.ranges[range].standings.find(
      (item) => item.deck_label === finalDeckLabel.value,
    )
    if (row) output[range] = row
  }
  return output
})
const finalStanding = computed(() =>
  finalStandings.value[sortRange.value]
  ?? Object.values(finalStandings.value)[0]
  ?? null,
)
const focusedStandings = computed(() => selectedDeckLabel.value == null
  ? {}
  : finalStandingsFor(selectedDeckLabel.value))
const focusedStanding = computed(() =>
  focusedStandings.value[sortRange.value]
  ?? Object.values(focusedStandings.value)[0]
  ?? null,
)
const familyOptions = computed(() => (props.evidence?.families ?? []).map(
  (family) => ({
    label: `${family.display_name} · ${family.deck_labels.length}`,
    value: family.family_id,
  }),
))
const selectedDeckDisplayName = computed(() =>
  props.evidence?.ranges.cumulative.standings.find(
    (item) => item.deck_label === selectedDeckLabel.value,
  )?.display_name ?? selectedDeckLabel.value ?? '',
)
const selectedDeckHash = computed(() =>
  props.evidence?.ranges.cumulative.standings.find(
    (item) => item.deck_label === selectedDeckLabel.value,
  )?.deck_hash ?? null,
)
const checkpointEvidenceUnavailable = computed(
  () => props.evidence != null
    && !props.evidence.ranges.checkpoint.metadata.available,
)

onMounted(() => {
  restoreRanges()
  restoreDecision()
})

watch(() => props.run, () => {
  sortRange.value = 'checkpoint'
  sortDirection.value = 'desc'
  selectedDeckLabel.value = null
  familyFilter.value = null
  searchQuery.value = ''
  restoreRanges()
  restoreDecision()
})
watch(() => props.checkpoint?.pair_manifest_sha256, () => {
  restoreDecision()
})
watch(() => props.evidence?.checkpoint_pair_manifest_sha256, () => {
  const families = new Set(props.evidence?.families.map((item) => item.family_id))
  if (familyFilter.value != null && !families.has(familyFilter.value)) {
    familyFilter.value = null
  }
  restoreDecision()
})
watch(selectedRanges, () => {
  if (!selectedRanges.value.length) {
    selectedRanges.value = [...allRanges]
  }
  persistRangeState()
}, { deep: true })

function restoreRanges() {
  try {
    const stored = JSON.parse(
      window.localStorage.getItem(`ptcg-training-deck-ranges:${props.run}`) ?? '[]',
    ) as string[]
    const valid = stored.filter(
      (item): item is TrainingEvidenceRange =>
        allRanges.includes(item as TrainingEvidenceRange),
    )
    selectedRanges.value = valid.length ? valid : [...allRanges]
  } catch {
    selectedRanges.value = [...allRanges]
  }
}

function restoreDecision() {
  finalDeckLabel.value = null
  const key = decisionStorageKey()
  if (!key) return
  try {
    const stored = JSON.parse(window.localStorage.getItem(key) ?? '{}') as {
      finalDeckLabel?: string | null
    }
    const valid = new Set(props.evidence?.active_deck_labels ?? [])
    finalDeckLabel.value = stored.finalDeckLabel != null
      && valid.has(stored.finalDeckLabel)
      ? stored.finalDeckLabel
      : null
    if (finalDeckLabel.value != null && selectedDeckLabel.value == null) {
      selectedDeckLabel.value = finalDeckLabel.value
    }
  } catch {
    finalDeckLabel.value = null
  }
}

function persistRangeState() {
  window.localStorage.setItem(
    `ptcg-training-deck-ranges:${props.run}`,
    JSON.stringify(selectedRanges.value),
  )
}

function persistDecision() {
  const key = decisionStorageKey()
  if (!key) return
  window.localStorage.setItem(key, JSON.stringify({
    finalDeckLabel: finalDeckLabel.value,
  }))
}

function decisionStorageKey() {
  const artifact = props.checkpoint?.pair_manifest_sha256
  return artifact ? `ptcg-training-deck-decision:${props.run}:${artifact}` : null
}

function chooseFinal(deckLabel: string) {
  const standing = Object.values(finalStandingsFor(deckLabel))[0]
  if (standing?.route_compatible === false) return
  selectedDeckLabel.value = deckLabel
  finalDeckLabel.value = deckLabel
  persistDecision()
}

function finalStandingsFor(deckLabel: string) {
  const output: Partial<Record<TrainingEvidenceRange, TrainingDeckStanding>> = {}
  for (const range of selectedRanges.value) {
    const row = props.evidence?.ranges[range].standings.find(
      (item) => item.deck_label === deckLabel,
    )
    if (row) output[range] = row
  }
  return output
}

function sortBy(range: TrainingEvidenceRange) {
  if (sortRange.value === range) {
    sortDirection.value = sortDirection.value === 'desc' ? 'asc' : 'desc'
  } else {
    sortRange.value = range
    sortDirection.value = 'desc'
  }
}

function selectDeck(deckLabel: string) {
  selectedDeckLabel.value = deckLabel
}

function exportDecision(format: 'json' | 'markdown') {
  if (!props.evidence || !props.checkpoint || !finalDeckLabel.value) return
  exportTrainingDeckDecision({
    format,
    runId: props.run,
    checkpoint: props.checkpoint,
    evidence: props.evidence,
    selectedRanges: selectedRanges.value,
    finalDeckLabel: finalDeckLabel.value,
  })
}

</script>

<template>
  <n-tabs v-model:value="sourceTab" type="segment" animated>
    <n-tab-pane name="training" tab="训练观测">
      <section class="page-stack">
    <div class="page-heading">
      <div>
        <span class="page-eyebrow">OBSERVED TRAINING EVIDENCE</span>
        <h1>卡组强度与选择</h1>
        <p>直接读取已有训练统计；打开或刷新本页不会创建任务，也不会启动专项评测。</p>
      </div>
      <n-flex>
        <n-tag round type="info">{{ evidence?.active_deck_labels.length ?? 0 }} EXACT DECKS</n-tag>
        <n-tag round type="info">{{ evidence?.families.length ?? 0 }} FAMILIES</n-tag>
        <n-tag round :type="checkpoint ? 'success' : 'warning'">
          {{ checkpoint ? `v${checkpoint.version}` : '未选择 Checkpoint' }}
        </n-tag>
      </n-flex>
    </div>

    <n-alert v-if="error" type="error" closable @close="error = ''">{{ error }}</n-alert>
    <n-alert v-if="evidence" type="info">{{ evidence.source_warning }}</n-alert>
    <n-alert v-if="checkpointEvidenceUnavailable" type="warning">
      {{ evidence?.ranges.checkpoint.metadata.unavailable_reason }}。这只影响 checkpoint 专属列；
      仍可参考其他训练范围作出选择。
    </n-alert>

    <div v-if="evidence" class="deck-decision-layout">
      <div class="deck-evidence-main">
        <n-card class="work-card" :bordered="false">
          <template #header>
            <div class="section-title">
              <span>训练证据排名</span>
              <small>先按 family 浏览，再比较各 exact deck；点击行查看证据</small>
            </div>
          </template>
          <div class="deck-browser-toolbar">
            <n-select
              v-model:value="familyFilter"
              :options="familyOptions"
              clearable
              filterable
              placeholder="全部 family"
              class="deck-family-select"
            />
            <n-input
              v-model:value="searchQuery"
              clearable
              placeholder="搜索卡组、旧编号或 family"
              class="deck-search-input"
            />
            <span>
              {{ familyFilter ? '仅显示所选 family' : `共 ${evidence.families.length} 个 family` }}
            </span>
          </div>
          <div class="range-checkboxes">
            <n-checkbox
              v-for="option in rangeOptions"
              :key="option.value"
              :checked="selectedRanges.includes(option.value as TrainingEvidenceRange)"
              @update:checked="(checked) => {
                selectedRanges = checked
                  ? [...selectedRanges, option.value as TrainingEvidenceRange]
                  : selectedRanges.filter((item) => item !== option.value)
              }"
            >
              {{ option.label }}
            </n-checkbox>
          </div>
          <deck-table
            :evidence="evidence"
            :ranges="selectedRanges"
            :sort-range="sortRange"
            :sort-direction="sortDirection"
            :selected-deck-label="selectedDeckLabel"
            :family-filter="familyFilter"
            :search-query="searchQuery"
            @sort="sortBy"
            @select="selectDeck"
          />
        </n-card>

        <training-controller-panel
          v-model:range="controllerRange"
          :evidence="evidence"
          :range-options="rangeOptions"
        />

        <n-card class="work-card" :bordered="false">
          <template #header>
            <div class="section-title">
              <span>多卡组趋势</span>
              <small>每个点始终携带该完整训练窗口的实际局数</small>
            </div>
          </template>
          <template #header-extra>
            <n-flex>
              <n-select v-model:value="trendRange" :options="rangeOptions" class="deck-filter-select" />
              <n-select v-model:value="trendController" :options="controllerOptions" class="deck-filter-select" />
            </n-flex>
          </template>
          <n-alert v-if="trend && !trend.available" type="warning">
            {{ trend.unavailable_reason }}
          </n-alert>
          <trend-chart :series="trend?.series ?? []" :class="{ loading: trendLoading }" />
        </n-card>

        <n-card class="work-card" :bordered="false">
          <template #header>
            <div class="section-title">
              <span>精确 Matchup 热力图</span>
              <small>矩阵按 opponent deck 聚合；pilot identity 在卡组下钻中保留</small>
            </div>
          </template>
          <template #header-extra>
            <n-button
              quaternary
              size="small"
              class="matrix-refresh-button"
              :loading="matrixLoading"
              @click="refreshMatrix"
            >
              刷新热力图
            </n-button>
          </template>
          <div class="deck-filter-row">
            <n-select v-model:value="matrixRange" :options="rangeOptions" class="deck-filter-select" />
            <n-select v-model:value="matrixController" :options="controllerOptions" class="deck-filter-select" />
            <n-select v-model:value="matrixSeat" :options="seatOptions" class="deck-filter-select" />
            <n-select
              v-model:value="matrixOpponentSet"
              :options="[
                { label: '当前训练 Roster', value: 'active' },
                { label: '全部训练对手', value: 'all' },
              ]"
              class="deck-filter-select"
            />
          </div>
          <n-alert v-if="matrix && !matrix.available" type="warning">
            {{ matrix.unavailable_reason }}
          </n-alert>
          <matchup-heatmap
            :cells="matrix?.cells ?? []"
            :class="{ loading: matrixLoading }"
          />
        </n-card>
      </div>

      <training-decision-rail
        :focused-deck-label="selectedDeckLabel"
        :focused-standing="focusedStanding"
        :focused-standings="focusedStandings"
        :final-deck-label="finalDeckLabel"
        :final-standing="finalStanding"
        :selected-ranges="selectedRanges"
        @choose="chooseFinal"
        @inspect="selectDeck"
        @export="exportDecision"
      />
    </div>

    <div v-else class="empty-state">
      <strong>正在读取训练证据</strong>
      <span>本页不会因为数据缺失而创建评测任务。</span>
    </div>

    <training-matchup-drawer
      :show="selectedDeckLabel != null"
      v-model:range="detailRange"
      v-model:controller="detailController"
      v-model:seat="detailSeat"
      v-model:opponent-set="detailOpponentSet"
      :deck-label="selectedDeckLabel"
      :deck-hash="selectedDeckHash"
      :display-name="selectedDeckDisplayName"
      :detail="detail"
      :loading="detailLoading"
      :range-options="rangeOptions"
      :controller-options="controllerOptions"
      :seat-options="seatOptions"
      @update:show="!$event && (selectedDeckLabel = null)"
    />
      </section>
    </n-tab-pane>
    <n-tab-pane name="environment" tab="公开环境">
      <public-environment-panel
        :run="run"
      />
    </n-tab-pane>
    <n-tab-pane name="portfolio" tab="双席推荐">
      <two-deck-recommendation-panel
        :run="run"
        :checkpoint="checkpoint"
        :refresh-version="refreshVersion"
      />
    </n-tab-pane>
  </n-tabs>
</template>

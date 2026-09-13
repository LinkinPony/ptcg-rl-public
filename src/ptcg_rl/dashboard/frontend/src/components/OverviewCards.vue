<script setup lang="ts">
import { computed } from 'vue'
import { NCard, NGrid, NGridItem, NStatistic, NTag } from 'naive-ui'
import type { HealthPayload, PerformanceTable } from '../types'

const props = defineProps<{ table: PerformanceTable | null; health: HealthPayload | null }>()
const percent = (value: number | null | undefined) =>
  value == null ? '—' : `${(value * 100).toFixed(2)}%`
const version = (value: number | null | undefined) =>
  value == null || value < 0 ? '—' : `v${value}`
const count = (value: number | null | undefined) =>
  value == null ? '—' : value.toLocaleString('en-US')
const age = computed(() => {
  const seconds = props.health?.data_age_seconds
  return seconds == null ? '—' : `${Math.round(seconds)}s`
})
const inference = computed(() => props.health?.status.inference ?? null)
const lastSync = computed(() => inference.value?.last_sync ?? null)
const snapshotPool = computed(() => lastSync.value?.snapshot_pool ?? null)
const decisionsPerSecond = computed(() => {
  const value = inference.value?.decisions_per_second
  return value == null ? '— dec/s' : `${value.toFixed(1)} dec/s`
})
const deferred = computed(() => {
  const deferredVersion = lastSync.value?.deferred_weight_version
  if (deferredVersion == null) return '—'
  const reason = lastSync.value?.deferred_weight_reason
  return `${version(deferredVersion)}${reason ? ` · ${reason}` : ''}`
})
const residents = computed(() => {
  const values = snapshotPool.value?.resident_versions ?? []
  return values.length ? values.map((value) => version(value)).join(', ') : '—'
})
const leases = computed(() => {
  const values = snapshotPool.value?.leases_by_version ?? {}
  return Object.entries(values)
    .map(([leaseVersion, count]) => ({
      version: Number(leaseVersion),
      label: `v${leaseVersion} · ${count}`,
    }))
    .sort((left, right) => left.version - right.version)
})
const rolloutFeatures = computed(
  () => props.health?.status.actors?.rollout_features ?? null,
)
const runtime = computed(() => props.health?.status.runtime ?? null)
const learnerStatus = computed(() => props.health?.status.learner_status ?? null)
const statelessLearner = computed(() => {
  const status = learnerStatus.value
  return status?.format === 'simple-stateless-training-status-v1' ? status : null
})
const gibibytes = (value: number | null | undefined) =>
  value == null ? '—' : `${(value / 1024 ** 3).toFixed(1)} GiB`
const statelessProgress = computed(() => {
  const status = statelessLearner.value
  if (!status) return '—'
  return `${count(status.update_index)} / ${count(status.target_updates)}`
})
const statelessThroughput = computed(() => {
  const throughput = statelessLearner.value?.throughput
  if (!throughput) return '—'
  const rate = throughput.kept_decisions_per_second
  return [
    `${count(throughput.windows)} windows`,
    `${count(throughput.kept_decisions)} kept`,
    rate == null ? '— kept/s' : `${rate.toFixed(1)} kept/s`,
  ].join(' · ')
})
const statelessCollection = computed(() => {
  const collection = statelessLearner.value?.latest_collection
  if (!collection) return '—'
  const rate = collection.decisions_per_second
  return [
    `${count(collection.games_finished)} / ${count(collection.games_started)} games`,
    `${count(collection.candidate_decisions)} candidate`,
    `${count(collection.mirror_opponent_decisions)} mirror-opponent`,
    rate == null ? '— dec/s' : `${rate.toFixed(1)} dec/s`,
  ].join(' · ')
})
const statelessAllocator = computed(() => {
  const timing = statelessLearner.value?.latest_timing
  if (!timing) return '—'
  return [
    `active ${gibibytes(timing.cuda_checkpoint_peak_allocated_bytes)}`,
    `reserved ${gibibytes(timing.cuda_checkpoint_peak_reserved_bytes)}`,
    `retries ${count(timing.cuda_checkpoint_allocation_retries)}`,
    `OOM ${count(timing.cuda_checkpoint_ooms)}`,
  ].join(' · ')
})
const statelessAllocatorWarning = computed(() => {
  const timing = statelessLearner.value?.latest_timing
  return Boolean(
    (timing?.cuda_checkpoint_allocation_retries ?? 0) > 0 ||
      (timing?.cuda_checkpoint_ooms ?? 0) > 0,
  )
})
const finalSupervisor = computed(
  () => props.health?.status.summary?.supervisor ?? null,
)
const actorRecycles = computed(
  () => runtime.value?.actor_recycles ?? finalSupervisor.value?.actor_recycles ?? null,
)
const stalenessRefill = computed(
  () => props.health?.status.learner_status?.staleness_refill ?? null,
)
const hasLearnerRefillTelemetry = computed(() => {
  const refill = stalenessRefill.value
  return Boolean(
    refill &&
      [
        'accumulated_raw_decisions',
        'stale_decisions',
        'retained_budget_decisions',
        'remaining_decisions',
        'topup_chunks',
      ].some((field) => field in refill),
  )
})
const hasRecurrentTelemetry = computed(() => {
  const features = rolloutFeatures.value
  return Boolean(
    (features &&
      ('live_game_count' in features ||
        'stale_recurrent_recycle_polls' in features ||
        'stale_recurrent_games_recycled' in features)) ||
      runtime.value?.actor_recycle_requests != null ||
      actorRecycles.value != null ||
      hasLearnerRefillTelemetry.value,
  )
})
const learnerRefill = computed(() => {
  const refill = stalenessRefill.value
  if (!refill) return '—'
  return [
    `raw ${count(refill.accumulated_raw_decisions)}`,
    `stale ${count(refill.stale_decisions)}`,
    `retained ${count(refill.retained_budget_decisions)}`,
    `remaining ${count(refill.remaining_decisions)}`,
    `topups ${count(refill.topup_chunks)}`,
  ].join(' · ')
})
const liveGameSteps = computed(() => {
  const features = rolloutFeatures.value
  if (!features) return '—'
  const values: string[] = []
  if (features.live_game_count != null) {
    values.push(`${count(features.live_game_count)} live`)
  }
  for (const [label, value] of [
    ['p50', features.live_game_steps_p50],
    ['p90', features.live_game_steps_p90],
    ['p99', features.live_game_steps_p99],
  ] as const) {
    if (value != null) values.push(`${label} ${count(value)}`)
  }
  if (features.live_game_steps_max != null) {
    values.push(`max ${count(features.live_game_steps_max)}`)
  }
  if (
    features.max_live_game_steps_seen != null &&
    features.max_live_game_steps_seen !== features.live_game_steps_max
  ) {
    values.push(`peak ${count(features.max_live_game_steps_seen)}`)
  }
  return values.join(' · ') || '—'
})
const staleCandidateAge = computed(() => {
  const features = rolloutFeatures.value
  const value = features?.stale_recurrent_max_candidate_version_age
  const oldestVersion = features?.stale_recurrent_oldest_candidate_policy_version
  if (value == null && (oldestVersion == null || oldestVersion < 0)) return '—'
  const values = value == null ? [] : [`${count(value)} versions`]
  if (oldestVersion != null && oldestVersion >= 0) {
    values.push(`oldest ${version(oldestVersion)}`)
  }
  return values.join(' · ')
})
const recycledSequences = computed(() => {
  const features = rolloutFeatures.value
  const games = features?.stale_recurrent_games_recycled
  const sequences = features?.stale_recurrent_sequences_released
  if (games == null && sequences == null) return '—'
  return `${count(games)} games · ${count(sequences)} seq`
})
const deferredEvidence = computed(() => {
  const value =
    rolloutFeatures.value?.stale_recurrent_games_deferred_pending_evidence
  return value == null ? '—' : `${count(value)} observations`
})
const watchdog = computed(() => {
  const requests = runtime.value?.actor_recycle_requests
  if (requests == null && actorRecycles.value == null) return '—'
  return `req ${count(requests)} · done ${count(actorRecycles.value)}`
})
const recurrentWarning = computed(
  () =>
    (rolloutFeatures.value?.stale_recurrent_games_deferred_pending_evidence ?? 0) > 0,
)
</script>

<template>
  <div class="overview-stack">
    <n-grid cols="1 s:2 m:3 l:6" responsive="screen" :x-gap="14" :y-gap="14">
      <n-grid-item>
        <n-card class="metric-card metric-primary" :bordered="false"><n-statistic label="综合得分率" :value="percent(table?.overall.all?.score_rate)" /></n-card>
      </n-grid-item>
      <n-grid-item>
        <n-card class="metric-card metric-blue" :bordered="false"><n-statistic label="Self-play" :value="percent(table?.overall.self_play?.score_rate)" /></n-card>
      </n-grid-item>
      <n-grid-item>
        <n-card class="metric-card metric-violet" :bordered="false"><n-statistic label="Sentinel" :value="percent(table?.overall.sentinel?.score_rate)" /></n-card>
      </n-grid-item>
      <n-grid-item>
        <n-card class="metric-card metric-blue" :bordered="false"><n-statistic label="Adaptive history" :value="percent(table?.overall.adaptive_history?.score_rate)" /></n-card>
      </n-grid-item>
      <n-grid-item>
        <n-card class="metric-card metric-violet" :bordered="false"><n-statistic label="Scripted" :value="percent(table?.overall.scripted?.score_rate)" /></n-card>
      </n-grid-item>
      <n-grid-item>
        <n-card class="metric-card metric-health" :bordered="false">
          <n-statistic label="镜像数据年龄" :value="age" />
          <n-tag :type="health?.data_state === 'ready' ? 'success' : 'warning'" size="small">
            {{ health?.data_state ?? 'loading' }}
          </n-tag>
        </n-card>
      </n-grid-item>
    </n-grid>

    <n-card
      v-if="statelessLearner"
      class="inference-card stateless-card"
      :bordered="false"
    >
      <div class="inference-card__header">
        <div>
          <div class="inference-card__eyebrow">STATELESS PPO</div>
          <div class="inference-card__title">Simple-stateless learner</div>
        </div>
        <n-tag
          round
          :type="statelessAllocatorWarning ? 'warning' : 'success'"
          size="small"
          :bordered="false"
        >
          {{ statelessAllocatorWarning ? 'allocator warning' : 'healthy' }}
        </n-tag>
      </div>
      <div class="inference-card__metrics">
        <div class="inference-field">
          <span>Update</span>
          <strong>{{ statelessProgress }}</strong>
        </div>
        <div class="inference-field inference-field--wide">
          <span>Throughput</span>
          <strong>{{ statelessThroughput }}</strong>
        </div>
        <div class="inference-field inference-field--wide">
          <span>Latest cohort</span>
          <strong>{{ statelessCollection }}</strong>
        </div>
        <div class="inference-field inference-field--wide">
          <span>CUDA allocator peak</span>
          <strong>{{ statelessAllocator }}</strong>
        </div>
      </div>
    </n-card>

    <n-card class="inference-card" :bordered="false">
      <div class="inference-card__header">
        <div>
          <div class="inference-card__eyebrow">INFERENCE SNAPSHOTS</div>
          <div class="inference-card__title">推理服务</div>
        </div>
        <n-tag round type="info" size="small" :bordered="false">
          {{ decisionsPerSecond }}
        </n-tag>
      </div>
      <div class="inference-card__metrics">
        <div class="inference-field">
          <span>Current</span>
          <strong>{{ version(lastSync?.current_weight_version) }}</strong>
        </div>
        <div class="inference-field">
          <span>Deferred</span>
          <strong>{{ deferred }}</strong>
        </div>
        <div class="inference-field">
          <span>Min gap</span>
          <strong>{{ lastSync?.snapshot_min_version_gap ?? '—' }}</strong>
        </div>
        <div class="inference-field inference-field--wide">
          <span>Resident</span>
          <strong>{{ residents }}</strong>
        </div>
        <div class="inference-field inference-field--wide">
          <span>Leases by version</span>
          <div class="inference-leases">
            <n-tag
              v-for="lease in leases"
              :key="lease.version"
              size="small"
              :bordered="false"
            >
              {{ lease.label }}
            </n-tag>
            <strong v-if="!leases.length">—</strong>
          </div>
        </div>
      </div>
    </n-card>

    <n-card
      v-if="hasRecurrentTelemetry"
      class="inference-card recurrent-card"
      :bordered="false"
    >
      <div class="inference-card__header">
        <div>
          <div class="inference-card__eyebrow recurrent-card__eyebrow">
            RECURRENT LIVENESS
          </div>
          <div class="inference-card__title">循环序列存活性</div>
        </div>
        <n-tag
          round
          :type="recurrentWarning ? 'warning' : 'success'"
          size="small"
          :bordered="false"
        >
          {{ recurrentWarning ? 'deferred observed' : 'healthy' }}
        </n-tag>
      </div>
      <div class="inference-card__metrics recurrent-card__metrics">
        <div class="inference-field inference-field--wide">
          <span>Live game steps</span>
          <strong>{{ liveGameSteps }}</strong>
        </div>
        <div class="inference-field">
          <span>Candidate age</span>
          <strong>{{ staleCandidateAge }}</strong>
        </div>
        <div class="inference-field">
          <span>Stale recycled</span>
          <strong>{{ recycledSequences }}</strong>
        </div>
        <div class="inference-field">
          <span>Deferred evidence</span>
          <strong>{{ deferredEvidence }}</strong>
        </div>
        <div class="inference-field">
          <span>Actor watchdog</span>
          <strong>{{ watchdog }}</strong>
        </div>
        <div
          v-if="hasLearnerRefillTelemetry"
          class="inference-field inference-field--wide"
        >
          <span>Learner refill</span>
          <strong>{{ learnerRefill }}</strong>
        </div>
      </div>
    </n-card>
  </div>
</template>

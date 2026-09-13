<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { NAlert, NButton, NCard, NSelect, NTag } from 'naive-ui'
import { api } from '../api'
import { compactHash, decimal, percent } from '../format'
import type {
  CheckpointInfo,
  ComparisonPayload,
  ComparisonReference,
  RunInfo,
} from '../types'

const props = defineProps<{
  runs: RunInfo[]
  currentRun: string
  currentCheckpoint: number | null
}>()

const references = ref<ComparisonReference[]>([])
const checkpointCatalog = ref<Record<string, CheckpointInfo[]>>({})
const result = ref<ComparisonPayload | null>(null)
const loading = ref(false)
const error = ref('')
let initialized = false

const runOptions = computed(() => props.runs.map((run) => ({
  label: `${run.current ? 'CURRENT · ' : ''}${run.run_id}`,
  value: run.run_id,
})))

watch(
  () => [props.currentRun, props.runs.length] as const,
  async () => {
    if (initialized || !props.currentRun || props.runs.length < 1) return
    initialized = true
    const alternative = props.runs.find((run) => run.run_id !== props.currentRun)
    references.value = [
      { run_id: props.currentRun, checkpoint_version: props.currentCheckpoint },
      {
        run_id: alternative?.run_id ?? props.currentRun,
        checkpoint_version: null,
      },
    ]
    await Promise.all(references.value.map((_, index) => loadCheckpoints(index)))
  },
  { immediate: true },
)

async function loadCheckpoints(index: number) {
  const reference = references.value[index]
  if (!reference) return
  if (!checkpointCatalog.value[reference.run_id]) {
    checkpointCatalog.value[reference.run_id] = await api.checkpoints(reference.run_id)
  }
  const options = checkpointCatalog.value[reference.run_id] ?? []
  if (
    reference.checkpoint_version == null
    || !options.some((checkpoint) => checkpoint.version === reference.checkpoint_version)
  ) {
    reference.checkpoint_version = options[0]?.version ?? null
  }
}

function checkpointOptions(reference: ComparisonReference) {
  return (checkpointCatalog.value[reference.run_id] ?? []).map((checkpoint) => ({
    label: `v${checkpoint.version}${checkpoint.metric_available ? ' · metrics' : ''}`,
    value: checkpoint.version,
  }))
}

async function changeRun(index: number, runId: string) {
  const reference = references.value[index]
  if (!reference) return
  reference.run_id = runId
  reference.checkpoint_version = null
  result.value = null
  await loadCheckpoints(index)
}

async function addReference() {
  if (references.value.length >= 4) return
  const candidate = props.runs.find(
    (run) => !references.value.some((reference) => reference.run_id === run.run_id),
  ) ?? props.runs[0]
  if (!candidate) return
  references.value.push({ run_id: candidate.run_id, checkpoint_version: null })
  await loadCheckpoints(references.value.length - 1)
}

async function compare() {
  loading.value = true
  error.value = ''
  try {
    result.value = await api.compare(references.value)
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : String(reason)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <section class="page-stack">
    <div class="page-heading">
      <div>
        <span class="page-eyebrow">ARTIFACT-AWARE COMPARISON</span>
        <h1>Run / Checkpoint 对比</h1>
        <p>同屏比较 2–4 个 immutable pair；模型或 roster 身份不同会显式标记为不可直接归因。</p>
      </div>
      <n-button type="primary" :loading="loading" :disabled="references.length < 2" @click="compare">
        运行对比
      </n-button>
    </div>

    <n-alert v-if="error" type="error">{{ error }}</n-alert>

    <n-card class="work-card" :bordered="false">
      <template #header>
        <div class="section-title"><span>比较对象</span><small>选择明确的 run 与 checkpoint version</small></div>
      </template>
      <template #header-extra>
        <n-button size="small" :disabled="references.length >= 4" @click="addReference">添加对象</n-button>
      </template>
      <div class="reference-list">
        <div v-for="(reference, index) in references" :key="index" class="reference-row">
          <span class="reference-index">{{ index + 1 }}</span>
          <n-select
            :value="reference.run_id"
            :options="runOptions"
            filterable
            class="reference-run"
            @update:value="changeRun(index, $event)"
          />
          <n-select
            v-model:value="reference.checkpoint_version"
            :options="checkpointOptions(reference)"
            class="reference-checkpoint"
            placeholder="checkpoint"
          />
          <n-button
            quaternary
            type="error"
            :disabled="references.length <= 2"
            @click="references.splice(index, 1); result = null"
          >
            移除
          </n-button>
        </div>
      </div>
    </n-card>

    <template v-if="result">
      <n-alert :type="result.all_compatible ? 'success' : 'warning'">
        {{
          result.all_compatible
            ? '所选对象具有相同 model + training roster identity，可进行同口径比较。'
            : '存在 model 或 roster identity 差异；结果可并列查看，但不能把差异直接归因于训练进度。'
        }}
      </n-alert>
      <div class="comparison-grid">
        <n-card v-for="(row, index) in result.rows" :key="index" class="comparison-card" :bordered="false">
          <div class="comparison-card__header">
            <span>OBJECT {{ index + 1 }}</span>
            <n-tag size="small" :type="row.learner_metric ? 'success' : 'warning'">
              {{ row.learner_metric ? 'METRICS' : 'PARTIAL' }}
            </n-tag>
          </div>
          <strong class="comparison-card__run">{{ row.reference.run_id }}</strong>
          <div class="comparison-version">
            v{{ row.checkpoint?.version ?? '—' }}
            <span class="mono">{{ compactHash(row.checkpoint?.pair_manifest_sha256) }}</span>
          </div>
          <div class="comparison-facts">
            <div><span>Training pool posterior</span><strong>{{ percent(row.training_pool.posterior_mean, 2) }}</strong></div>
            <div><span>95% CI</span><strong>{{ percent(row.training_pool.credible_low) }}–{{ percent(row.training_pool.credible_high) }}</strong></div>
            <div><span>Loss</span><strong>{{ decimal(row.learner_metric?.loss) }}</strong></div>
            <div><span>Approx KL</span><strong>{{ decimal(row.learner_metric?.approximate_kl, 5) }}</strong></div>
            <div><span>Entropy</span><strong>{{ decimal(row.learner_metric?.entropy) }}</strong></div>
            <div><span>Kept dec/s</span><strong>{{ row.learner_metric?.kept_decisions_per_second.toFixed(1) ?? '—' }}</strong></div>
          </div>
          <div class="identity-block">
            <span>MODEL + ROSTER IDENTITY</span>
            <code>{{ row.compatible_group ?? 'unknown' }}</code>
          </div>
          <small v-for="warning in row.warnings" :key="warning" class="comparison-warning">{{ warning }}</small>
        </n-card>
      </div>
    </template>
  </section>
</template>

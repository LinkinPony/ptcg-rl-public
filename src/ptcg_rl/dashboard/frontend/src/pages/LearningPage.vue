<script setup lang="ts">
import { computed, ref } from 'vue'
import { NAlert, NCard, NSelect, NTag } from 'naive-ui'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart } from 'echarts/charts'
import {
  DataZoomComponent,
  GridComponent,
  MarkLineComponent,
  TooltipComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { decimal, duration, gibibytes, percent } from '../format'
import type {
  LearnerMetricRecord,
  LearnerSeriesPayload,
  WorkbenchSummary,
} from '../types'

use([
  CanvasRenderer,
  LineChart,
  DataZoomComponent,
  GridComponent,
  MarkLineComponent,
  TooltipComponent,
])

const props = defineProps<{
  summary: WorkbenchSummary | null
  series: LearnerSeriesPayload | null
  checkpoint: number | null
}>()

type MetricKey =
  | 'loss'
  | 'policy_loss'
  | 'value_loss'
  | 'entropy'
  | 'approximate_kl'
  | 'clip_fraction'
  | 'gradient_norm'
  | 'learning_rate'
  | 'kept_decisions_per_second'

const metric = ref<MetricKey>('approximate_kl')
const metrics: { label: string; value: MetricKey; format: (value: number) => string }[] = [
  { label: 'Approximate KL', value: 'approximate_kl', format: (value) => value.toFixed(5) },
  { label: 'Total loss', value: 'loss', format: (value) => value.toFixed(5) },
  { label: 'Policy loss', value: 'policy_loss', format: (value) => value.toFixed(5) },
  { label: 'Value loss', value: 'value_loss', format: (value) => value.toFixed(5) },
  { label: 'Entropy', value: 'entropy', format: (value) => value.toFixed(4) },
  { label: 'Clip fraction', value: 'clip_fraction', format: (value) => `${(value * 100).toFixed(1)}%` },
  { label: 'Gradient norm', value: 'gradient_norm', format: (value) => value.toFixed(4) },
  { label: 'Learning rate', value: 'learning_rate', format: (value) => value.toExponential(2) },
  { label: 'Kept dec/s', value: 'kept_decisions_per_second', format: (value) => value.toFixed(1) },
]
const selectedMetric = computed(() => metrics.find((item) => item.value === metric.value) ?? metrics[0])

const option = computed(() => ({
  animationDuration: 260,
  backgroundColor: 'transparent',
  grid: { left: 66, right: 24, top: 22, bottom: 56 },
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(7, 12, 20, .97)',
    borderColor: 'rgba(148, 163, 184, .2)',
    textStyle: { color: '#e8eef6' },
    formatter: (raw: { value: [number, number]; dataIndex: number }[]) => {
      const item = raw[0]
      const record = props.series?.records[item.dataIndex]
      if (!record) return ''
      return `<strong>checkpoint v${record.checkpoint_version}</strong><br/>`
        + `${selectedMetric.value.label}: ${selectedMetric.value.format(item.value[1])}<br/>`
        + `<span style="color:#8190a5">${record.decisions.toLocaleString()} decisions · `
        + `${record.fragments_stale.toLocaleString()} stale</span>`
    },
  },
  xAxis: {
    type: 'value',
    name: 'checkpoint version',
    nameTextStyle: { color: '#65758b' },
    axisLabel: { color: '#718198' },
    axisLine: { lineStyle: { color: 'rgba(148,163,184,.14)' } },
    splitLine: { show: false },
  },
  yAxis: {
    type: 'value',
    scale: true,
    axisLabel: { color: '#718198' },
    splitLine: { lineStyle: { color: 'rgba(148,163,184,.07)' } },
  },
  dataZoom: [{ type: 'inside' }, { type: 'slider', height: 14, bottom: 12 }],
  series: [{
    type: 'line',
    smooth: 0.18,
    showSymbol: true,
    symbolSize: 7,
    lineStyle: { width: 2.5, color: '#63e6be' },
    itemStyle: { color: '#8cefd1', borderColor: '#0f1723', borderWidth: 2 },
    data: (props.series?.records ?? []).map((record: LearnerMetricRecord) => [
      record.checkpoint_version,
      record[metric.value],
    ]),
    markLine: props.checkpoint == null ? undefined : {
      symbol: 'none',
      label: { color: '#a9b7ca', formatter: `selected v${props.checkpoint}` },
      lineStyle: { color: '#74c0fc', type: 'dashed' },
      data: [{ xAxis: props.checkpoint }],
    },
  }],
}))
</script>

<template>
  <section class="page-stack">
    <div class="page-heading">
      <div>
        <span class="page-eyebrow">OPTIMIZATION & THROUGHPUT</span>
        <h1>学习</h1>
        <p>将 checkpoint 级 learner 指标与实时快照分开，避免把一次波动误判为趋势。</p>
      </div>
      <n-tag :type="series?.complete ? 'success' : 'warning'" round>
        {{ series?.complete ? 'ARTIFACT-BOUND HISTORY' : 'PARTIAL HISTORY' }}
      </n-tag>
    </div>

    <n-alert v-if="series?.warning" type="warning" title="历史存在显式缺口">
      {{ series.warning }}
    </n-alert>

    <div class="two-column-grid two-column-grid--learning">
      <n-card class="work-card" :bordered="false">
        <template #header>
          <div class="section-title"><span>当前优化信号</span><small>latest learner_status snapshot</small></div>
        </template>
        <div class="fact-grid fact-grid--three">
          <div><span>Total loss</span><strong>{{ decimal(summary?.latest_learner.loss) }}</strong></div>
          <div><span>Policy loss</span><strong>{{ decimal(summary?.latest_learner.policy_loss) }}</strong></div>
          <div><span>Value loss</span><strong>{{ decimal(summary?.latest_learner.value_loss) }}</strong></div>
          <div><span>Entropy</span><strong>{{ decimal(summary?.latest_learner.entropy) }}</strong></div>
          <div><span>Approx KL</span><strong>{{ decimal(summary?.latest_learner.approximate_kl, 5) }}</strong></div>
          <div><span>Clip fraction</span><strong>{{ percent(summary?.latest_learner.clip_fraction) }}</strong></div>
          <div><span>Gradient norm</span><strong>{{ decimal(summary?.latest_learner.gradient_norm) }}</strong></div>
          <div><span>Learning rate</span><strong>{{ summary?.latest_learner.learning_rate?.toExponential(2) ?? '—' }}</strong></div>
          <div><span>Stale fragments</span><strong>{{ summary?.latest_learner.fragments_stale?.toLocaleString() ?? '—' }}</strong></div>
        </div>
      </n-card>

      <n-card class="work-card" :bordered="false">
        <template #header>
          <div class="section-title"><span>窗口耗时与资源</span><small>发现采集、learner 或 checkpoint 瓶颈</small></div>
        </template>
        <div class="timing-stack">
          <div><span>Collection</span><strong>{{ duration(summary?.latest_learner.collection_seconds) }}</strong></div>
          <div><span>Learner</span><strong>{{ duration(summary?.latest_learner.learner_seconds) }}</strong></div>
          <div><span>Checkpoint</span><strong>{{ duration(summary?.latest_learner.checkpoint_seconds) }}</strong></div>
          <div><span>CUDA allocated peak</span><strong>{{ gibibytes(summary?.latest_learner.cuda_peak_allocated_bytes) }}</strong></div>
          <div><span>CUDA reserved peak</span><strong>{{ gibibytes(summary?.latest_learner.cuda_peak_reserved_bytes) }}</strong></div>
        </div>
      </n-card>
    </div>

    <n-card class="work-card" :bordered="false">
      <template #header>
        <div class="section-title"><span>Checkpoint 趋势</span><small>每个点绑定 immutable pair fingerprint</small></div>
      </template>
      <template #header-extra>
        <n-select v-model:value="metric" :options="metrics" class="metric-select" />
      </template>
      <v-chart v-if="series?.records.length" class="learner-chart" :option="option" autoresize />
      <div v-else class="empty-state">
        <strong>等待首个新 checkpoint 指标分片</strong>
        <span>旧 run 不会由 WebUI 轮询结果伪造回填；当前实时值仍显示在上方。</span>
      </div>
    </n-card>
  </section>
</template>

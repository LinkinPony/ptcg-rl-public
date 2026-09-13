<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart } from 'echarts/charts'
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { TrainingDeckSeries } from '../types'

use([
  CanvasRenderer,
  LineChart,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
])

interface TooltipParam {
  marker: string
  seriesName: string
  value: [string, number | null, number, number | null]
}

const props = defineProps<{ series: TrainingDeckSeries[] }>()
const palette = ['#63e6be', '#74c0fc', '#b197fc', '#ff8787', '#ffd43b', '#f783ac', '#69db7c']
const percent = (value: number | null) => value == null ? '—' : `${(value * 100).toFixed(1)}%`
const timeLabel = (value: string) => new Intl.DateTimeFormat('zh-CN', {
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
}).format(new Date(value))

const observedSeries = computed(() => props.series
  .map((deck) => ({
    ...deck,
    points: deck.points.filter((point) => point.score_rate != null),
  }))
  .filter((deck) => deck.points.length > 0))

const option = computed(() => ({
  backgroundColor: 'transparent',
  animationDuration: 450,
  animationDurationUpdate: 280,
  color: palette,
  tooltip: {
    trigger: 'axis',
    confine: true,
    backgroundColor: 'rgba(10, 15, 25, 0.96)',
    borderColor: 'rgba(148, 163, 184, 0.22)',
    borderWidth: 1,
    padding: [12, 14],
    textStyle: { color: '#e7edf6', fontSize: 12 },
    axisPointer: {
      type: 'line',
      lineStyle: { color: 'rgba(148, 163, 184, 0.24)', width: 1 },
    },
    formatter: (raw: TooltipParam | TooltipParam[]) => {
      const params = Array.isArray(raw) ? raw : [raw]
      if (!params.length) return ''
      const rows = params
        .filter((item) => item.value[1] != null)
        .sort((a, b) => (b.value[1] ?? -1) - (a.value[1] ?? -1))
        .map((item) => (
          `<div class="chart-tooltip-row">${item.marker}<span>${item.seriesName}</span>`
          + `<strong>${percent(item.value[1])}</strong><small>${item.value[2]}局</small></div>`
        ))
        .join('')
      return `<div class="chart-tooltip-time">${timeLabel(params[0].value[0])}</div>${rows}`
    },
  },
  legend: {
    type: 'scroll',
    top: 8,
    left: 8,
    right: 8,
    icon: 'roundRect',
    itemWidth: 18,
    itemHeight: 4,
    itemGap: 18,
    pageIconColor: '#7dd3fc',
    pageIconInactiveColor: '#475569',
    pageTextStyle: { color: '#8b9bb0' },
    textStyle: { color: '#b7c2d2', fontSize: 12 },
  },
  grid: { left: 58, right: 24, top: 72, bottom: 42, containLabel: false },
  dataZoom: [{ type: 'inside', filterMode: 'none', zoomOnMouseWheel: 'shift' }],
  xAxis: {
    type: 'time',
    boundaryGap: false,
    axisLine: { lineStyle: { color: 'rgba(148, 163, 184, 0.16)' } },
    axisTick: { show: false },
    axisLabel: {
      color: '#718198',
      hideOverlap: true,
      formatter: (value: number) => timeLabel(new Date(value).toISOString()),
    },
    splitLine: { show: false },
  },
  yAxis: {
    type: 'value',
    min: 0,
    max: 1,
    splitNumber: 4,
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: {
      color: '#718198',
      formatter: (value: number) => `${Math.round(value * 100)}%`,
    },
    splitLine: {
      lineStyle: { color: 'rgba(148, 163, 184, 0.075)', width: 1 },
    },
  },
  series: observedSeries.value.map((deck, index) => ({
    id: deck.deck_label,
    name: deck.display_name,
    type: 'line',
    showSymbol: deck.points.length === 1,
    symbolSize: 7,
    smooth: 0.22,
    connectNulls: false,
    sampling: 'lttb',
    z: 10 + index,
    data: deck.points.map((point) => [
      point.ended_at_utc,
      point.score_rate,
      point.games,
      point.score_rate,
    ]),
    lineStyle: { width: 2.8, opacity: 0.94 },
    emphasis: {
      focus: 'series',
      lineStyle: { width: 4 },
    },
    blur: { lineStyle: { opacity: 0.12 } },
  })),
}))
</script>

<template>
  <v-chart v-if="observedSeries.length" class="chart" :option="option" autoresize />
  <div v-else class="chart-empty">等待首个完整的训练统计窗口…</div>
</template>

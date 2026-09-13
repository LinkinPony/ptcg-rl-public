<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { HeatmapChart } from 'echarts/charts'
import {
  DataZoomComponent,
  GridComponent,
  TooltipComponent,
  VisualMapComponent,
} from 'echarts/components'
import type { HeatmapPaletteName, TrainingMatrixCell } from '../types'

use([
  CanvasRenderer,
  HeatmapChart,
  DataZoomComponent,
  GridComponent,
  TooltipComponent,
  VisualMapComponent,
])
const props = withDefaults(defineProps<{
  cells: TrainingMatrixCell[]
  palette?: HeatmapPaletteName
}>(), {
  palette: 'semantic',
})

const palettes: Record<HeatmapPaletteName, string[]> = {
  semantic: ['#8f3f55', '#51364c', '#273344', '#276d69', '#43c59e'],
  viridis: ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
  icefire: ['#285f92', '#4da3bd', '#263242', '#bd684d', '#f1b45b'],
  teal: ['#172038', '#24445e', '#287180', '#32a38e', '#8bdc9d'],
}

const compact = (value: string, limit = 28) => (
  value.length <= limit ? value : `${value.slice(0, limit - 1)}…`
)

const option = computed(() => {
  const candidateNames = new Map<string, string>()
  const candidateHashes = new Map<string, string>()
  const opponentNames = new Map<string, string>()
  const opponentHashes = new Map<string, string | null>()
  const opponentGames = new Map<string, number>()
  for (const cell of props.cells) {
    candidateNames.set(cell.candidate_deck_label, cell.candidate_display_name)
    candidateHashes.set(cell.candidate_deck_label, cell.candidate_deck_hash)
    opponentNames.set(cell.opponent_deck_label, cell.opponent_display_name)
    opponentHashes.set(cell.opponent_deck_label, cell.opponent_deck_hash)
    opponentGames.set(
      cell.opponent_deck_label,
      (opponentGames.get(cell.opponent_deck_label) ?? 0)
        + cell.posterior.observed.games,
    )
  }
  const candidates = [...candidateNames.keys()]
  const opponents = [...opponentGames]
    .sort((left, right) => right[1] - left[1])
    .map(([opponent]) => opponent)
  const data = props.cells
    .filter((cell) => cell.posterior.posterior_mean != null)
    .map((cell) => [
      opponents.indexOf(cell.opponent_deck_label),
      candidates.indexOf(cell.candidate_deck_label),
      cell.posterior.posterior_mean,
      cell.posterior.observed.games,
      cell.pilot_count,
    ])
  return {
    backgroundColor: 'transparent',
    tooltip: {
      backgroundColor: 'rgba(10, 15, 25, 0.96)',
      borderColor: 'rgba(148, 163, 184, 0.22)',
      textStyle: { color: '#e7edf6' },
      formatter: (p: { value: [number, number, number, number, number] }) => {
        const candidate = candidates[p.value[1]]
        const opponent = opponents[p.value[0]]
        const candidateHash = candidateHashes.get(candidate)
        const opponentHash = opponentHashes.get(opponent)
        return `${candidateNames.get(candidate) ?? candidate}`
          + `${candidateHash ? `<br/><span class="heatmap-hash">${candidateHash}</span>` : ''}`
          + `<br/><span class="heatmap-vs">vs</span> ${opponentNames.get(opponent) ?? opponent}`
          + `${opponentHash ? `<br/><span class="heatmap-hash">${opponentHash}</span>` : ''}`
          + `<br/><strong>${(p.value[2] * 100).toFixed(2)}%</strong>`
          + ` · ${p.value[3]}局 · ${p.value[4]} pilots`
      },
    },
    grid: { left: 235, right: 28, top: 20, bottom: 190 },
    dataZoom: [
      {
        type: 'inside',
        xAxisIndex: 0,
        startValue: 0,
        endValue: Math.min(opponents.length - 1, 17),
        filterMode: 'weakFilter',
      },
      {
        type: 'slider',
        xAxisIndex: 0,
        startValue: 0,
        endValue: Math.min(opponents.length - 1, 17),
        bottom: 62,
        height: 14,
        borderColor: 'rgba(148, 163, 184, 0.1)',
        backgroundColor: 'rgba(8, 13, 21, 0.5)',
        fillerColor: 'rgba(99, 230, 190, 0.12)',
        handleStyle: { color: '#63e6be', borderColor: '#63e6be' },
        moveHandleStyle: { color: '#63e6be' },
        textStyle: { color: '#66758a' },
      },
    ],
    xAxis: {
      type: 'category',
      data: opponents,
      axisLine: { lineStyle: { color: 'rgba(148, 163, 184, 0.14)' } },
      axisTick: { show: false },
      axisLabel: {
        rotate: 38,
        color: '#8796aa',
        lineHeight: 15,
        width: 145,
        overflow: 'truncate',
        formatter: (opponent: string) => {
          const name = compact(opponentNames.get(opponent) ?? opponent)
          const hash = opponentHashes.get(opponent)
          return hash ? `${name}\n${hash}` : name
        },
      },
    },
    yAxis: {
      type: 'category',
      data: candidates,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        color: '#aeb8c8',
        lineHeight: 15,
        width: 205,
        overflow: 'truncate',
        formatter: (candidate: string) => {
          const name = compact(candidateNames.get(candidate) ?? candidate, 32)
          const hash = candidateHashes.get(candidate)
          return hash ? `${name}\n${hash}` : name
        },
      },
    },
    visualMap: {
      min: 0,
      max: 1,
      dimension: 2,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 10,
      text: ['高后验', '低后验'],
      precision: 2,
      inRange: { color: palettes[props.palette] },
      textStyle: { color: '#8492a7' },
    },
    series: [{
      type: 'heatmap',
      data,
      itemStyle: { borderColor: '#111827', borderWidth: 2, borderRadius: 3 },
      emphasis: { itemStyle: { borderColor: '#d9f9ef', borderWidth: 2 } },
      label: {
        show: true,
        color: '#f4f7fb',
        fontWeight: 600,
        formatter: (p: { value: [number, number, number] }) =>
          `${(p.value[2] * 100).toFixed(0)}%`,
      },
    }],
  }
})
</script>

<template>
  <v-chart v-if="cells.length" class="heatmap" :option="option" autoresize />
  <div v-else class="chart-empty">当前筛选没有 matchup 观测。</div>
</template>

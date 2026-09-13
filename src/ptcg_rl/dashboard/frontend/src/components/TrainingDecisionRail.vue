<script setup lang="ts">
import { NAlert, NButton, NTag } from 'naive-ui'
import { percent } from '../format'
import type {
  TrainingDeckStanding,
  TrainingEvidenceRange,
} from '../types'

defineProps<{
  focusedDeckLabel: string | null
  focusedStanding: TrainingDeckStanding | null
  focusedStandings: Partial<Record<TrainingEvidenceRange, TrainingDeckStanding>>
  finalDeckLabel: string | null
  finalStanding: TrainingDeckStanding | null
  selectedRanges: TrainingEvidenceRange[]
}>()

const emit = defineEmits<{
  choose: [deckLabel: string]
  inspect: [deckLabel: string]
  export: [format: 'json' | 'markdown']
}>()

const rangeLabels: Record<TrainingEvidenceRange, string> = {
  checkpoint: '所选 Checkpoint',
  recent_15m: '近 15 分钟',
  recent_60m: '近 60 分钟',
  cumulative: 'Run 累计',
}
</script>

<template>
  <aside class="training-decision-rail">
    <div class="decision-rail-heading">
      <span class="page-eyebrow">INSPECT &amp; DECIDE</span>
      <n-tag
        v-if="focusedDeckLabel"
        size="small"
        round
        :type="focusedDeckLabel === finalDeckLabel ? 'success' : 'info'"
      >
        {{ focusedDeckLabel === finalDeckLabel ? '已选择' : '查看中' }}
      </n-tag>
    </div>
    <template v-if="focusedDeckLabel && focusedStanding">
      <strong>{{ focusedStanding.display_name }}</strong>
      <span v-if="focusedStanding.family_display_name" class="decision-family">
        {{ focusedStanding.family_display_name }}
      </span>
      <small class="mono">旧编号 · {{ focusedStanding.deck_hash }}</small>
      <n-alert v-if="focusedStanding.route_compatible === false" type="error">
        该 exact deck 不在所选 checkpoint 的 route registry 中，不能确认。
      </n-alert>
      <div
        v-for="range in selectedRanges"
        :key="range"
        class="decision-range-card"
      >
        <span>{{ rangeLabels[range] }}</span>
        <template v-if="focusedStandings[range]?.posterior.evidence_state === 'ready'">
          <strong>
            #{{ focusedStandings[range]?.rank }}
            · {{ percent(focusedStandings[range]?.posterior.posterior_mean ?? null, 1) }}
          </strong>
          <small>{{ focusedStandings[range]?.posterior.observed.games.toLocaleString() }} 局</small>
          <div class="decision-controller-grid">
            <span>Self <b>{{ percent(focusedStandings[range]?.controller_scores.self_play?.score_rate ?? null, 0) }}</b></span>
            <span>Sentinel <b>{{ percent(focusedStandings[range]?.controller_scores.sentinel?.score_rate ?? null, 0) }}</b></span>
            <span>Adaptive <b>{{ percent(focusedStandings[range]?.controller_scores.adaptive_history?.score_rate ?? null, 0) }}</b></span>
            <span>Scripted <b>{{ percent(focusedStandings[range]?.controller_scores.scripted?.score_rate ?? null, 0) }}</b></span>
          </div>
          <small>
            强：{{ focusedStandings[range]?.strongest_matchup?.opponent_display_name ?? '—' }}
            · 弱：{{ focusedStandings[range]?.weakest_matchup?.opponent_display_name ?? '—' }}
          </small>
        </template>
        <strong v-else>不可用</strong>
      </div>
      <n-button
        v-if="focusedDeckLabel !== finalDeckLabel"
        block
        type="primary"
        class="decision-primary-action"
        :disabled="focusedStanding.route_compatible === false"
        @click="emit('choose', focusedDeckLabel)"
      >
        设为最终选择
      </n-button>
    </template>
    <template v-else>
      <strong>选择一行开始查看</strong>
      <small>表格按 family 聚合 exact deck；点击任意卡组即可查看完整证据并确认。</small>
    </template>

    <div v-if="finalDeckLabel && finalStanding" class="decision-confirmed">
      <span>FINAL SELECTION</span>
      <strong>{{ finalStanding.display_name }}</strong>
      <small class="mono">旧编号 · {{ finalStanding.deck_hash }}</small>
      <n-button
        v-if="focusedDeckLabel !== finalDeckLabel"
        block
        quaternary
        @click="emit('inspect', finalDeckLabel)"
      >
        返回已选卡组
      </n-button>
      <div class="decision-export-actions">
        <n-button block secondary @click="emit('export', 'json')">导出 JSON</n-button>
        <n-button block secondary @click="emit('export', 'markdown')">导出 Markdown</n-button>
      </div>
      <small>本地决定绑定所选 checkpoint；不会创建 profile 或上传 submission。</small>
    </div>
  </aside>
</template>

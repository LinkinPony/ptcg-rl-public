<script setup lang="ts">
import { computed } from 'vue'
import { NAlert, NCard, NGrid, NGridItem, NProgress, NTag } from 'naive-ui'
import { compactHash, count, decimal, gibibytes, percent, relativeAge } from '../format'
import type { WorkbenchSummary } from '../types'

const props = defineProps<{ summary: WorkbenchSummary | null; loading: boolean }>()

const progress = computed(() => {
  const current = props.summary?.progress.update_index
  const target = props.summary?.progress.target_updates
  if (current == null || target == null || target === 0) return 0
  return Math.min(100, (current / target) * 100)
})
const statusLabel = computed(() => {
  const alerts = props.summary?.alerts ?? []
  if (alerts.some((alert) => alert.severity === 'critical')) return 'NEEDS ATTENTION'
  if (alerts.some((alert) => alert.severity === 'warning')) return 'WATCH'
  return 'HEALTHY'
})
</script>

<template>
  <section class="page-stack">
    <div class="page-heading">
      <div>
        <span class="page-eyebrow">LIVE COMMAND CENTER</span>
        <h1>现在</h1>
        <p>一屏判断训练是否健康、是否仍在学习，以及是否需要人工介入。</p>
      </div>
      <n-tag
        round
        size="large"
        :type="statusLabel === 'HEALTHY' ? 'success' : statusLabel === 'WATCH' ? 'warning' : 'error'"
      >
        {{ statusLabel }}
      </n-tag>
    </div>

    <n-grid cols="1 s:2 l:5" responsive="screen" :x-gap="12" :y-gap="12">
      <n-grid-item>
        <n-card class="signal-card signal-card--accent" :bordered="false">
          <span>训练池后验得分</span>
          <strong>{{ percent(summary?.evidence.posterior_mean, 2) }}</strong>
          <small v-if="summary?.evidence.seat_balanced">
            95% CI {{ percent(summary.evidence.credible_low) }}–{{ percent(summary.evidence.credible_high) }}
          </small>
          <small v-else>seat 证据不完整，仅显示 observed W/D/L</small>
        </n-card>
      </n-grid-item>
      <n-grid-item>
        <n-card class="signal-card" :bordered="false">
          <span>证据规模</span>
          <strong>{{ count(summary?.evidence.observed.games) }}</strong>
          <small>
            W {{ count(summary?.evidence.observed.wins) }} ·
            D {{ count(summary?.evidence.observed.draws) }} ·
            L {{ count(summary?.evidence.observed.losses) }}
          </small>
        </n-card>
      </n-grid-item>
      <n-grid-item>
        <n-card class="signal-card signal-card--accent" :bordered="false">
          <span>累计训练对局</span>
          <strong>
            {{ summary?.training_games?.is_lower_bound ? '≥ ' : '' }}{{ count(summary?.training_games?.total_games) }}
          </strong>
          <small>
            当前段 {{ count(summary?.training_games?.selected_run_games) }} ·
            {{ count(summary?.training_games?.counted_run_count) }} 段有终局 telemetry
          </small>
          <small v-if="summary?.training_games?.is_lower_bound">
            {{ count(summary.training_games.unreported_run_count) }} 段缺历史 ·
            {{ count(summary.training_games.invalid_run_count) }} 段无效
          </small>
        </n-card>
      </n-grid-item>
      <n-grid-item>
        <n-card class="signal-card" :bordered="false">
          <span>保留决策吞吐</span>
          <strong>{{ count(summary?.progress.kept_decisions_per_second) }}</strong>
          <small>kept decisions / second</small>
        </n-card>
      </n-grid-item>
      <n-grid-item>
        <n-card class="signal-card" :bordered="false">
          <span>数据年龄</span>
          <strong>{{ relativeAge(summary?.data_age_seconds) }}</strong>
          <small>{{ summary?.data_state ?? 'loading' }} · performance mirror</small>
        </n-card>
      </n-grid-item>
    </n-grid>

    <div class="two-column-grid">
      <n-card class="work-card" :bordered="false">
        <template #header>
          <div class="section-title"><span>训练进度</span><small>learner / immutable checkpoint</small></div>
        </template>
        <div class="progress-block">
          <div class="progress-block__headline">
            <strong>v{{ summary?.progress.update_index ?? '—' }}</strong>
            <span>/ {{ count(summary?.progress.target_updates) }} updates</span>
          </div>
          <n-progress
            type="line"
            :percentage="progress"
            :show-indicator="false"
            :height="7"
            rail-color="rgba(148,163,184,.12)"
          />
          <div class="fact-grid">
            <div><span>Optimizer step</span><strong>{{ count(summary?.progress.optimizer_step_index) }}</strong></div>
            <div><span>Decisions seen</span><strong>{{ count(summary?.progress.decisions_seen) }}</strong></div>
            <div><span>Latest pair</span><strong>v{{ summary?.checkpoint?.version ?? '—' }}</strong></div>
            <div><span>Pair SHA</span><strong class="mono">{{ compactHash(summary?.checkpoint?.pair_manifest_sha256) }}</strong></div>
          </div>
        </div>
      </n-card>

      <n-card class="work-card" :bordered="false">
        <template #header>
          <div class="section-title"><span>需要关注</span><small>由观测事实归纳，不自动执行修复</small></div>
        </template>
        <div class="alert-list">
          <div
            v-for="alert in summary?.alerts ?? []"
            :key="alert.code"
            class="alert-row"
            :class="`alert-row--${alert.severity}`"
          >
            <span class="alert-row__dot" />
            <div><strong>{{ alert.title }}</strong><small>{{ alert.detail }}</small></div>
          </div>
        </div>
      </n-card>
    </div>

    <n-card class="work-card" :bordered="false">
      <template #header>
        <div class="section-title"><span>最新 Learner 信号</span><small>latest snapshot；历史趋势在“学习”页</small></div>
      </template>
      <div class="metric-ribbon">
        <div><span>Loss</span><strong>{{ decimal(summary?.latest_learner.loss) }}</strong></div>
        <div><span>Policy</span><strong>{{ decimal(summary?.latest_learner.policy_loss) }}</strong></div>
        <div><span>Value</span><strong>{{ decimal(summary?.latest_learner.value_loss) }}</strong></div>
        <div><span>Entropy</span><strong>{{ decimal(summary?.latest_learner.entropy) }}</strong></div>
        <div><span>Approx KL</span><strong>{{ decimal(summary?.latest_learner.approximate_kl) }}</strong></div>
        <div><span>Clip frac</span><strong>{{ percent(summary?.latest_learner.clip_fraction) }}</strong></div>
        <div><span>Grad norm</span><strong>{{ decimal(summary?.latest_learner.gradient_norm) }}</strong></div>
        <div><span>CUDA peak</span><strong>{{ gibibytes(summary?.latest_learner.cuda_peak_allocated_bytes) }}</strong></div>
      </div>
    </n-card>

    <n-alert v-if="!summary && !loading" type="info">选择一个可用 run 以查看当前状态。</n-alert>
  </section>
</template>

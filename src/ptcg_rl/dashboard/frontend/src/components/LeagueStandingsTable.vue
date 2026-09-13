<script setup lang="ts">
import { computed, ref } from 'vue'
import type { LeagueStanding } from '../types'

const props = defineProps<{
  rows: LeagueStanding[]
  view: 'bundle' | 'checkpoint' | 'deck' | 'candidate'
}>()

const query = ref('')
const showAll = ref(false)

const visibleRows = computed(() => {
  const needle = query.value.trim().toLowerCase()
  return props.rows.filter((row) => {
    if (props.view !== 'candidate' && !showAll.value && !row.rank_eligible) {
      return false
    }
    if (!needle) return true
    return [
      row.label,
      row.identity,
      row.controller_id,
      row.controller_label,
      row.deck_hash,
      row.deck_digest,
      row.deck_label,
      ...row.aliases,
    ].some((value) => value?.toLowerCase().includes(needle))
  })
})

function compactIdentity(identity: string): string {
  const [prefix, value] = identity.split(':', 2)
  return value ? `${prefix}:${value.slice(0, 12)}` : identity.slice(0, 16)
}

function stateLabel(row: LeagueStanding): string {
  if (!row.active) return '已停用'
  if (row.candidate_state === 'incumbent') return 'Incumbent'
  if (row.candidate_state === 'candidate') return '评测中'
  if (row.candidate_state === 'rejected') return '已剔除'
  if (row.candidate_state === 'anchor') return '公开锚点'
  return 'Active'
}

function stateClass(row: LeagueStanding): string {
  if (!row.active || row.candidate_state === 'rejected') return 'is-muted'
  if (row.candidate_state === 'candidate') return 'is-warning'
  return 'is-success'
}

function percent(value: number | null): string {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`
}
</script>

<template>
  <div class="standing-toolbar">
    <label class="standing-search">
      <span>筛选</span>
      <input v-model="query" placeholder="名称、旧编号或资产 SHA" />
    </label>
    <label v-if="view !== 'candidate'" class="standing-toggle">
      <input v-model="showAll" type="checkbox" />
      显示未入榜资产
    </label>
    <span class="standing-count">{{ visibleRows.length }} / {{ rows.length }}</span>
  </div>

  <div v-if="!visibleRows.length" class="standing-empty">
    {{ rows.length ? '没有符合筛选条件的条目。' : '暂无数据。' }}
  </div>
  <div v-else class="standing-scroll">
    <table class="standing-table">
      <thead>
        <tr>
          <th>#</th>
          <th>身份</th>
          <th>μ</th>
          <th>σ</th>
          <th>TrueSkill 保守分 (μ−3σ)</th>
          <th>百分位</th>
          <th>P(top20)</th>
          <th>W / D / L</th>
          <th>已决</th>
          <th>未决</th>
          <th>状态</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(row, index) in visibleRows" :key="row.identity">
          <td class="rank-cell">{{ row.rank_eligible ? index + 1 : '—' }}</td>
          <td class="identity-cell">
            <template v-if="view === 'bundle'">
              <strong>{{ row.controller_label ?? row.label }}</strong>
              <span>{{ row.deck_label ?? '卡组旧编号未记录' }}</span>
              <code>卡组旧编号 · {{ row.deck_hash ?? '未记录' }}</code>
              <code>Bundle · {{ compactIdentity(row.identity) }}</code>
            </template>
            <template v-else>
              <strong>{{ row.label }}</strong>
              <span v-if="view === 'deck'">{{ row.bundle_count }} 个组合</span>
              <span v-else-if="row.aliases.length">{{ row.aliases.join(' · ') }}</span>
              <code v-if="view === 'deck'">旧编号 · {{ row.deck_hash ?? '未记录' }}</code>
              <code v-else>{{ compactIdentity(row.identity) }}</code>
            </template>
          </td>
          <td>{{ row.mu.toFixed(3) }}</td>
          <td>{{ row.sigma.toFixed(3) }}</td>
          <td class="score-cell">{{ row.conservative.toFixed(3) }}</td>
          <td>{{ row.rank_eligible ? percent(row.percentile) : '—' }}</td>
          <td>{{ row.rank_eligible ? percent(row.p_top20) : '—' }}</td>
          <td>{{ row.wins }} / {{ row.draws }} / {{ row.losses }}</td>
          <td>{{ row.games }}</td>
          <td :class="{ 'unresolved-cell': row.unresolved > 0 }">{{ row.unresolved }}</td>
          <td><span class="state-pill" :class="stateClass(row)">{{ stateLabel(row) }}</span></td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.standing-toolbar { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
.standing-search { display: flex; align-items: center; gap: 8px; flex: 1; color: #94a3b8; }
.standing-search input { width: min(420px, 100%); border: 1px solid rgba(148, 163, 184, .18); border-radius: 8px; background: rgba(15, 23, 42, .72); color: #e2e8f0; padding: 8px 11px; outline: none; }
.standing-search input:focus { border-color: rgba(99, 230, 190, .62); }
.standing-toggle { display: flex; align-items: center; gap: 6px; color: #94a3b8; white-space: nowrap; }
.standing-count { color: #64748b; font-variant-numeric: tabular-nums; }
.standing-scroll { overflow: auto; max-height: 66vh; }
.standing-table { width: 100%; border-collapse: separate; border-spacing: 0; font-variant-numeric: tabular-nums; }
.standing-table th { position: sticky; top: 0; z-index: 1; background: #111827; color: #94a3b8; font-size: 11px; letter-spacing: .04em; text-transform: uppercase; }
.standing-table th, .standing-table td { padding: 10px 11px; border-bottom: 1px solid rgba(148, 163, 184, .12); text-align: right; white-space: nowrap; }
.standing-table tbody tr:hover { background: rgba(99, 230, 190, .035); }
.standing-table th:nth-child(2), .standing-table td:nth-child(2) { text-align: left; }
.rank-cell { color: #64748b; }
.identity-cell strong, .identity-cell span, .identity-cell code { display: block; max-width: 360px; overflow: hidden; text-overflow: ellipsis; }
.identity-cell span { margin-top: 2px; color: #94a3b8; }
.identity-cell code { margin-top: 3px; color: #64748b; font-size: 11px; }
.score-cell { color: #d1fae5; font-weight: 700; }
.unresolved-cell { color: #fbbf24; }
.state-pill { display: inline-flex; border-radius: 999px; padding: 3px 8px; background: rgba(59, 130, 246, .12); color: #93c5fd; font-size: 11px; }
.state-pill.is-success { background: rgba(16, 185, 129, .12); color: #6ee7b7; }
.state-pill.is-warning { background: rgba(245, 158, 11, .12); color: #fcd34d; }
.state-pill.is-muted { background: rgba(100, 116, 139, .16); color: #94a3b8; }
.standing-empty { padding: 36px 16px; color: #64748b; text-align: center; }
@media (max-width: 760px) { .standing-toolbar { align-items: stretch; flex-direction: column; } }
</style>

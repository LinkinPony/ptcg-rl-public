<script setup lang="ts">
import type { WorkbenchView } from '../types'

defineProps<{ modelValue: WorkbenchView }>()
const emit = defineEmits<{ 'update:modelValue': [value: WorkbenchView] }>()

const items: { value: WorkbenchView; mark: string; label: string; hint: string }[] = [
  { value: 'now', mark: '●', label: '现在', hint: '巡检与异常' },
  { value: 'learning', mark: '↗', label: '学习', hint: 'PPO 与吞吐' },
  { value: 'allocation', mark: '◎', label: '对手分配', hint: '目标、证据与债务' },
  { value: 'decks', mark: '◇', label: '卡组选择', hint: '比较与最终确认' },
  { value: 'compare', mark: '≋', label: '对比', hint: 'Run / checkpoint' },
  { value: 'tasks', mark: '▷', label: '任务', hint: '评测与校验' },
  { value: 'league', mark: '♜', label: '持续联赛', hint: 'TrueSkill 常驻层级' },
]
</script>

<template>
  <aside class="workbench-nav">
    <div class="nav-brand">
      <div class="nav-brand__mark">RL</div>
      <div>
        <strong>PTCG Workbench</strong>
        <small>TRAINING DECISION SYSTEM</small>
      </div>
    </div>
    <nav>
      <button
        v-for="item in items"
        :key="item.value"
        class="nav-item"
        :class="{ active: modelValue === item.value }"
        type="button"
        @click="emit('update:modelValue', item.value)"
      >
        <span class="nav-item__mark">{{ item.mark }}</span>
        <span>
          <strong>{{ item.label }}</strong>
          <small>{{ item.hint }}</small>
        </span>
      </button>
    </nav>
    <div class="nav-footer">
      <span class="nav-footer__dot" />
      localhost control plane
    </div>
  </aside>
</template>

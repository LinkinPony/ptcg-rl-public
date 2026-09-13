import { percent } from './format'
import type {
  CheckpointInfo,
  TrainingDeckStanding,
  TrainingDeckStrengthPayload,
  TrainingEvidenceRange,
} from './types'

export interface TrainingDeckDecisionExportInput {
  format: 'json' | 'markdown'
  runId: string
  checkpoint: CheckpointInfo
  evidence: TrainingDeckStrengthPayload
  selectedRanges: TrainingEvidenceRange[]
  finalDeckLabel: string
}

export function exportTrainingDeckDecision(
  input: TrainingDeckDecisionExportInput,
) {
  const payload = buildTrainingDeckDecision(input)
  if (!payload) return
  const text = input.format === 'json'
    ? `${JSON.stringify(payload, null, 2)}\n`
    : renderMarkdown(payload)
  download(
    text,
    `training-deck-decision-v${input.checkpoint.version}.${
      input.format === 'json' ? 'json' : 'md'
    }`,
    input.format === 'json' ? 'application/json' : 'text/markdown',
  )
}

export function buildTrainingDeckDecision(
  input: TrainingDeckDecisionExportInput,
) {
  const finalByRange = standingsFor(input, input.finalDeckLabel)
  const route = Object.values(finalByRange)[0]?.route_compatible
  if (!Object.keys(finalByRange).length || route === false) return null
  return {
    schema: 'ptcg_observed_training_deck_decision_v2',
    exported_at_utc: new Date().toISOString(),
    run_id: input.runId,
    checkpoint: {
      version: input.checkpoint.version,
      pair_manifest_sha256: input.checkpoint.pair_manifest_sha256,
      policy_sha256: input.checkpoint.policy_sha256,
      learner_state_sha256: input.checkpoint.learner_state_sha256,
      model_config_fingerprint: input.checkpoint.model_config_fingerprint,
      training_roster_fingerprint:
        input.checkpoint.training_roster_fingerprint,
      exact_registry_fingerprint:
        input.checkpoint.exact_registry_fingerprint,
      active_exact_deck_digests:
        input.checkpoint.active_exact_deck_digests,
    },
    semantics: input.evidence.semantics,
    source_warning: input.evidence.source_warning,
    selected_ranges: input.selectedRanges,
    range_metadata: Object.fromEntries(
      input.selectedRanges.map((range) => [
        range,
        input.evidence.ranges[range].metadata,
      ]),
    ),
    final_deck: {
      deck_label: input.finalDeckLabel,
      deck_hash: Object.values(finalByRange)[0]?.deck_hash ?? input.finalDeckLabel,
      deck_digest: Object.values(finalByRange)[0]?.deck_digest ?? null,
      family_id: Object.values(finalByRange)[0]?.family_id ?? null,
      family_display_name:
        Object.values(finalByRange)[0]?.family_display_name ?? null,
      evidence: finalByRange,
    },
  }
}

function standingsFor(
  input: TrainingDeckDecisionExportInput,
  deckLabel: string,
) {
  const output: Partial<Record<TrainingEvidenceRange, TrainingDeckStanding>> = {}
  for (const range of input.selectedRanges) {
    const standing = input.evidence.ranges[range].standings.find(
      (item) => item.deck_label === deckLabel,
    )
    if (standing) output[range] = standing
  }
  return output
}

interface DecisionPayload {
  exported_at_utc: string
  run_id: string
  checkpoint: {
    version: number
    pair_manifest_sha256: string
    policy_sha256: string
    learner_state_sha256: string
    model_config_fingerprint: string | null
    training_roster_fingerprint: string | null
    exact_registry_fingerprint: string | null
    active_exact_deck_digests: string[]
  }
  source_warning: string
  selected_ranges: TrainingEvidenceRange[]
  final_deck: {
    deck_label: string
    deck_hash: string
    deck_digest: string | null
    family_id: string | null
    family_display_name: string | null
    evidence: Partial<Record<TrainingEvidenceRange, TrainingDeckStanding>>
  }
}

function renderMarkdown(payload: DecisionPayload) {
  const first = Object.values(payload.final_deck.evidence)[0]
  const rangeRows = payload.selected_ranges.map((range) => {
    const row = payload.final_deck.evidence[range]
    return row?.posterior.evidence_state === 'ready'
      ? `| ${range} | #${row.rank} | ${percent(row.posterior.posterior_mean, 2)} | ${row.posterior.observed.games} | ${row.posterior.observed.wins}-${row.posterior.observed.draws}-${row.posterior.observed.losses} |`
      : `| ${range} | — | — | 0 | — |`
  }).join('\n')
  return `# 训练观测卡组选择决定

- Run: ${payload.run_id}
- Checkpoint: v${payload.checkpoint.version}
- Pair manifest: ${payload.checkpoint.pair_manifest_sha256}
- Policy: ${payload.checkpoint.policy_sha256}
- Learner state: ${payload.checkpoint.learner_state_sha256}
- Model config: ${payload.checkpoint.model_config_fingerprint ?? 'unknown'}
- Training roster: ${payload.checkpoint.training_roster_fingerprint ?? 'unknown'}
- Exact registry: ${payload.checkpoint.exact_registry_fingerprint ?? 'unknown'}
- Exported at: ${payload.exported_at_utc}
- Evidence: ${payload.source_warning}

## 最终卡组

**${first?.display_name ?? payload.final_deck.deck_label}**（${first?.deck_hash ?? 'unknown'}）

- 旧编号: ${payload.final_deck.deck_hash}
- Family: ${payload.final_deck.family_display_name ?? 'unclassified'}

| 范围 | 排名 | 后验得分 | 局数 | W-D-L |
|---|---:|---:|---:|---:|
${rangeRows}

此文件只记录本地决策；未创建 submission profile，也未上传。
`
}

function download(text: string, filename: string, mediaType: string) {
  const blob = new Blob([text], { type: `${mediaType};charset=utf-8` })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

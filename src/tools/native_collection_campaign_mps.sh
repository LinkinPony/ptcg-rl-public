#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  src/tools/native_collection_campaign_mps.sh <plan.json> <device-index>

Run one campaign device shard under its own CUDA MPS control plane. The plan's
physical device binding is exported as CUDA_VISIBLE_DEVICES. Launch this
foreground command inside a deterministic tmux session after training stops.

Environment:
  PTCG_RL_EVAL_MPS_RUN_ID  Optional short MPS run ID.
  PTCG_RL_MPS_ROOT         Optional MPS root; defaults to <repo>/tmp/mps.
EOF
}

if [[ $# -ne 2 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && exit 0
  exit 64
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
plan_path="$1"
if [[ "${plan_path}" != /* ]]; then
  plan_path="${repo_root}/${plan_path}"
fi
device_index="$2"
device_binding="$(python - "${plan_path}" "${device_index}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    plan = json.load(handle)
print(plan["device_bindings"][int(sys.argv[2])])
PY
)"

export CUDA_VISIBLE_DEVICES="${device_binding}"
export PTCG_RL_CAMPAIGN_DEVICE_BINDING="${device_binding}"
# The daemon consumes the physical binding and exposes it as client ordinal 0.
export PTCG_RL_MPS_CLIENT_VISIBLE_DEVICES=0
export PTCG_RL_MPS_ROOT="${PTCG_RL_MPS_ROOT:-${repo_root}/tmp/mps}"
export PTCG_RL_MPS_RUN_ID="${PTCG_RL_EVAL_MPS_RUN_ID:-campaign_d${device_index}_$$}"
export PYTHONPATH="${repo_root}/data/sample_submission:${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${repo_root}"
exec "${repo_root}/src/tools/rl_train_mps.sh" \
  python "${repo_root}/src/tools/run_native_collection_campaign.py" \
  "${plan_path}" --device-index "${device_index}" --execute

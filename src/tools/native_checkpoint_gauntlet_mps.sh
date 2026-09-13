#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=0 src/tools/native_checkpoint_gauntlet_mps.sh \
    <config-name> [Hydra overrides ...]

Run the configured persistent native worker replicas under one CUDA MPS
control plane.  Launch this foreground command inside a named tmux session for
a long campaign.  MPS sockets are kept under the repository tmp/mps directory
instead of the potentially overlong evaluation output path.

Environment:
  PTCG_RL_EVAL_MPS_RUN_ID  Optional short MPS directory name.
  PTCG_RL_MPS_ROOT         Optional short MPS root; defaults to <repo>/tmp/mps.
EOF
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  [[ $# -eq 0 ]] && exit 64
  exit 0
fi

config_name="$1"
shift
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
safe_config_name="${config_name//[^A-Za-z0-9_.-]/_}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# A one-GPU daemon exposes its selected physical device as client ordinal 0.
export PTCG_RL_MPS_CLIENT_VISIBLE_DEVICES="${PTCG_RL_MPS_CLIENT_VISIBLE_DEVICES:-0}"
# Replica workers now divide the process CPU affinity before loading models.
# Preserve explicit operator limits, but do not manufacture a one-thread limit
# that would override that host-aware allocation.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PTCG_RL_MPS_ROOT="${PTCG_RL_MPS_ROOT:-${repo_root}/tmp/mps}"
export PTCG_RL_MPS_RUN_ID="${PTCG_RL_EVAL_MPS_RUN_ID:-${safe_config_name}_$$}"
export PYTHONPATH="${repo_root}/data/sample_submission:${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${repo_root}"
exec "${repo_root}/src/tools/rl_train_mps.sh" \
  python "${repo_root}/src/tools/native_checkpoint_gauntlet.py" \
  --config-name "${config_name}" "$@"

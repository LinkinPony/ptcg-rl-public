#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  PYTHONPATH=data/sample_submission:src src/tools/rl_train_mps.sh \
    python src/tools/rl_train.py --config-name rl/train/versions/formal_eightdeck ...

Environment:
  PTCG_RL_MPS_ROOT      Directory for per-run MPS pipe/log folders.
                        Defaults to outputs/rl/mps.
  PTCG_RL_MPS_RUN_ID    Optional stable subdirectory name for this MPS daemon.
  PTCG_RL_MPS_CLIENT_VISIBLE_DEVICES
                        Optional logical device list for the wrapped client.
                        Use 0 for a daemon bound to one nonzero physical GPU.

The wrapped command inherits CUDA_MPS_PIPE_DIRECTORY and CUDA_MPS_LOG_DIRECTORY.
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is left unchanged. CUDA_VISIBLE_DEVICES is
changed only when PTCG_RL_MPS_CLIENT_VISIBLE_DEVICES is explicit: MPS remaps
the daemon's physical device selection to logical ordinals starting at zero.
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 64
fi
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "nvidia-cuda-mps-control was not found on PATH" >&2
  exit 127
fi

mps_root="${PTCG_RL_MPS_ROOT:-outputs/rl/mps}"
mps_run_id="${PTCG_RL_MPS_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
mkdir -p "${mps_root}"
mps_root_abs="$(cd "${mps_root}" && pwd -P)"
mps_dir="${mps_root_abs}/${mps_run_id}"
pipe_dir="${mps_dir}/pipe"
log_dir="${mps_dir}/log"
mkdir -p "${pipe_dir}" "${log_dir}"

# CUDA MPS uses AF_UNIX endpoints.  Its longest endpoint is
# control_privileged, not control.  The NVIDIA daemon rejects a 107-byte
# control_privileged path before emitting a useful foreground error, so retain
# one byte of headroom below that empirically observed boundary.
control_socket="${pipe_dir}/control_privileged"
if (( ${#control_socket} >= 107 )); then
  echo "CUDA MPS control socket path is too long (${#control_socket} bytes): ${control_socket}" >&2
  echo "Set PTCG_RL_MPS_ROOT to a short path such as tmp/mps." >&2
  exit 64
fi

mps_started=0
child_pid=""

stop_mps() {
  if [[ "${mps_started}" != "1" ]]; then
    return
  fi
  printf 'quit\n' | env CUDA_MPS_PIPE_DIRECTORY="${pipe_dir}" \
    nvidia-cuda-mps-control >/dev/null 2>&1 || true
  mps_started=0
}

cleanup() {
  local status=$?
  stop_mps
  return "${status}"
}

terminate() {
  local signal="$1"
  if [[ -n "${child_pid}" ]]; then
    kill "-${signal}" "${child_pid}" >/dev/null 2>&1 || true
    wait "${child_pid}" >/dev/null 2>&1 || true
  fi
  stop_mps
  case "${signal}" in
    INT) exit 130 ;;
    TERM) exit 143 ;;
    *) exit 1 ;;
  esac
}

trap cleanup EXIT
trap 'terminate INT' INT
trap 'terminate TERM' TERM

echo "Starting CUDA MPS daemon with pipe ${pipe_dir}" >&2
env CUDA_MPS_PIPE_DIRECTORY="${pipe_dir}" CUDA_MPS_LOG_DIRECTORY="${log_dir}" \
  nvidia-cuda-mps-control -d
mps_started=1

# The control daemon creates its pipe asynchronously.  Launching the first CUDA
# client before that endpoint answers is a real race on the H200 driver and can
# make an otherwise valid training command exit immediately.  Probe the new,
# run-local control endpoint before handing CUDA initialization to the child.
mps_ready=0
for _ in {1..100}; do
  if printf 'get_server_list\n' | env CUDA_MPS_PIPE_DIRECTORY="${pipe_dir}" \
    nvidia-cuda-mps-control >/dev/null 2>&1; then
    mps_ready=1
    break
  fi
  sleep 0.1
done
if [[ "${mps_ready}" != "1" ]]; then
  echo "CUDA MPS daemon did not become ready at ${pipe_dir}" >&2
  exit 1
fi

(
  export CUDA_MPS_PIPE_DIRECTORY="${pipe_dir}"
  export CUDA_MPS_LOG_DIRECTORY="${log_dir}"
  if [[ -n "${PTCG_RL_MPS_CLIENT_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${PTCG_RL_MPS_CLIENT_VISIBLE_DEVICES}"
  fi
  exec "$@"
) &
child_pid=$!
wait "${child_pid}"

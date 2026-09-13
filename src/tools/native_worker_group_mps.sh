#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: native_worker_group_mps.sh PIPE_DIR LOG_DIR READY_PATH" >&2
  exit 64
fi
if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "nvidia-cuda-mps-control was not found on PATH" >&2
  exit 127
fi

pipe_dir="$1"
log_dir="$2"
ready_path="$3"
mkdir -p "${pipe_dir}" "${log_dir}" "$(dirname "${ready_path}")"
rm -f "${ready_path}"
mps_started=0

cleanup() {
  local status=$?
  trap - EXIT
  rm -f "${ready_path}"
  if [[ "${mps_started}" == "1" ]]; then
    printf 'quit\n' | env CUDA_MPS_PIPE_DIRECTORY="${pipe_dir}" \
      nvidia-cuda-mps-control >/dev/null 2>&1 || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

env CUDA_MPS_PIPE_DIRECTORY="${pipe_dir}" CUDA_MPS_LOG_DIRECTORY="${log_dir}" \
  nvidia-cuda-mps-control -d
mps_started=1

for _ in {1..200}; do
  if printf 'get_server_list\n' | env CUDA_MPS_PIPE_DIRECTORY="${pipe_dir}" \
    nvidia-cuda-mps-control >/dev/null 2>&1; then
    touch "${ready_path}"
    while true; do
      sleep 5
    done
  fi
  sleep 0.1
done

echo "CUDA MPS daemon did not become ready at ${pipe_dir}" >&2
exit 1

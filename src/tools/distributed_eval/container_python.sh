#!/bin/sh
set -eu

: "${PTCG_EVAL_IMAGE:?PTCG_EVAL_IMAGE must name an immutable local image ID}"
: "${PTCG_EVAL_REMOTE_ROOT:?PTCG_EVAL_REMOTE_ROOT must be an absolute path}"

case "$PTCG_EVAL_REMOTE_ROOT" in
  /*) ;;
  *) echo "PTCG_EVAL_REMOTE_ROOT must be absolute" >&2; exit 2 ;;
esac

resolved_image=$(
  docker image inspect --format '{{.Id}}' "$PTCG_EVAL_IMAGE"
)
mkdir -p "$PTCG_EVAL_REMOTE_ROOT/runs"

if [ -n "${PTCG_EVAL_CPUSET:-}" ] && [ -n "${PTCG_EVAL_GPU_INDEX:-}" ]; then
  exec docker run --rm --init --network none \
    --mount "type=bind,src=$PTCG_EVAL_REMOTE_ROOT,dst=$PTCG_EVAL_REMOTE_ROOT,readonly" \
    --mount "type=bind,src=$PTCG_EVAL_REMOTE_ROOT/runs,dst=$PTCG_EVAL_REMOTE_ROOT/runs" \
    --workdir "$PWD" \
    --cpuset-cpus "$PTCG_EVAL_CPUSET" \
    --gpus "device=$PTCG_EVAL_GPU_INDEX" \
    --env PYTHONPATH \
    --env CUDA_VISIBLE_DEVICES \
    --env OMP_NUM_THREADS \
    --env MKL_NUM_THREADS \
    --env OPENBLAS_NUM_THREADS \
    --env NUMEXPR_NUM_THREADS \
    --env "PTCG_EVAL_LOCAL_IMAGE_ID=$resolved_image" \
    "$PTCG_EVAL_IMAGE" "$@"
fi

if [ -n "${PTCG_EVAL_CPUSET:-}" ]; then
  exec docker run --rm --init --network none \
    --mount "type=bind,src=$PTCG_EVAL_REMOTE_ROOT,dst=$PTCG_EVAL_REMOTE_ROOT,readonly" \
    --mount "type=bind,src=$PTCG_EVAL_REMOTE_ROOT/runs,dst=$PTCG_EVAL_REMOTE_ROOT/runs" \
    --workdir "$PWD" \
    --cpuset-cpus "$PTCG_EVAL_CPUSET" \
    --env PYTHONPATH \
    --env CUDA_VISIBLE_DEVICES \
    --env OMP_NUM_THREADS \
    --env MKL_NUM_THREADS \
    --env OPENBLAS_NUM_THREADS \
    --env NUMEXPR_NUM_THREADS \
    --env "PTCG_EVAL_LOCAL_IMAGE_ID=$resolved_image" \
    "$PTCG_EVAL_IMAGE" "$@"
fi

if [ -n "${PTCG_EVAL_GPU_INDEX:-}" ]; then
  exec docker run --rm --init --network none \
    --mount "type=bind,src=$PTCG_EVAL_REMOTE_ROOT,dst=$PTCG_EVAL_REMOTE_ROOT,readonly" \
    --mount "type=bind,src=$PTCG_EVAL_REMOTE_ROOT/runs,dst=$PTCG_EVAL_REMOTE_ROOT/runs" \
    --workdir "$PWD" \
    --gpus "device=$PTCG_EVAL_GPU_INDEX" \
    --env PYTHONPATH \
    --env CUDA_VISIBLE_DEVICES \
    --env OMP_NUM_THREADS \
    --env MKL_NUM_THREADS \
    --env OPENBLAS_NUM_THREADS \
    --env NUMEXPR_NUM_THREADS \
    --env "PTCG_EVAL_LOCAL_IMAGE_ID=$resolved_image" \
    "$PTCG_EVAL_IMAGE" "$@"
fi

exec docker run --rm --init --network none \
  --mount "type=bind,src=$PTCG_EVAL_REMOTE_ROOT,dst=$PTCG_EVAL_REMOTE_ROOT,readonly" \
  --mount "type=bind,src=$PTCG_EVAL_REMOTE_ROOT/runs,dst=$PTCG_EVAL_REMOTE_ROOT/runs" \
  --workdir "$PWD" \
  --env PYTHONPATH \
  --env CUDA_VISIBLE_DEVICES \
  --env OMP_NUM_THREADS \
  --env MKL_NUM_THREADS \
  --env OPENBLAS_NUM_THREADS \
  --env NUMEXPR_NUM_THREADS \
  --env "PTCG_EVAL_LOCAL_IMAGE_ID=$resolved_image" \
  "$PTCG_EVAL_IMAGE" "$@"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Teacher stays in the official vLLM image; no local vLLM checkout is needed.
# Existing containers are never stopped or deleted by this script.
exec docker run -d \
  --name "${TEACHER_CONTAINER_NAME:-38}" \
  --gpus '"device=3"' \
  --ipc=host \
  -p 127.0.0.1:8011:8000 \
  -v "${TEACHER_MODEL_PATH:-/xx}:/model:ro" \
  "${VLLM_IMAGE:-vllm/vllm-openai:v0.32.0}" \
  --model /model \
  --served-model-name xx \
  --dtype bfloat16 \
  --max-num-seqs 8 \
  --max-model-len "${TEACHER_MAX_MODEL_LEN:-32768}" \
  --limit-mm-per-prompt '{"image":16}' \
  --no-enable-prefix-caching \
  --enable-scale-out \
  --mm-processor-kwargs '{"do_resize":false}' \
  "$@"

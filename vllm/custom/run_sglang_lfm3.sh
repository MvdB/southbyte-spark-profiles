#!/usr/bin/env bash
# LFM3 (LiquidAI) über SGLang auf dem DGX Spark — Port 8018, für Testläufe im --endpoint-Modus.
#
# vLLM kennt LFM3 nicht (weder v0.32 noch der Hauptzweig, Stand 29.10.2026); --model-impl
# transformers scheitert an den Conv-Schichten (Liquid4All/docs#157). SGLang hat die Architektur
# in einem offenen PR (sgl-project/sglang#46858), das Dev-Image ist für arm64 gebaut.
# Läuft für Lfm3ForCausalLM, Lfm3VlForConditionalGeneration und Lfm3OmniForConditionalGeneration
# (Omni über SGLang: Text/Bild/Ton rein, nur Text raus).
#
#   MODEL_DIR=LiquidAI--LFM3-1.7B-NVFP4 ./run_sglang_lfm3.sh
#   MODEL_DIR=LiquidAI--LFM3-VL-2B-NVFP4 SERVED_NAME=LFM3-VL-2B-Vision ./run_sglang_lfm3.sh
set -euo pipefail

HF_MODELS_DIR="${HF_MODELS_DIR:-$HOME/hf_models}"
MODEL_DIR="${MODEL_DIR:?MODEL_DIR fehlt, z.B. LiquidAI--LFM3-1.7B-NVFP4}"
SERVED_NAME="${SERVED_NAME:-${MODEL_DIR#*--}}"
CONTAINER_NAME="${CONTAINER_NAME:-sglang-lfm3}"
HOST_PORT="${HOST_PORT:-8018}"
IMAGE="${IMAGE:-lmsysorg/sglang:dev-cu13-lfm3}"
MEM_FRACTION="${MEM_FRACTION:-0.2}"      # klein halten: GPU-OOM haengt den GB10 auf
CONTEXT="${CONTEXT:-32768}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

docker run -d --name "${CONTAINER_NAME}" --gpus all --ipc=host -p "${HOST_PORT}:30000" \
  -e HF_HUB_OFFLINE=1 -v "${HF_MODELS_DIR}:/models:ro" "${IMAGE}" \
  sglang serve --model-path "/models/${MODEL_DIR}" --trust-remote-code \
    --reasoning-parser auto --tool-call-parser auto --served-model-name "${SERVED_NAME}" \
    --context-length "${CONTEXT}" --mem-fraction-static "${MEM_FRACTION}" --host 0.0.0.0 --port 30000

echo "Gestartet: http://0.0.0.0:${HOST_PORT}  (${MODEL_DIR} als ${SERVED_NAME})"

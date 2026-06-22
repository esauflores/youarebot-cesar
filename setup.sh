#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="$(dirname "$0")/models"
MODEL_REPO="Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_URL="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"

mkdir -p "$MODELS_DIR"

if [ -f "$MODELS_DIR/$MODEL_FILE" ]; then
    echo "Model already downloaded: $MODELS_DIR/$MODEL_FILE"
    exit 0
fi

echo "Downloading $MODEL_FILE"
curl -L -C - -o "$MODELS_DIR/$MODEL_FILE" "$MODEL_URL"

echo "Done. Model at $MODELS_DIR/$MODEL_FILE"

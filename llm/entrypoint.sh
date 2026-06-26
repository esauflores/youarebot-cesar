#!/bin/sh
MODEL="Qwen3.5-0.8B-Q4_K_M.gguf"
MODEL_PATH="/models/$MODEL"
MODEL_URL="https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/$MODEL"

mkdir -p /models
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading model..."
    wget -q -O "$MODEL_PATH" "$MODEL_URL"
    echo "Done."
fi

exec llama-server -m "$MODEL_PATH" --host 0.0.0.0 --port 8080 -c 4096

# setup — model + data downloads

# download the GGUF model (one-time)
model:
    #!/usr/bin/env bash
    set -euo pipefail
    MODEL_FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"
    MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/${MODEL_FILE}"
    mkdir -p models
    if [ ! -f "models/$MODEL_FILE" ]; then
        echo "Downloading $MODEL_FILE (491MB)..."
        curl -L -C - -o "models/$MODEL_FILE" "$MODEL_URL"
    else
        echo "Model exists: models/$MODEL_FILE"
    fi

# download Kaggle dataset (one-time)
data:
    uv run python3 scripts/download_data.py

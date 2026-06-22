#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"
MODELS_DIR="$SCRIPT_DIR/models"
DATA_DIR="$SCRIPT_DIR/data"
MODEL_REPO="Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_URL="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"
COMPETITION="you-are-bot-2"

# --- LLM model ---
mkdir -p "$MODELS_DIR"

if [ ! -f "$MODELS_DIR/$MODEL_FILE" ]; then
    echo "Downloading $MODEL_FILE (491MB)..."
    curl -L -C - -o "$MODELS_DIR/$MODEL_FILE" "$MODEL_URL"
    echo "Done. Model at $MODELS_DIR/$MODEL_FILE"
else
    echo "Model already downloaded: $MODELS_DIR/$MODEL_FILE"
fi

# --- Kaggle dataset ---
mkdir -p "$DATA_DIR"

if [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    cd "$SCRIPT_DIR"
    echo "Downloading Kaggle dataset '$COMPETITION'..."
    poetry run python -c "
import kagglehub, shutil, os
path = kagglehub.competition_download('$COMPETITION')
for f in os.listdir(path):
    src = os.path.join(path, f)
    dst = os.path.join('$DATA_DIR', f)
    shutil.copy2(src, dst)
    print(f'  {f} ({os.path.getsize(src):,} bytes)')
print(f'Dataset copied to $DATA_DIR')
"
else
    echo "Dataset already downloaded: $DATA_DIR"
fi
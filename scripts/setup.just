# setup — model + data downloads

# download the GGUF model (one-time)
model:
    mkdir -p models && curl -L -C - -o models/Qwen3.5-0.8B-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf

# download Kaggle dataset (one-time)
data:
    uv run python3 scripts/download_data.py

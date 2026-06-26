# YouAreBot v2

Microservice architecture for bot detection in dialog participants.

## Architecture

```
                    ┌──────────────┐
                    │ orchestrator │  :8672  (public)
                    │  FastAPI     │
                    └──┬────────┬──┘
              /predict │        │ /get_message
                       ▼        ▼
              ┌──────────┐  ┌──────────┐
              │classifier│  │   llm    │
              │ FastAPI   │  │llama.cpp │
              │  :8000    │  │  :8080   │
              └─────┬─────┘  └──────────┘
                    │
                    ▼
              ┌──────────┐
              │  mlflow  │
              │ tracking │
              │  :5000   │
              └──────────┘
```

| Service | Port | Role |
|---------|------|------|
| `orchestrator` | 8672 | Public gateway — routes to internal services |
| `classifier` | 8000 | Bot detection — ONNX model baked into image |
| `llm` | 8080 | llama.cpp — Qwen3.5-0.8B chat replies |
| `streamlit` | 8501 | Chat UI with live bot probability |
| `postgres` | 5432 | Database for prediction logs |

## Quick Start

```bash
# 1. Download the LLM model (one-time)
just model

# 2. Export trained model to ONNX (one-time)
uv run training/export_onnx.py

# 3. Start all services
docker compose up --build

# 4. Test endpoints
just test-predict
just test-message
```

## Endpoints

### Orchestrator (port 8672)

**`POST /predict`** — Classify a message as bot or human.

```bash
curl -X POST http://localhost:8672/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello how are you today?"}'
```

Response: `{"is_bot_probability": 0.34}`

**`POST /get_message`** — Get a chat reply from the LLM.

```bash
curl -X POST http://localhost:8672/get_message \
  -H 'Content-Type: application/json' \
  -d '{"dialog_id":"00000000-0000-0000-0000-000000000001","last_msg_text":"Hello, how are you?","last_message_id":"00000000-0000-0000-0000-000000000002"}'
```

Response: `{"new_msg_text": "I'm doing well, thanks for asking!", "dialog_id": "..."}`

**`GET /health`**

### Classifier (port 8000)

**`POST /predict`** — Direct access (same payload as orchestrator).

## Training Experiments

See `training/EXPERIMENTS.md` for the full experiment list.

Run `build_features.py` first, then `uv run training/<script>.py`.

### MLflow Logging

Every experiment logs the following to MLflow:

| What | Method | Example |
|------|--------|---------|
| **Model name** | `mlflow.log_params` | `"model": "bert-base-uncased"` |
| **Approach** | `mlflow.log_params` | `"approach": "LoRA-finetune"` |
| **LoRA / DoRA config** | `mlflow.log_params` | `"lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.1, "lora_target_modules": [...], "lora_modules_to_save": [...]` |
| **Training params** | `mlflow.log_params` | `"learning_rate": 2e-5, "batch_size": 32, "epochs": 4, "val_split": 0.1` |
| **Validation metrics** (per-epoch) | `report_to="mlflow"` in Trainer | `eval_log_loss`, `eval_accuracy`, `eval_f1` |
| **Validation metrics** (final) | `mlflow.log_metrics` | `log_loss`, `accuracy`, `f1` |
| **Model artifacts** | `mlflow.log_artifacts` | LoRA adapter / full checkpoint weights |
| **Notes** | `mlflow.set_tag` | What changed in this run (model, features, approach) |

Zero-shot experiments (00, 01) log model name, approach, threshold, and final metrics only (no training).

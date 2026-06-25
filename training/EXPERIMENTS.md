# Experiments

Goal: minimize **log loss** on You-Are-Bot v2. All runs logged to `mlruns/` — view with `mlflow ui`.

| # | Script | Approach | Model | Features | LoRA | MLflow run name |
|---|--------|----------|-------|----------|------|-----------------|
| 0 | `00_zero_shot.py` | Zero-shot per-message | distilbert-mnli | none (raw JSON) | – | `zero-shot-distilbert` |
| 1 | `01_zero_shot_features.py` | Zero-shot dialog-level | distilbert-mnli | dialog context + numeric | – | `zero-shot-features` |
| 2 | `02_bert_full.py` | Full fine-tune | BERT-base | dialog context + numeric | – | `bert-full-finetune` |
| 3 | `03_bert_lora.py` | LoRA | BERT-base | dialog context + numeric | r=8, a=16 | `bert-lora-finetune` |
| 4 | `04_modernbert_lora.py` | LoRA | ModernBERT-base | dialog context + numeric | r=8, a=16 | `modernbert-lora-finetune` |
| 5 | `05_modernbert_full.py` | Full fine-tune | ModernBERT-base | dialog context + numeric | – | `modernbert-full-finetune` |

All experiments (except #0) run `build_features.py` first to generate `clean_train.csv` / `clean_test.csv`.
All log to MLflow: model name, approach, hyperparams, metrics, model artifacts, notes.

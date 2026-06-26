# Experiments

Goal: minimize **log loss** on You-Are-Bot v2. All runs logged to `mlruns/` — view with `mlflow ui`.

| # | Script | Approach | Model | Features | Adapter | MLflow run name |
|---|--------|----------|-------|----------|---------|-----------------|
| 0 | `00_zero_shot.py` | Zero-shot per-msg | distilbert-mnli | none (raw JSON) | – | `zero-shot-distilbert` |
| 1 | `01_zero_shot_features.py` | Zero-shot dialog | distilbert-mnli | numeric prefix | – | `zero-shot-features` |
| 2 | `02_bert_full.py` | Full fine-tune | BERT-base | numeric prefix | – | `bert-full-finetune` |
| 3 | `03_bert_lora.py` | LoRA | BERT-base | numeric prefix | r=8, a=16 | `bert-lora-finetune` |
| 4 | `04_modernbert_lora.py` | LoRA | ModernBERT-base | numeric prefix | r=8, a=16 | `modernbert-lora-finetune` |
| 5 | `05_modernbert_full.py` | Full fine-tune | ModernBERT-base | numeric prefix | – | `modernbert-full-finetune` |
| 6 | `06_modernbert_dora.py` | DoRA | ModernBERT-base | **none** | r=8, a=16 | `modernbert-dora-no-feats` |
| 7 | `07_modernbert_dora_feats.py` | DoRA | ModernBERT-base | numeric prefix | r=8, a=16 | `modernbert-dora-feats` |
| 8 | `08_modernbert_dora_pissa.py` | DoRA+PiSSA | ModernBERT-base | numeric prefix | r=8, a=16, pissa | `modernbert-dora-pissa-feats` |

**Key comparisons:** #6 vs #7 isolates whether numeric features help. #7 vs #8 tests PiSSA init benefit.

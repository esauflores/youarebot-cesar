# %% [markdown]
# Export Model to ONNX
# Loads best checkpoint, merges LoRA/DoRA, exports to classifier.onnx

# %%
import torch
import json
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from pathlib import Path
from sklearn.metrics import accuracy_score, log_loss
import numpy as np

MODEL_NAME = "answerdotai/ModernBERT-base"
CHECKPOINT = Path(__file__).parent / "checkpoints"
CLASSIFIER_DIR = Path(__file__).parent.parent / "classifier"
OUTPUT = CLASSIFIER_DIR / "classifier.onnx"
DATA = Path(__file__).parent / "data"

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Find best PEFT checkpoint by stored eval_log_loss
best_ckpt = None
best_loss = float("inf")
for p in CHECKPOINT.glob("checkpoint-*"):
    trainer_state = p / "trainer_state.json"
    if not (p / "adapter_config.json").exists() or not trainer_state.exists():
        continue
    try:
        state = json.loads(trainer_state.read_text())
        for entry in state.get("log_history", []):
            if "eval_log_loss" in entry and entry["eval_log_loss"] < best_loss:
                best_loss = entry["eval_log_loss"]
                best_ckpt = str(p)
    except Exception:
        continue

if not best_ckpt:
    raise FileNotFoundError(f"No valid PEFT checkpoint found in {CHECKPOINT}.")

print(f"Best checkpoint: {best_ckpt} (eval_log_loss={best_loss:.4f})")
ckpt = best_ckpt
model = PeftModel.from_pretrained(base_model, ckpt)
model = model.merge_and_unload()
model.eval()

# Quick quality check on 100 train samples
print("\nQuality check:")
with open(DATA / "train.json") as f:
    dialogs = json.load(f)
with open(DATA / "ytrain.csv") as f:
    rows = list(csv.DictReader(f))

def build_dialog(msgs, target_idx):
    tag = lambda m: "<self>" if m["participant_index"] == target_idx else "<other>"
    return " [SEP] ".join(f"{tag(m)} {m['text']}" for m in msgs)

test_samples = []
for r in rows[:100]:
    did, idx = r["dialog_id"], int(r["participant_index"])
    if did in dialogs:
        test_samples.append((build_dialog(dialogs[did], idx), int(r["is_bot"])))

texts = [s[0] for s in test_samples]
y_true = [s[1] for s in test_samples]
enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
with torch.no_grad():
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
y_pred = (probs > 0.5).astype(int)
acc = accuracy_score(y_true, y_pred)
ll = log_loss(y_true, probs)
print(f"  Accuracy: {acc:.4f}  Log Loss: {ll:.4f}  (100 samples)")

# %%
dummy = tokenizer("hello world", return_tensors="pt", padding=True, max_length=256)
torch.onnx.export(
    model,
    (dummy["input_ids"], dummy["attention_mask"]),
    OUTPUT,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch"},
    },
    opset_version=17,
)

print(f"Exported {OUTPUT} ({OUTPUT.stat().st_size / 1024 / 1024:.1f} MB)")
print("Ready for docker compose up --build")

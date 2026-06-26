# %% [markdown]
# Experiment 12: 3-Fold CV Ensemble (DoRA + tuned hypers)
# ModernBERT-base + DoRA, 3-fold CV — minimize log loss
# No build_features dependency — reads raw JSON/CSV

# %%
import csv
from pathlib import Path
import json
import tempfile, shutil
import torch
import numpy as np
import mlflow
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

DATA = Path(__file__).parent / "data"
MODEL_NAME = "answerdotai/ModernBERT-base"
BATCH_SIZE = 32
EPOCHS = 10
LR = 7e-4
N_FOLDS = 3
SEED = 42

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

mlflow.set_experiment("youarebot-v2")

# %% [markdown]
## 1. Load raw data

# %%
with open(DATA / "train.json") as f:
    dialogs = json.load(f)
with open(DATA / "ytrain.csv") as f:
    rows = list(csv.DictReader(f))

def build_dialog(msgs, target_idx):
    tag = lambda m: "<self>" if m["participant_index"] == target_idx else "<other>"
    return " [SEP] ".join(f"{tag(m)} {m['text']}" for m in msgs)

samples, labels = [], []
for r in rows:
    did, idx = r["dialog_id"], int(r["participant_index"])
    if did in dialogs:
        samples.append(build_dialog(dialogs[did], idx))
        labels.append(int(r["is_bot"]))

print(f"{len(samples)} participants — Bot: {sum(labels)}, Human: {len(labels)-sum(labels)}")

# %% [markdown]
## 2. Tokenize all data

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encodings = tokenizer(samples, truncation=True, padding=True, max_length=256)
print(f"Tokenized {len(encodings['input_ids'])} texts")

class BotDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item

# %% [markdown]
## 3. 3-Fold CV

# %%
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_probs = np.zeros(len(labels))
models = []

with mlflow.start_run(run_name="dora-3fold-cv"):
    mlflow.log_params({
        "model": MODEL_NAME,
        "approach": "DoRA-3fold-CV",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "n_folds": N_FOLDS,
        "warmup_ratio": 0.1,
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "label_smoothing": 0.05,
    })

    for fold, (train_idx, val_idx) in enumerate(skf.split(samples, labels)):
        print(f"\n=== Fold {fold+1}/{N_FOLDS} ===")

        train_enc = {k: [v[i] for i in train_idx] for k, v in encodings.items()}
        val_enc = {k: [v[i] for i in val_idx] for k, v in encodings.items()}
        train_ds = BotDataset(train_enc, [labels[i] for i in train_idx])
        val_ds = BotDataset(val_enc, [labels[i] for i in val_idx])

        base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            use_dora=True,
            target_modules=["Wqkv", "Wo"],
            modules_to_save=["classifier"],
        )
        model = get_peft_model(base_model, lora_config)

        args = TrainingArguments(
            output_dir=f"./checkpoints/cv_fold{fold}",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            learning_rate=LR,
            num_train_epochs=EPOCHS,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_log_loss",
            greater_is_better=False,
            fp16=True,
            report_to="none",
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            label_smoothing_factor=0.05,
            dataloader_num_workers=4,
        )

        def compute_metrics(eval_pred):
            logits, labs = eval_pred
            preds = logits.argmax(-1)
            probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
            return {
                "log_loss": log_loss(labs, probs),
                "accuracy": accuracy_score(labs, preds),
                "f1": f1_score(labs, preds),
            }

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer.train()

        val_preds = trainer.predict(val_ds)
        oof_probs[val_idx] = torch.nn.functional.softmax(
            torch.tensor(val_preds.predictions), dim=-1
        )[:, 1].numpy()
        models.append(trainer)

        fold_ll = log_loss([labels[i] for i in val_idx], oof_probs[val_idx])
        fold_acc = accuracy_score(labels[val_idx], oof_probs[val_idx] > 0.5)
        mlflow.log_metrics({f"fold{fold}_log_loss": fold_ll, f"fold{fold}_accuracy": fold_acc})
        print(f"Fold {fold+1} — log_loss: {fold_ll:.4f}, accuracy: {fold_acc:.4f}")

    oof_preds = (oof_probs > 0.5).astype(int)
    oof_ll = log_loss(labels, oof_probs)
    oof_acc = accuracy_score(labels, oof_preds)
    oof_f1 = f1_score(labels, oof_preds)

    mlflow.log_metrics({"oof_log_loss": oof_ll, "oof_accuracy": oof_acc, "oof_f1": oof_f1})
    print(f"\nOOF — log_loss: {oof_ll:.4f}, accuracy: {oof_acc:.4f}, f1: {oof_f1:.4f}")

# %% [markdown]
## 4. Generate submission (ensemble mean)

# %%
with open(DATA / "test.json") as f:
    test_dialogs = json.load(f)
with open(DATA / "ytest.csv") as f:
    test_rows = list(csv.DictReader(f))

test_texts, test_ids = [], []
for r in test_rows:
    did, idx = r["dialog_id"], int(r["participant_index"])
    if did in test_dialogs:
        test_texts.append(build_dialog(test_dialogs[did], idx))
        test_ids.append(r["ID"])

print(f"Tokenizing {len(test_texts)} test participants...")
test_enc = tokenizer(test_texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
test_ds = BotDataset(test_enc, [0] * len(test_texts))

all_probs = []
for fold, trainer in enumerate(models):
    trainer.compute_metrics = None
    preds = trainer.predict(test_ds)
    probs = torch.nn.functional.softmax(torch.tensor(preds.predictions), dim=-1)[:, 1].numpy()
    all_probs.append(probs)
    print(f"Fold {fold+1} done")

avg_probs = np.mean(all_probs, axis=0)

out = DATA / "submission.csv"
with open(out, "w") as f:
    w = csv.DictWriter(f, fieldnames=["ID", "is_bot"])
    w.writeheader()
    for id_, prob in zip(test_ids, avg_probs):
        w.writerow({"ID": id_, "is_bot": float(prob)})
print(f"Saved {out} ({len(test_ids)} rows, {N_FOLDS}-fold ensemble mean)")

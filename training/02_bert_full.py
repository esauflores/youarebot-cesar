# %% [markdown]
# Experiment: BERT Full Fine-Tune
# Full fine-tuning (no LoRA) — minimize log loss
# Run `build_features.py` first

# %%
import csv
from pathlib import Path
import tempfile, shutil
import torch
import mlflow
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import classification_report, accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

DATA = Path(__file__).parent / "data"
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 32
EPOCHS = 6
LR = 2e-5
VAL_SPLIT = 0.1

mlflow.set_experiment("youarebot-v2")

# %% [markdown]
## 1. Load clean data

# %%
samples, labels = [], []
with open(DATA / "clean_train.csv") as f:
    for r in csv.DictReader(f):
        samples.append(r["text_with_features"])
        labels.append(int(r["is_bot"]))

print(f"{len(samples)} participants — Bot: {sum(labels)}, Human: {len(labels)-sum(labels)}")

# %% [markdown]
## 2. Tokenize + dataset

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

X_tr, X_val, y_tr, y_val = train_test_split(
    samples, labels, test_size=VAL_SPLIT, stratify=labels, random_state=42
)

train_enc = tokenizer(X_tr, truncation=True, padding=True, max_length=512)
val_enc = tokenizer(X_val, truncation=True, padding=True, max_length=512)

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

train_ds = BotDataset(train_enc, y_tr)
val_ds = BotDataset(val_enc, y_val)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

# %% [markdown]
## 3. Full fine-tune

# %%
with mlflow.start_run(run_name="bert-full-finetune"):
    mlflow.log_params({
        "model": MODEL_NAME,
        "approach": "full-finetune",
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "val_split": VAL_SPLIT,
    })
    mlflow.set_tag("notes", "Full fine-tuning of BERT-base, dialog context + numeric features.")

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir="./checkpoints",
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

    best = trainer.state.best_metric or 0
    mlflow.log_metric("best_eval_log_loss", best)

    tmp = tempfile.mkdtemp()
    trainer.save_model(tmp)
    mlflow.log_artifacts(tmp, "model")
    shutil.rmtree(tmp)

    print(f"Best eval_log_loss: {best:.4f}")

# %% [markdown]
## 4. Validation

# %%
preds = trainer.predict(val_ds)
y_true = preds.label_ids
y_pred = preds.predictions.argmax(-1)
print(classification_report(y_true, y_pred, target_names=["human", "bot"]))
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

# %% [markdown]
## 5. Generate submission

# %%
test_texts, test_ids = [], []
with open(DATA / "clean_test.csv") as f:
    for r in csv.DictReader(f):
        test_texts.append(r["text_with_features"])
        test_ids.append(r["ID"])

print(f"Tokenizing {len(test_texts)} test participants...")
test_enc = tokenizer(test_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
test_ds = BotDataset(test_enc, [0] * len(test_texts))

trainer.compute_metrics = None
preds = trainer.predict(test_ds)
probs = torch.nn.functional.softmax(torch.tensor(preds.predictions), dim=-1)[:, 1].tolist()

out = DATA / "submission.csv"
with open(out, "w") as f:
    w = csv.DictWriter(f, fieldnames=["ID", "is_bot"])
    w.writeheader()
    for id_, prob in zip(test_ids, probs):
        w.writerow({"ID": id_, "is_bot": prob})
print(f"Saved {out} ({len(test_ids)} rows)")

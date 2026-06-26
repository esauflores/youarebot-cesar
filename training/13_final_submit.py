# %% [markdown]
# Experiment 13: Final Submission Model (with features)
# Best config (09) on 90/10 → 100% retrain — maximize training data
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
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import classification_report, accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

DATA = Path(__file__).parent / "data"
MODEL_NAME = "answerdotai/ModernBERT-base"
BATCH_SIZE = 32
EPOCHS = 10
LR = 7e-4
VAL_SPLIT = 0.1
EPOCHS = 10

# LoRA config
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

mlflow.set_experiment("youarebot-v2")

# %% [markdown]
## 1. Load clean data (with features)

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

train_enc = tokenizer(X_tr, truncation=True, padding=True, max_length=256)
val_enc = tokenizer(X_val, truncation=True, padding=True, max_length=256)

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
## 3. Phase 1: 90/10 to find best epoch

# %%
with mlflow.start_run(run_name="final-submission"):
    mlflow.log_params({
        "model": MODEL_NAME,
        "approach": "DoRA-final",
        "input": "dialog_text_with_numeric_features",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": ["Wqkv", "Wo"],
        "lora_modules_to_save": ["classifier"],
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "warmup_ratio": 0.1,
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "label_smoothing": 0.05,
    })
    mlflow.set_tag("notes", "Final submission: best config from 09, trained on 90/10 then 100% data with numeric features.")

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
    model.print_trainable_parameters()

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

    steps_per_epoch = len(train_ds) // BATCH_SIZE
    best_step = int(trainer.state.best_model_checkpoint.split("-")[-1]) if trainer.state.best_model_checkpoint else None
    best_epoch = best_step // steps_per_epoch if best_step else min(EPOCHS, 6)
    print(f"Best epoch: {best_epoch} (step {best_step} / {steps_per_epoch})")

    preds = trainer.predict(val_ds)
    val_probs = torch.nn.functional.softmax(torch.tensor(preds.predictions), dim=-1)[:, 1].numpy()
    val_preds = preds.predictions.argmax(-1)
    mlflow.log_metrics({
        "val_log_loss": log_loss(y_val, val_probs),
        "val_accuracy": accuracy_score(y_val, val_preds),
        "val_f1": f1_score(y_val, val_preds),
        "best_epoch": float(best_epoch),
    })
    print(f"Val: log_loss={log_loss(y_val, val_probs):.4f}, accuracy={accuracy_score(y_val, val_preds):.4f}")

    tmp = tempfile.mkdtemp()
    trainer.save_model(tmp)
    mlflow.log_artifacts(tmp, "model_90_10")
    shutil.rmtree(tmp)

# %% [markdown]
## 4. Phase 2: Retrain on 100% data (best epoch count)

# %%
all_enc = tokenizer(samples, truncation=True, padding=True, max_length=256)
all_ds = BotDataset(all_enc, labels)
print(f"All data: {len(all_ds)}")

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

final_args = TrainingArguments(
    output_dir="./checkpoints/final",
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    num_train_epochs=best_epoch,
    save_strategy="no",
    logging_strategy="epoch",
    fp16=True,
    report_to="mlflow",
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    label_smoothing_factor=0.05,
    dataloader_num_workers=4,
)

final_trainer = Trainer(model=model, args=final_args, train_dataset=all_ds)
final_trainer.train()

tmp = tempfile.mkdtemp()
final_trainer.save_model(tmp)
mlflow.log_artifacts(tmp, "model_final")
shutil.rmtree(tmp)

# %% [markdown]
## 5. Generate submission

# %%
test_texts, test_ids = [], []
with open(DATA / "clean_test.csv") as f:
    for r in csv.DictReader(f):
        test_texts.append(r["text_with_features"])
        test_ids.append(r["ID"])

print(f"Tokenizing {len(test_texts)} test participants...")
test_enc = tokenizer(test_texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
test_ds = BotDataset(test_enc, [0] * len(test_texts))

final_trainer.compute_metrics = None
preds = final_trainer.predict(test_ds)
probs = torch.nn.functional.softmax(torch.tensor(preds.predictions), dim=-1)[:, 1].tolist()

out = DATA / "submission.csv"
with open(out, "w") as f:
    w = csv.DictWriter(f, fieldnames=["ID", "is_bot"])
    w.writeheader()
    for id_, prob in zip(test_ids, probs):
        w.writerow({"ID": id_, "is_bot": prob})
print(f"Saved {out} ({len(test_ids)} rows)")

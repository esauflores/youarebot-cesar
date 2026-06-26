# %% [markdown]
# Experiment 10: Optuna Hyperparameter Search
# DoRA on ModernBERT-base, Optuna tunes LR, dropout, weight decay, label smoothing
# No build_features dependency — reads raw JSON/CSV

# %%
import csv
from pathlib import Path
import json
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
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
import optuna

DATA = Path(__file__).parent / "data"
MODEL_NAME = "answerdotai/ModernBERT-base"
BATCH_SIZE = 32
EPOCHS = 10
VAL_SPLIT = 0.1
N_TRIALS = 10

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
## 2. Tokenize + split

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
## 3. Optuna objective

# %%
def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    lora_r = trial.suggest_categorical("lora_r", [4, 8, 16])
    lora_alpha = lora_r * 2  # standard 2x rank
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.2)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)

    with mlflow.start_run(run_name=f"optuna-trial-{trial.number}", nested=True):
        mlflow.log_params({
            "model": MODEL_NAME,
            "approach": "DoRA-Optuna",
            "lr": lr,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "weight_decay": weight_decay,
            "label_smoothing": label_smoothing,
            "warmup_ratio": warmup_ratio,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
        })

        base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_dora=True,
            target_modules=["Wqkv", "Wo"],
            modules_to_save=["classifier"],
        )
        model = get_peft_model(base_model, lora_config)

        args = TrainingArguments(
            output_dir=f"./checkpoints/optuna_trial_{trial.number}",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            learning_rate=lr,
            num_train_epochs=EPOCHS,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_log_loss",
            greater_is_better=False,
            fp16=True,
            report_to="none",
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="cosine",
            weight_decay=weight_decay,
            label_smoothing_factor=label_smoothing,
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
        best = trainer.state.best_metric or 1.0
        mlflow.log_metric("best_eval_log_loss", best)

        print(f"Trial {trial.number}: best_eval_log_loss={best:.4f} lr={lr:.6f} r={lora_r} dropout={lora_dropout:.2f}")
        return best


# %% [markdown]
## 4. Run Optuna

# %%
pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
study = optuna.create_study(direction="minimize", pruner=pruner)

with mlflow.start_run(run_name="optuna-dora-search"):
    mlflow.log_param("n_trials", N_TRIALS)
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best log_loss: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    mlflow.log_metrics({"best_overall_log_loss": study.best_value})
    for k, v in study.best_params.items():
        mlflow.log_param(f"best_{k}", v)

    # save best model
    best_params = study.best_params
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=best_params["lora_r"],
        lora_alpha=best_params["lora_r"] * 2,
        lora_dropout=best_params["lora_dropout"],
        use_dora=True,
        target_modules=["Wqkv", "Wo"],
        modules_to_save=["classifier"],
    )
    model = get_peft_model(base_model, lora_config)

    args = TrainingArguments(
        output_dir="./checkpoints/optuna_best",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=best_params["lr"],
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_log_loss",
        greater_is_better=False,
        fp16=True,
        report_to="mlflow",
        warmup_ratio=best_params["warmup_ratio"],
        lr_scheduler_type="cosine",
        weight_decay=best_params["weight_decay"],
        label_smoothing_factor=best_params["label_smoothing"],
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

    best_trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    best_trainer.train()

    tmp = tempfile.mkdtemp()
    best_trainer.save_model(tmp)
    mlflow.log_artifacts(tmp, "best_model")
    shutil.rmtree(tmp)

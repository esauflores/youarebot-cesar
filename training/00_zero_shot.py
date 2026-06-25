# %% [markdown]
# Experiment 0: Zero-Shot Baseline
# DistilBERT-MNLI — per-message, no training, no feature engineering

# %%
import json, csv
from pathlib import Path
import mlflow
from transformers import pipeline
from sklearn.metrics import classification_report, accuracy_score, log_loss
from tqdm import tqdm

DATA = Path(__file__).parent / "data"
MODEL_NAME = "typeform/distilbert-base-uncased-mnli"
CANDIDATE_LABELS = ["bot", "human"]
HYPOTHESIS_TEMPLATE = "This message was written by a {}."

mlflow.set_experiment("youarebot-v2")

# %% [markdown]
## 1. Load raw data

# %%
with open(DATA / "train.json") as f:
    dialogs = json.load(f)
with open(DATA / "ytrain.csv") as f:
    rows = list(csv.DictReader(f))

samples = []
for r in rows:
    did, idx = r["dialog_id"], int(r["participant_index"])
    if did in dialogs and idx < len(dialogs[did]):
        samples.append((dialogs[did][idx]["text"], int(r["is_bot"])))

bot = sum(l for _, l in samples)
print(f"{len(dialogs)} dialogs, {len(samples)} messages — Bot: {bot}, Human: {len(samples)-bot}")

# %% [markdown]
## 2. Load model

# %%
print(f"Loading {MODEL_NAME}...")
classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME,
    hypothesis_template=HYPOTHESIS_TEMPLATE,
    device=0,
)
print("Ready.")

# %% [markdown]
## 3. Predict + log to MLflow

# %%
with mlflow.start_run(run_name="zero-shot-distilbert"):
    mlflow.log_param("model", MODEL_NAME)
    mlflow.log_param("approach", "zero-shot")
    mlflow.log_param("threshold", 0.5)
    mlflow.set_tag("notes", "Baseline: no training, distilbert-mnli zero-shot per-message.")

    y_true, y_pred, y_prob = [], [], []
    for text, label in tqdm(samples):
        r = classifier(text, candidate_labels=CANDIDATE_LABELS)
        bot_prob = r["scores"][r["labels"].index("bot")]
        y_prob.append(bot_prob)
        y_pred.append(1 if bot_prob > 0.5 else 0)
        y_true.append(label)

    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_prob)

    mlflow.log_metrics({
        "accuracy": acc,
        "log_loss": ll,
    })

    print(classification_report(y_true, y_pred, target_names=["human", "bot"]))
    print(f"Accuracy: {acc:.4f}  Log Loss: {ll:.4f}")

# %% [markdown]
## 4. Generate submission

# %%
with open(DATA / "test.json") as f:
    test_dialogs = json.load(f)
with open(DATA / "ytest.csv") as f:
    test_rows = list(csv.DictReader(f))

ids, probs = [], []
for r in tqdm(test_rows):
    did, idx = r["dialog_id"], int(r["participant_index"])
    if did in test_dialogs and idx < len(test_dialogs[did]):
        res = classifier(test_dialogs[did][idx]["text"], candidate_labels=CANDIDATE_LABELS)
        ids.append(r["ID"])
        probs.append(res["scores"][res["labels"].index("bot")])

out = DATA / "submission.csv"
with open(out, "w") as f:
    w = csv.DictWriter(f, fieldnames=["ID", "is_bot"])
    w.writeheader()
    for id_, prob in zip(ids, probs):
        w.writerow({"ID": id_, "is_bot": prob})
print(f"Saved {out} ({len(ids)} rows)")

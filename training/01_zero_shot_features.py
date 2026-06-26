# %% [markdown]
# Experiment 1: Zero-Shot with Features
# DistilBERT-MNLI on dialog context + feature prefix
# Run `build_features.py` first

# %%
import csv
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
## 1. Load clean data (dialog-level with features)

# %%
samples, labels = [], []
with open(DATA / "clean_train.csv") as f:
    for r in csv.DictReader(f):
        samples.append(r["text_with_features"])
        labels.append(int(r["is_bot"]))

bot = sum(labels)
print(f"{len(samples)} participants — Bot: {bot}, Human: {len(samples)-bot}")

# %% [markdown]
## 2. Load model

# %%
print(f"Loading {MODEL_NAME}...")
classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME,
    hypothesis_template=HYPOTHESIS_TEMPLATE,
    device=0,
    batch_size=8,
)
print("Ready.")

# %% [markdown]
## 3. Predict + log to MLflow

# %%
with mlflow.start_run(run_name="zero-shot-features"):
    mlflow.log_param("model", MODEL_NAME)
    mlflow.log_param("approach", "zero-shot-features")
    mlflow.log_param("threshold", 0.5)
    mlflow.log_param("input", "dialog_context_with_features")
    mlflow.set_tag("notes", "Zero-shot on full dialog context with <self>/<other> markers + numeric features.")

    y_true, y_pred, y_prob = [], [], []
    for text, label in tqdm(zip(samples, labels), total=len(samples)):
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
test_texts, test_ids = [], []
with open(DATA / "clean_test.csv") as f:
    for r in csv.DictReader(f):
        test_texts.append(r["text_with_features"])
        test_ids.append(r["ID"])

ids, probs = [], []
for text, id_ in tqdm(zip(test_texts, test_ids), total=len(test_texts)):
    r = classifier(text, candidate_labels=CANDIDATE_LABELS)
    ids.append(id_)
    probs.append(r["scores"][r["labels"].index("bot")])

out = DATA / "submission.csv"
with open(out, "w") as f:
    w = csv.DictWriter(f, fieldnames=["ID", "is_bot"])
    w.writeheader()
    for id_, prob in zip(ids, probs):
        w.writerow({"ID": id_, "is_bot": prob})
print(f"Saved {out} ({len(ids)} rows)")

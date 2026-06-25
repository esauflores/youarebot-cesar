# %% [markdown]
# Build Features
# Process raw JSON → clean CSV files consumed by training scripts

# %%
import json, csv, math
from pathlib import Path
from collections import Counter

DATA = Path(__file__).parent / "data"


# %%
def build_dialog(msgs, target_idx):
    n_self = sum(1 for m in msgs if m["participant_index"] == target_idx)
    n_other = len(msgs) - n_self

    self_texts = [m["text"] for m in msgs if m["participant_index"] == target_idx]
    other_texts = [m["text"] for m in msgs if m["participant_index"] != target_idx]

    self_avg_len = round(sum(len(t) for t in self_texts) / max(len(self_texts), 1))
    other_avg_len = round(sum(len(t) for t in other_texts) / max(len(other_texts), 1))

    starts = 1 if msgs[0]["participant_index"] == target_idx else 0
    ends = 1 if msgs[-1]["participant_index"] == target_idx else 0

    def char_diversity(texts):
        all_chars = "".join(texts)
        if not all_chars:
            return 0.0
        counts = Counter(all_chars)
        total = len(all_chars)
        entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
        return round(entropy, 2)

    self_div = char_diversity(self_texts)
    other_div = char_diversity(other_texts)

    tag = lambda m: "<self>" if m["participant_index"] == target_idx else "<other>"
    dialog_text = " [SEP] ".join(f"{tag(m)} {m['text']}" for m in msgs)

    prefix = (
        f"self_msgs={n_self} other_msgs={n_other} "
        f"self_len={self_avg_len} other_len={other_avg_len} "
        f"starts={starts} ends={ends} "
        f"self_div={self_div} other_div={other_div} [SEP] "
    )

    return {
        "text": dialog_text,
        "text_with_features": prefix + dialog_text,
        "self_msgs": n_self,
        "other_msgs": n_other,
        "self_avg_len": self_avg_len,
        "other_avg_len": other_avg_len,
        "starts_dialog": starts,
        "ends_dialog": ends,
        "self_char_div": self_div,
        "other_char_div": other_div,
    }


# %%
def build(name):
    with open(DATA / f"{name}.json") as f:
        dialogs = json.load(f)
    with open(DATA / f"y{name}.csv") as f:
        rows = list(csv.DictReader(f))

    label_field = "is_bot" if name == "train" else "ID"
    out_path = DATA / f"clean_{name}.csv"

    fields = [
        "text", "text_with_features",
        "self_msgs", "other_msgs", "self_avg_len", "other_avg_len",
        "starts_dialog", "ends_dialog", "self_char_div", "other_char_div", label_field,
    ]

    with open(out_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            did, idx = r["dialog_id"], int(r["participant_index"])
            if did in dialogs:
                feats = build_dialog(dialogs[did], idx)
                feats[label_field] = r[label_field]
                w.writerow(feats)

    print(f"{out_path.name} — {len(rows)} rows")

    if name == "train":
        bot = sum(1 for r in rows if r["is_bot"] == "1")
        print(f"  Bot: {bot}, Human: {len(rows)-bot}")


# %%
def build_raw(name):
    """Per-message CSV (no dialog context) — for zero-shot baseline."""
    with open(DATA / f"{name}.json") as f:
        dialogs = json.load(f)
    with open(DATA / f"y{name}.csv") as f:
        rows = list(csv.DictReader(f))

    label_field = "is_bot" if name == "train" else "ID"
    out_path = DATA / f"clean_{name}_msg.csv"

    with open(out_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=["text", label_field])
        w.writeheader()
        for r in rows:
            did, idx = r["dialog_id"], int(r["participant_index"])
            if did in dialogs and idx < len(dialogs[did]):
                w.writerow({
                    "text": dialogs[did][idx]["text"],
                    label_field: r[label_field],
                })

    print(f"{out_path.name} — {len(rows)} rows")


# %% [markdown]
## Run

# %%
build("train")

# %%
build("test")

# %%
build_raw("train")

# %%
build_raw("test")

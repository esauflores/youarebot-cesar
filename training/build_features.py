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

    def text_stats(texts):
        joined = " ".join(texts)
        words = joined.split()
        n_words = len(words)
        if n_words == 0:
            return {"ttr": 0, "repet": 0, "hapax": 0, "total": 0}
        unique = len(set(w.lower() for w in words))
        avg_wl = sum(len(w) for w in words) / n_words
        word_counts = Counter(w.lower() for w in words)
        repet = sum(c - 1 for c in word_counts.values()) / max(n_words, 1)
        hapax = sum(1 for c in word_counts.values() if c == 1) / n_words
        return {
            "ttr": round(unique / n_words, 3),
            "repet": round(repet, 3),
            "hapax": round(hapax, 3),
            "total": n_words,
        }

    self_stats = text_stats(self_texts)
    other_stats = text_stats(other_texts)

    bot_mentions = sum(1 for m in msgs if m["participant_index"] != target_idx and "bot" in m["text"].lower())

    tag = lambda m: "<self>" if m["participant_index"] == target_idx else "<other>"
    dialog_text = " [SEP] ".join(f"{tag(m)} {m['text']}" for m in msgs)

    prefix = (
        f"self_msgs={n_self} other_msgs={n_other} "
        f"self_len={self_avg_len} other_len={other_avg_len} "
        f"starts={starts} ends={ends} "
        f"self_div={self_div} other_div={other_div} "
        f"self_ttr={self_stats['ttr']} other_ttr={other_stats['ttr']} "
        f"self_repet={self_stats['repet']} other_repet={other_stats['repet']} "
        f"self_hapax={self_stats['hapax']} other_hapax={other_stats['hapax']} "
        f"self_words={self_stats['total']} other_words={other_stats['total']} "
        f"bot_mentions={bot_mentions} [SEP] "
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
        "self_ttr": self_stats["ttr"],
        "other_ttr": other_stats["ttr"],
        "self_repet_rate": self_stats["repet"],
        "other_repet_rate": other_stats["repet"],
        "self_hapax_ratio": self_stats["hapax"],
        "other_hapax_ratio": other_stats["hapax"],
        "self_total_words": self_stats["total"],
        "other_total_words": other_stats["total"],
        "bot_mentions": bot_mentions,
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
        "starts_dialog", "ends_dialog", "self_char_div", "other_char_div",
        "self_ttr", "other_ttr",
        "self_repet_rate", "other_repet_rate",
        "self_hapax_ratio", "other_hapax_ratio",
        "self_total_words", "other_total_words",
        "bot_mentions",
        label_field,
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


# %%
def analyze():
    """Compare bot vs human feature distributions."""
    with open(DATA / "clean_train.csv") as f:
        rows = list(csv.DictReader(f))

    bots = [r for r in rows if r["is_bot"] == "1"]
    humans = [r for r in rows if r["is_bot"] == "0"]

    numeric_cols = [c for c in rows[0] if c not in ("text", "text_with_features", "is_bot")]
    print(f"{'Feature':<24} {'Bot mean':>10} {'Human mean':>10} {'Diff':>10}")
    print("-" * 58)
    results = []
    for col in numeric_cols:
        b_vals = [float(r[col]) for r in bots if r[col]]
        h_vals = [float(r[col]) for r in humans if r[col]]
        if not b_vals or not h_vals:
            continue
        b_mean = sum(b_vals) / len(b_vals)
        h_mean = sum(h_vals) / len(h_vals)
        diff = b_mean - h_mean
        results.append((col, abs(diff), diff, b_mean, h_mean))
        print(f"{col:<24} {b_mean:10.3f} {h_mean:10.3f} {diff:+10.3f}")

    print(f"\nBot: {len(bots)}, Human: {len(humans)}")
    print(f"Strongest signal: {max(results, key=lambda x: x[1])[0]} ({max(results, key=lambda x: x[1])[1]:.3f})")


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

# %%
analyze()

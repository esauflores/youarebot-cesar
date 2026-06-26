# Build Features

`build_features.py` processes raw data into clean CSV files for training.

## Input

```
training/data/
├── train.json      # dialogs keyed by dialog_id, list of {text, message, participant_index}
├── ytrain.csv      # dialog_id, participant_index, is_bot
├── test.json       # same format as train.json
└── ytest.csv       # dialog_id, participant_index, ID
```

## Output

| File | Description |
|------|-------------|
| `clean_train.csv` | Dialog-level context with features, per participant |
| `clean_test.csv` | Same format for test set |
| `clean_train_msg.csv` | Raw message text, per message (for zero-shot) |
| `clean_test_msg.csv` | Same format for test set |

### clean_train.csv / clean_test.csv columns

| Column | Description |
|--------|-------------|
| `text` | Full dialog with `<self>`/`<other>` speaker markers |
| `text_with_features` | All numeric features prepended as text (use this for BERT/ModernBERT) |

**Conversation structure:**
| `self_msgs`, `other_msgs` | Message count per participant |
| `self_avg_len`, `other_avg_len` | Average message length (chars) |
| `starts_dialog`, `ends_dialog` | Who speaks first/last (0 or 1) |

**Linguistic diversity:**
| `self_char_div`, `other_char_div` | Character entropy (higher = more varied) |
| `self_ttr`, `other_ttr` | Type-Token Ratio (unique words / total words) |
| `self_avg_word_len`, `other_avg_word_len` | Average word length (chars) |

**Stylistic markers:**
| `self_punct_ratio`, `other_punct_ratio` | Punctuation density (.,!?, etc per char) |
| `self_caps_ratio`, `other_caps_ratio` | ALL CAPS word ratio |
| `self_quest_ratio`, `other_quest_ratio` | Question mark density |
| `self_repet_rate`, `other_repet_rate` | Word repetition rate (higher = more repetitive) |

**Interaction:**
| `bot_mentions` | How often the OTHER participant called THIS one "bot" |
| `is_bot` / `ID` | Label (train) or submission ID (test) |

### clean_train_msg.csv / clean_test_msg.csv columns

| Column | Description |
|--------|-------------|
| `text` | Raw message text |
| `is_bot` / `ID` | Label (train) or submission ID (test) |

## Usage

```bash
uv run training/build_features.py
```

Produces all four clean CSVs in one run. Re-run after changing features.

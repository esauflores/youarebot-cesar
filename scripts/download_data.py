"""Download Kaggle dataset (one-time)."""
import kagglehub, shutil, os
from pathlib import Path

DATA = Path(__file__).parent.parent / "training" / "data"
DATA.mkdir(parents=True, exist_ok=True)

path = kagglehub.competition_download("you-are-bot-2")
for f in os.listdir(path):
    src = os.path.join(path, f)
    dst = DATA / f
    shutil.copy2(src, dst)
    print(f"  {f} ({os.path.getsize(src):,} bytes)")
print("Dataset copied to training/data/")

import kagglehub, shutil, os
path = kagglehub.competition_download('you-are-bot-2')
for f in os.listdir(path):
    src = os.path.join(path, f)
    dst = os.path.join('training/data', f)
    shutil.copy2(src, dst)
    print(f'  {f} ({os.path.getsize(src):,} bytes)')
print('Dataset copied to training/data/')

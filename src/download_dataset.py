import os, zipfile, shutil
from pathlib import Path

RAW_DIR = Path('/content/data/raw')
RAW_DIR.mkdir(parents=True, exist_ok=True)

try:
    import kagglehub
    path = kagglehub.dataset_download("loctran0941/crowdhuman")
    for item in Path(path).iterdir():
        dest = RAW_DIR / item.name
        if not dest.exists():
            shutil.copytree(str(item), str(dest)) if item.is_dir() else shutil.copy2(str(item), str(dest))
    print(" Датасет загружен")
except Exception as e:
    print(f" Ошибка: {e}")

for z in RAW_DIR.glob('*.zip'):
    with zipfile.ZipFile(z, 'r') as zf: zf.extractall(RAW_DIR)
    z.unlink()
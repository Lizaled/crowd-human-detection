import json, random, shutil
from PIL import Image
from tqdm.notebook import tqdm

OUTPUT_DIR = Path('/content/data/yolo_crowdhuman')
TARGET_COUNT = 5000

def find_file(d, f):
    for p in d.rglob(f): return p

def load_ann(p):
    ann = {}
    with open(p, 'r') as f:
        for line in tqdm(f, desc="Аннотации"):
            try:
                d = json.loads(line.strip())
                if 'ID' in d and 'gtboxes' in d: ann[d['ID']] = d['gtboxes']
            except: continue
    return ann

def to_yolo(boxes, w, h):
    labels = []
    for b in boxes:
        key = 'vbox' if 'vbox' in b else 'fbox'
        if key not in b: continue
        x,y,bw,bh = b[key]
        if bw<=0 or bh<=0: continue
        cx,cy,nw,nh = (x+bw/2)/w, (y+bh/2)/h, bw/w, bh/h
        if nw*w<5 or nh*h<5: continue
        labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return labels

# Поиск папки с картинками
DATASET_DIR = RAW_DIR / 'CrowdHuman'
if not DATASET_DIR.exists():
    cands = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    DATASET_DIR = cands[0] if len(cands)==1 else next((c for c in cands if find_file(c,'annotation_train.odgt')), None)

ann_path = find_file(DATASET_DIR, 'annotation_train.odgt')
annotations = load_ann(ann_path)

train_img_dir = None
for name in ['Images','images','train','Train']:
    cand = DATASET_DIR / name
    if cand.exists() and len(list(cand.glob('*.jpg')))>0:
        train_img_dir = cand; break

if not train_img_dir:
    for d in DATASET_DIR.rglob('*'):
        if d.is_dir() and not d.name.startswith('.') and len(list(d.glob('*.jpg')))>10:
            train_img_dir = d; print(f" Найдена: {d.relative_to(DATASET_DIR)}"); break

pairs = [(p, annotations[p.stem]) for p in tqdm(list(train_img_dir.rglob('*.jpg'))) if p.stem in annotations]
print(f" Пар: {len(pairs)}")

selected = random.sample(pairs, TARGET_COUNT) if len(pairs)>TARGET_COUNT else pairs
random.seed(42)
split = int(len(selected)*0.8)
splits = {'train': selected[:split], 'val': selected[split:]}

for sname, data in splits.items():
    idir = OUTPUT_DIR/sname/'images'; ldir = OUTPUT_DIR/sname/'labels'
    idir.mkdir(parents=True, exist_ok=True); ldir.mkdir(parents=True, exist_ok=True)
    for img, gt in tqdm(data, desc=sname):
        with Image.open(img) as im: w,h = im.size
        lbls = to_yolo(gt, w, h)
        if lbls:
            shutil.copy2(img, idir/img.name)
            (ldir/f"{img.stem}.txt").write_text('\n'.join(lbls))

cfg = f"""path: {OUTPUT_DIR.resolve()}
train: train/images
val: val/images
nc: 1
names: ['person']"""
Path('/content/config.yaml').write_text(cfg)
print(" Датасет готов")
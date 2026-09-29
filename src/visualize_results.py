import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

# График обучения
res_path = list(Path('runs/detect').rglob('results.png'))[0] if list(Path('runs/detect').rglob('results.png')) else None
if res_path:
    plt.figure(figsize=(16,9)); plt.imshow(Image.open(res_path)); plt.axis('off'); plt.show()

# Примеры детекции
best_pt = list(Path('runs/detect').rglob('crowdhuman_demo/weights/best.pt'))[0]
model = YOLO(str(best_pt))
val_imgs = list(Path('/content/data/yolo_crowdhuman/val/images').glob('*.jpg'))[:3]
fig, axes = plt.subplots(1,3, figsize=(18,6))
for ax, ip in zip(axes, val_imgs):
    r = model.predict(str(ip), conf=0.25)[0]
    ax.imshow(r.plot()[...,::-1]); ax.set_title(f"Найдено: {len(r.boxes)}"); ax.axis('off')
plt.tight_layout(); plt.show()
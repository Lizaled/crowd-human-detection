from ultralytics import YOLO
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8n.pt')
results = model.train(
    data='/content/config.yaml', epochs=25, imgsz=640,
    batch=16 if DEVICE!='cpu' else 4, device=DEVICE,
    project='runs/detect', name='crowdhuman_demo',
    patience=5, mosaic=1.0, mixup=0.1, copy_paste=0.3, close_mosaic=5
)
print(" Обучение завершено!")
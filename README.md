# CrowdHuman Person Detection 

Проект по детекции людей в условиях высокой плотности толпы на базе датасета CrowdHuman.

##  Быстрый старт

1. Скачайте датасет: `python src/download_dataset.py`
2. Подготовьте данные: `python src/prepare_dataset.py`
3. Обучите модель: `python src/train.py`
4. Визуализируйте результаты: `python src/visualize_results.py --model runs/detect/crowdhuman_demo/weights/best.pt`

##  Результаты
| Модель | mAP@50 | mAP@50-95 | Params |
| :--- | :--- | :--- | :--- |
| YOLOv8n | 0.579 | 0.339 | 3.2M |

##  Веса модели
Файл `best.pt` лежит в папке `models/`.

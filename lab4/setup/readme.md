# СКРИПТЫ ПРЕОБРАЗОВАНИЯ МОДЕЛИ НЕЙРОННЫХ СЕТЕЙ

1. Перевести модель yolov8(выбранной размерности, доступны - YOLOv8n,YOLOv8s,YOLOv8m,YOLOv8l,YOLOv8x) в onnx с помощью 
`ultralytics` на **компьютере**

`pip3 install ultralytics`

```
from ultralytics import YOLO

model = YOLO("best_new.pt")  # load a pretrained model (recommended for training)
success = model.export(format="onnx", imgsz = (640,640))  # export the model to ONNX format
```

[Для визуализации onnx](https://netron.app/)

2. Полученый `onnx` файл необходимо перевести **engine** с помощью `onnx2tensorrt.py`

`python3 onnx2tensorrt.py -v --onnx-file best.onnx --img 640 640`





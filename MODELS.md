# Supported Models

## Object Detection

### YOLO Models

1. **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** - YOLOv8 and YOLOv11
   - YOLOv8: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
   - YOLOv11: yolov11n, yolov11s, yolov11m, yolov11l, yolov11x
2. [YOLOv7](https://github.com/WongKinYiu/yolov7)
3. [YOLOv9](https://github.com/WongKinYiu/yolov9)
4. [YOLOv10](https://github.com/THU-MIG/yolov10)
5. [YOLOv12](https://github.com/sunsmarterjie/yolov12)
6. [YOLOv13](https://github.com/iMoonLab/yolov13)
7. [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

### DETR Models

8. [RT-DETRv1](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch)
9. [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)
10. [RT-DETRv3](https://github.com/clxia12/RT-DETRv3)
11. [D-FINE](https://github.com/Peterande/D-FINE)
12. [DEIM](https://github.com/Intellindust-AI-Lab/DEIM)
13. [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)
14. [RF-DETR](https://github.com/roboflow/rf-detr)

## Classification

1. [Torchvision Classifiers](https://docs.pytorch.org/vision/main/models.html#classification)

## Model Download Support

The following models can be automatically downloaded and converted to ONNX format using the `download` CLI command:

### YOLO Models
- **YOLOv7**: All variants with pretrained weights
- **YOLOv8**: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x (via Ultralytics)
- **YOLOv9**: All variants with pretrained weights
- **YOLOv10**: All variants with pretrained weights
- **YOLOv11**: yolov11n, yolov11s, yolov11m, yolov11l, yolov11x (via Ultralytics)
- **YOLOv12**: All variants with pretrained weights
- **YOLOv13**: yolov13n, yolov13s, yolov13l, yolov13x

### DETR Models
- **RT-DETRv1**: Multiple configurations available
- **RT-DETRv2**: Multiple configurations available
- **RT-DETRv3**: Multiple configurations available
- **D-FINE**: Multiple configurations available
- **DEIM**: Multiple configurations available
- **DEIMv2**: deimv2_atto, deimv2_femto, deimv2_pico, deimv2_n, deimv2_s, deimv2_m, deimv2_l, deimv2_x
- **RF-DETR**: Multiple configurations available

Example usage:
```bash
python -m trtutils download --model yolov8n --output yolov8n.onnx
python -m trtutils download --model yolov11m --output yolov11m.onnx --imgsz 640 --opset 17
```

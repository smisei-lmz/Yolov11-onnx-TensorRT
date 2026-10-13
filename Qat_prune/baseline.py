from ultralytics import YOLO

DATA = "ultralytics_yolo11-main/coco-val.local.yaml"
model = YOLO("models/yolo11s.pt")
metrics = model.val(data=DATA, imgsz=640, batch=1, device=0)
print("FP32 mAP50-95:", metrics.box.map)
print("FP32 mAP50:", metrics.box.map50)


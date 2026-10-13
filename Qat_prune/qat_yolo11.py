from pathlib import Path
import onnx
from ultralytics import YOLO

DATA = "coco-train.local.yaml"
RUN = Path("runs/qat/yolo11s")

model = YOLO("models/yolo11s.pt")
model.train(
    data=DATA,
    imgsz=640,
    epochs=5,             # 学习实验先跑 5 轮
    batch=8,              # 显存不足则调小
    device=0,
    quantize=8,           # 开启 QAT
    optimizer="AdamW",
    lr0=1e-5,
    lrf=0.1,
    warmup_epochs=0.5,
    mosaic=0.0,
    project="runs/qat",
    name="yolo11s",
    exist_ok=True,
)

best = RUN / "weights/best.pt"
qat = YOLO(str(best))
metrics = qat.val(data=DATA, imgsz=640, batch=1, device=0)
print("QAT checkpoint mAP50-95:", metrics.box.map)

onnx_path = Path(qat.export(
    format="onnx",
    imgsz=640,
    batch=1,
    dynamic=False,
    simplify=False,
    nms=False,            # 保持与你当前推理代码相同的原始输出
))
graph = onnx.load(str(onnx_path))
ops = [node.op_type for node in graph.graph.node]
print("ONNX:", onnx_path)
print("Q nodes:", ops.count("QuantizeLinear"))
print("DQ nodes:", ops.count("DequantizeLinear"))
assert "QuantizeLinear" in ops and "DequantizeLinear" in ops
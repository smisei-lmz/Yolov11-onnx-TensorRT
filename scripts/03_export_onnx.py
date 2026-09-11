from ultralytics import YOLO
import torch
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PT_PATH = ROOT / "models/yolo11s.pt"
DST_PATH = ROOT / "models/yolo11s_static_Nodynamic_Nosimplify_Havenms.onnx"

def main():
    model = YOLO(PT_PATH)
    result = model.export(format="onnx",imgsz=640,batch=1,dynamic=False,simplify=False,nms=True)
    print("Export result:", result)
    src = Path(result)
    if src.resolve() != DST_PATH.resolve():
        src.rename(DST_PATH)
    print("Saved ONNX to:", DST_PATH)   

if __name__ == "__main__":
    main()



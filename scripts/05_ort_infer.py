import numpy as np
import onnxruntime as ort

ONNX_Model_Path = "/home/smisei/Yolov11-ONNX-TensorRT/models/yolo11s_static_Nodynamic_Nosimplify_Nonms.onnx"
INPUT_PATH = "/home/smisei/Yolov11-ONNX-TensorRT/data/input_random.npy"

def main():
    # -----------------------------
    # 1. 看当前ORT支持哪些Provider
    # -----------------------------
    print("Available providers:")
    print(ort.get_available_providers())
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "CUDAExecutionProvider 不可用，请先检查 onnxruntime-gpu / CUDA / cuDNN 环境"
        )
    # -----------------------------
    # 2. 创建InferenceSession
    # -----------------------------
    session = ort.InferenceSession(ONNX_Model_Path,providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    print("\nSession providers:")
    print(session.get_providers())

if __name__ == "__main__":
    main()
import numpy as np
import torch
import onnxruntime as ort

from ultralytics import YOLO


PT_PATH = "/home/smisei/Yolov11-ONNX-TensorRT/models/yolo11s.pt"
ONNX_PATH = "/home/smisei/Yolov11-ONNX-TensorRT/models/yolo11s_static_Nodynamic_Nosimplify_Nonms.onnx"
INPUT_PATH = "/home/smisei/Yolov11-ONNX-TensorRT/data/input_random.npy"


# ==========================================================
# 打印PyTorch返回结构
# ==========================================================

def print_structure(obj,prefix="output"):
    if torch.is_tensor(obj):
        print(prefix,"Tensor",tuple(obj.shape),obj.dtype)
    elif isinstance(obj,(list, tuple)):
        print(prefix,type(obj).__name__,"len=",len(obj))
        for i, item in enumerate(obj):
            print_structure(item,f"{prefix}[{i}]")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            print_structure(value,f"{prefix}[{key}]")
    else:
        print(prefix,type(obj))

# ==========================================================
# 把PT返回转换为要比较的raw prediction tensor
# ==========================================================
def get_pt_raw_output(output):
    # 如果直接返回Tensor
    if torch.is_tensor(output):
        return output
    # YOLO eval常见情况：
    # output是tuple，
    # 第一个通常是最终prediction tensor
    if isinstance(output,(tuple, list)):
        if (len(output) > 0 and torch.is_tensor(output[0])):
            return output[0]
    raise RuntimeError(
        "目前adapter无法识别PT输出，请根据实际输出结构修改"
    )

def main():
    # ======================================================
    # 1. 同一个输入
    # ======================================================
    x_np = np.load(INPUT_PATH).astype(np.float32)
    x_np = np.ascontiguousarray(x_np)
    print("Input:",x_np.shape,x_np.dtype)
    # =====================================================
    # 2. PyTorch
    # ======================================================
    print("\n========== PyTorch ==========")
    yolo = YOLO(PT_PATH)
    model = yolo.model
    model.cuda()
    model.eval()
    x_pt = torch.from_numpy(x_np).cuda()
    with torch.inference_mode():
        pt_output = model(x_pt)
    print_structure( pt_output)
    pt_raw = get_pt_raw_output(pt_output)
    pt_raw = (pt_raw.detach().float().cpu().numpy())
    print("PT raw:",pt_raw.shape)
    # ======================================================
    # 3. ONNX Runtime
    # ======================================================
    print("\n========== ONNX Runtime ==========")
    session = ort.InferenceSession(ONNX_PATH,providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    input_name = (session.get_inputs()[0].name)
    ort_outputs = session.run(None,{input_name: x_np})
    ort_raw = ort_outputs[0]
    print("ORT raw:",ort_raw.shape)
    # ======================================================
    # 4. shape必须先相同
    # ======================================================
    if (pt_raw.shape != ort_raw.shape):
        raise RuntimeError(
            f"Shape不同: "
            f"PT={pt_raw.shape}, "
            f"ORT={ort_raw.shape}"
        )
    # ======================================================
    # 5. Error
    # ======================================================
    diff = np.abs(pt_raw - ort_raw)
    print("\n========== ERROR ==========")
    print("max abs error:",diff.max())
    print("mean abs error:",diff.mean())
    print("median abs error:",np.median(diff))
    # ======================================================
    # 6. cosine similarity
    # ======================================================
    pt_flat = pt_raw.reshape(-1)
    ort_flat = ort_raw.reshape(-1)
    cosine = (np.dot(pt_flat,ort_flat)/(np.linalg.norm(pt_flat)*np.linalg.norm(ort_flat)+1e-12))
    print("cosine similarity:", cosine)
    # ======================================================
    # 7. allclose
    # ======================================================
    close = np.allclose(pt_raw,ort_raw,rtol=1e-4,atol=1e-4)
    print("allclose:",close)

if __name__ == "__main__":
    main()
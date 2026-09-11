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
    # 加载 pip 安装在 NVIDIA site-packages 目录中的 CUDA/cuDNN 运行库。
    ort.preload_dlls(directory="")
    session = ort.InferenceSession(ONNX_Model_Path,providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    print("\nSession providers:")
    print(session.get_providers())
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("CUDA 后端加载失败，当前 session 已回退到 CPU，请检查前面的错误日志")
    # -----------------------------
    # 3. 查看模型输入
    # -----------------------------
    print("\n===== INPUTS =====")
    for x in session.get_inputs():
        print("name = ",x.name)
        print("shape = ",x.shape)
        print("type = ",x.type)
    # -----------------------------
    # 4. 查看模型输出
    # -----------------------------
    print("\n===== OUTPUTS =====")
    for x in session.get_outputs():
        print("name = ",x.name)
        print("shape = ",x.shape)
        print("type = ",x.type)
    # -----------------------------
    # 5. 加载Day 1固定输入
    # -----------------------------
    x = np.load(INPUT_PATH)
    x = np.ascontiguousarray(x,dtype=np.float32)
    print("\n===== INPUT DATA =====")
    print("shape:",x.shape)
    print("dtype:",x.dtype)
    print("min:",x.min())
    print("max:",x.max())
    print("mean:",x.mean())
    # -----------------------------
    # 6. 获取输入名字
    # -----------------------------
    input_name = (session.get_inputs()[0].name)    
    # -----------------------------
    # 7. ONNX Runtime执行
    # -----------------------------
    outputs = session.run(None,{input_name:x})
    # -----------------------------
    # 8. 打印输出
    # ----------------------------- 
    print("\n===== ORT OUTPUT =====")
    print("number of outputs:",len(outputs))
    for i, y in enumerate(outputs):
        print(f"output[{i}]")
        print("shape:",y.shape)
        print("dtype:", y.dtype)
        print("min:",y.min())
        print("max:",y.max())
        print("mean:",y.mean())
        print("finite:",np.isfinite(y).all())
        # 保存结果
        np.save(f"results/ort_output_{i}.npy",y)


if __name__ == "__main__":
    main()

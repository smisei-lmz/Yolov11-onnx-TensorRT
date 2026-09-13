import csv
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


ONNX_PATH = "/home/smisei/Yolov11-ONNX-TensorRT/models/yolo11s_static_Nodynamic_Nosimplify_Nonms.onnx"

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data/preprocessed_image.npy"

RESULT_PATH = ROOT / "results/ort_benchmark.csv"

WARMUP = 50
ITERATIONS = 300

def main():
    ort.preload_dlls(directory="")
    option = ort.SessionOptions()
    option.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    session = ort.InferenceSession(
        ONNX_PATH,sess_options=option,providers=["CUDAExecutionProvider","CPUExecutionProvider"]
    )
    print(session.get_providers())
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("CUDA 后端未启用，当前结果不能作为 GPU benchmark")
    x = np.load(INPUT_PATH)
    x = np.ascontiguousarray(x,dtype=np.float32)
    input_name = session.get_inputs()[0].name
    feed = {input_name:x}
    for _ in range(WARMUP):
        session.run(None,feed)
    latencies_ms = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        session.run(None,feed)
        end = time.perf_counter()
        latency_ms = (end - start)*1000
        latencies_ms.append(
            latency_ms
        )
    latencies_ms = np.array(
        latencies_ms,
        dtype=np.float64
    )
    mean = latencies_ms.mean()
    minimum = latencies_ms.min()
    maximum = latencies_ms.max()    
    P50 = np.percentile(latencies_ms,50)
    P99 = np.percentile(latencies_ms,99)
    P75 = np.percentile(latencies_ms,75)
    fps = float(1000.0 / mean)
    print(f"mean: {mean:.3f} ms")
    print(f"min:  {minimum:.3f} ms")
    print(f"p50:  {P50:.3f} ms")
    print(f"p75:  {P75:.3f} ms")
    print(f"p99:  {P99:.3f} ms")
    print(f"max:  {maximum:.3f} ms")
    print(f"fps:  {fps:.2f}")

'''
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "latency_ms"])
        writer.writerows(enumerate(latencies_ms))
'''

if __name__ == "__main__":
    main()

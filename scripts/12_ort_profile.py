import csv
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort



ONNX_PATH = "/home/smisei/Yolov11-ONNX-TensorRT/models/yolo11s_static_Nodynamic_Nosimplify_Nonms.onnx"

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data/preprocessed_image.npy"

def main():
    ort.preload_dlls(directory="")
    options = ort.SessionOptions()

    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    # 开启profiling
    options.enable_profiling = True

    session = ort.InferenceSession(
        ONNX_PATH,

        sess_options=options,

        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
    )

    x = np.load(
        INPUT_PATH
    ).astype(
        np.float32
    )

    x = np.ascontiguousarray(
        x
    )

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    # Warmup
    for _ in range(20):

        session.run(
            None,
            {
                input_name: x
            }
        )

    # 正式跑一些
    for _ in range(50):

        session.run(
            None,
            {
                input_name: x
            }
        )

    # 结束并生成JSON文件
    profile_file = (
        session.end_profiling()
    )

    print(
        "Profile saved:"
    )

    print(
        profile_file
    )


if __name__ == "__main__":
    main()
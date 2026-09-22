from pathlib import Path

import numpy as np

from modelopt.onnx.quantization import quantize


ONNX_PATH = "models/yolo11s_static_Nodynamic_Nosimplify_Nonms.onnx"

CALIBRATION_PATH = "data/calibration.npy"

OUTPUT_PATH = "models/yolo11s_static_Nodynamic_Nosimplify_Nonms_int8.onnx"


def main():

    assert Path(ONNX_PATH).exists(), ONNX_PATH
    assert Path(CALIBRATION_PATH).exists(), CALIBRATION_PATH

    print("Loading calibration data...")

    calibration_data = np.load(
        CALIBRATION_PATH
    )

    print(
        "Calibration:",
        calibration_data.shape,
        calibration_data.dtype
    )

    print()
    print("Start INT8 PTQ...")
    print()

    quantize(
        onnx_path=ONNX_PATH,

        # INT8 PTQ
        quantize_mode="int8",

        # [N,3,640,640]
        calibration_data=calibration_data,

        # 激活值 calibration 算法
        calibration_method="entropy",

        # 优先使用GPU
        calibration_eps=[
            "cuda:0",
            "cpu"
        ],

        # 未量化部分允许使用 FP16
        high_precision_dtype="fp16",

        output_path=OUTPUT_PATH,

        log_level="INFO"
    )

    print()
    print("==============================")
    print("INT8 PTQ finished")
    print("==============================")
    print("Input :", ONNX_PATH)
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
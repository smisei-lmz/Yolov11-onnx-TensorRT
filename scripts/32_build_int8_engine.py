from pathlib import Path

import tensorrt as trt


ONNX_PATH = "/home/smisei/Yolov11-ONNX-TensorRT/models/yolo11s_static_Nodynamic_Nosimplify_Nonms_int8.onnx"

ENGINE_PATH = "/home/smisei/Yolov11-ONNX-TensorRT/models/yolo11s_static_Nodynamic_Nosimplify_Nonms_int8.engine"


TRT_LOGGER = trt.Logger(
    trt.Logger.INFO
)


def main():

    print(
        "TensorRT version:",
        trt.__version__
    )

    if not Path(ONNX_PATH).exists():
        raise FileNotFoundError(
            ONNX_PATH
        )

    builder = trt.Builder(
        TRT_LOGGER
    )

    # TensorRT 11:
    # 所有 network 默认 strongly typed
    network = builder.create_network()

    parser = trt.OnnxParser(
        network,
        TRT_LOGGER
    )

    print()
    print("Parsing ONNX...")

    success = parser.parse_from_file(
        ONNX_PATH
    )

    if not success:

        print()
        print("ONNX parse failed:")

        for i in range(
            parser.num_errors
        ):

            print(
                parser.get_error(i)
            )

        raise RuntimeError(
            "Failed to parse ONNX"
        )

    print("ONNX parse success")

    print()
    print("===== Network =====")

    print(
        "num_inputs :",
        network.num_inputs
    )

    print(
        "num_outputs:",
        network.num_outputs
    )

    for i in range(
        network.num_inputs
    ):

        tensor = network.get_input(i)

        print(
            f"Input {i}:",
            tensor.name,
            tensor.shape,
            tensor.dtype
        )

    for i in range(
        network.num_outputs
    ):

        tensor = network.get_output(i)

        print(
            f"Output {i}:",
            tensor.name,
            tensor.shape,
            tensor.dtype
        )

    config = builder.create_builder_config()

    # 最大 Builder workspace：4 GB
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        4 << 30
    )

    # -------------------------------
    # TensorRT 11 不要写：
    #
    # config.set_flag(
    #     trt.BuilderFlag.INT8
    # )
    #
    # 也不要：
    #
    # config.int8_calibrator = ...
    #
    # INT8信息已经写在Q/DQ ONNX里了
    # -------------------------------

    print()
    print(
        "Building INT8 TensorRT engine..."
    )

    serialized_engine = (
        builder.build_serialized_network(
            network,
            config
        )
    )

    if serialized_engine is None:
        raise RuntimeError(
            "Engine build failed"
        )

    Path(ENGINE_PATH).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        ENGINE_PATH,
        "wb"
    ) as f:

        f.write(
            serialized_engine
        )

    engine_size = (
        Path(ENGINE_PATH).stat().st_size
        / 1024
        / 1024
    )

    print()
    print("===========================")
    print("Engine build success")
    print("===========================")

    print(
        "Engine:",
        ENGINE_PATH
    )

    print(
        f"Size: {engine_size:.2f} MB"
    )


if __name__ == "__main__":
    main()
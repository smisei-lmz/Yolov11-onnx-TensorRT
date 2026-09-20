from pathlib import Path

import tensorrt as trt


ONNX_PATH = (
    "/home/smisei/Yolov11-ONNX-TensorRT/models/yolo11s_static_Nodynamic_Nosimplify_Nonms_mixed_fp16.onnx"
)

ENGINE_PATH = (
    "/home/smisei/Yolov11-ONNX-TensorRT/models/yolo11s_static_Nodynamic_Nosimplify_Nonms_mixed_fp16.engine"
)


def main():

    logger = trt.Logger(
        trt.Logger.INFO
    )


    # =====================================
    # Builder
    # =====================================

    builder = trt.Builder(
        logger
    )


    # TensorRT 11:
    # Strongly Typed默认开启
    network = (
        builder.create_network()
    )


    config = (
        builder
        .create_builder_config()
    )


    parser = trt.OnnxParser(
        network,
        logger
    )


    # =====================================
    # Parse ONNX
    # =====================================

    with open(
        ONNX_PATH,
        "rb"
    ) as f:

        success = parser.parse(
            f.read()
        )


    if not success:

        print(
            "ONNX parse failed"
        )

        for i in range(
            parser.num_errors
        ):

            print(
                parser.get_error(i)
            )

        raise RuntimeError(
            "parse failed"
        )


    print(
        "ONNX parse success"
    )


    # =====================================
    # Network IO
    # =====================================

    print(
        "\n===== Inputs ====="
    )


    for i in range(
        network.num_inputs
    ):

        t = network.get_input(i)

        print(
            t.name,
            t.shape,
            t.dtype
        )


    print(
        "\n===== Outputs ====="
    )


    for i in range(
        network.num_outputs
    ):

        t = network.get_output(i)

        print(
            t.name,
            t.shape,
            t.dtype
        )


    # =====================================
    # Build
    # =====================================

    print(
        "\nBuilding mixed FP16 engine..."
    )


    engine_bytes = (
        builder
        .build_serialized_network(
            network,
            config
        )
    )


    if engine_bytes is None:

        raise RuntimeError(
            "Engine build failed"
        )


    Path(
        ENGINE_PATH
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        ENGINE_PATH,
        "wb"
    ) as f:

        f.write(
            engine_bytes
        )


    print(
        "\nsaved:",
        ENGINE_PATH
    )


if __name__ == "__main__":

    main()
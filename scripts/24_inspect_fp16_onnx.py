from collections import Counter

import onnx

from onnx import TensorProto


MODEL_PATH = (
    "/home/smisei/Yolov11-ONNX-TensorRT/models/yolo11s_static_Nodynamic_Nosimplify_Nonms_mixed_fp16.onnx"
)


TYPE_NAMES = {

    TensorProto.FLOAT:
        "FP32",

    TensorProto.FLOAT16:
        "FP16",

    TensorProto.INT64:
        "INT64",

    TensorProto.INT32:
        "INT32"
}


def get_type_name(
    elem_type
):

    return TYPE_NAMES.get(
        elem_type,
        str(elem_type)
    )


def main():

    model = onnx.load(
        MODEL_PATH
    )


    print(
        "===== Inputs ====="
    )

    for inp in model.graph.input:

        elem_type = (
            inp
            .type
            .tensor_type
            .elem_type
        )

        print(
            inp.name,
            get_type_name(
                elem_type
            )
        )


    print(
        "\n===== Outputs ====="
    )

    for out in model.graph.output:

        elem_type = (
            out
            .type
            .tensor_type
            .elem_type
        )

        print(
            out.name,
            get_type_name(
                elem_type
            )
        )


    # =====================================
    # Weight dtype统计
    # =====================================

    weights = Counter()


    for initializer in (
        model.graph.initializer
    ):

        weights[
            get_type_name(
                initializer.data_type
            )
        ] += 1


    print(
        "\n===== Initializer Types ====="
    )


    for dtype, count in (
        weights.items()
    ):

        print(
            dtype,
            count
        )


    # =====================================
    # Cast统计
    # =====================================

    cast_count = sum(
        1
        for node
        in model.graph.node
        if node.op_type == "Cast"
    )


    print(
        "\nCast nodes:",
        cast_count
    )


    # =====================================
    # Operator统计
    # =====================================

    ops = Counter(
        node.op_type
        for node
        in model.graph.node
    )


    print(
        "\n===== Top Operators ====="
    )


    for name, count in (
        ops.most_common(20)
    ):

        print(
            f"{name:20s} {count}"
        )


if __name__ == "__main__":

    main()
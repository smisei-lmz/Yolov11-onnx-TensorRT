from collections import Counter

import onnx


MODEL_PATH = "/home/smisei/Yolov11-ONNX-TensorRT/models/yolo11s_static_Nodynamic_Nosimplify_Nonms_int8.onnx"


def get_shape(value_info):

    dims = []

    tensor_type = value_info.type.tensor_type

    for dim in tensor_type.shape.dim:

        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)

        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)

        else:
            dims.append("?")

    return dims


def main():

    print(f"Loading {MODEL_PATH}")

    model = onnx.load(MODEL_PATH)

    print()
    print("Checking ONNX...")
    onnx.checker.check_model(model)

    print("ONNX checker: PASS")

    print()
    print("===== Opset =====")

    for opset in model.opset_import:
        print(
            f"domain={opset.domain or 'ai.onnx'}, "
            f"version={opset.version}"
        )

    print()
    print("===== Inputs =====")

    for inp in model.graph.input:
        print(
            inp.name,
            get_shape(inp)
        )

    print()
    print("===== Outputs =====")

    for out in model.graph.output:
        print(
            out.name,
            get_shape(out)
        )

    op_counter = Counter(
        node.op_type
        for node in model.graph.node
    )

    print()
    print("===== Operator Count =====")

    for op, count in op_counter.most_common():
        print(f"{op:25s}: {count}")

    q_num = op_counter.get(
        "QuantizeLinear",
        0
    )

    dq_num = op_counter.get(
        "DequantizeLinear",
        0
    )

    print()
    print("===== Quantization =====")

    print(
        "QuantizeLinear   :",
        q_num
    )

    print(
        "DequantizeLinear :",
        dq_num
    )

    if q_num == 0 or dq_num == 0:

        print()
        print(
            "WARNING: No Q/DQ nodes found!"
        )

    else:

        print()
        print(
            "Q/DQ nodes exist."
        )

        print(
            "This is an explicit INT8 quantized model."
        )


if __name__ == "__main__":
    main()
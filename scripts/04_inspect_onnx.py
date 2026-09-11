import onnx
from collections import Counter

MODEL_PATH = "/home/smisei/Yolov11-ONNX-TensorRT/models/yolo11s_static_Nodynamic_Nosimplify_Nonms.onnx"


def get_shape(value_info):
    dims = (value_info.type.tensor_type.shape.dim)
    shape = []
    for dim in dims:
        if dim.dim_value:
            shape.append(dim.dim_value)
        elif dim.dim_param:
            shape.append(dim.dim_param)
        else:
            shape.append("?")
    return shape


def main():
    model = onnx.load(MODEL_PATH)
    # --------------------------------
    # 1. 检查ONNX是否合法
    # --------------------------------
    onnx.checker.check_model(model)
    print("\nONNX model check: PASS")
    # --------------------------------
    # 2. 模型基本信息
    # --------------------------------
    print("\n===== MODEL INFO =====")
    print("IR version:",model.ir_version)
    print("Producer:", model.producer_name)
    print("Producer version:",model.producer_version)
    print("\nOpset:")
    for opset in model.opset_import:
        print("domain:",opset.domain,"version:",opset.version)
    # --------------------------------
    # 3. 输入
    # --------------------------------
    print("\n===== INPUT =====")
    for x in model.graph.input:
        print("name:",x.name)
        print("shape:", get_shape(x))
        print("elem_type:",x.type.tensor_type.elem_type)
    # --------------------------------
    # 4. 输出
    # --------------------------------
    print("\n===== OUTPUT =====")
    for x in model.graph.output:
        print("name:",x.name)
        print( "shape:", get_shape(x))
        print( "elem_type:", x.type.tensor_type.elem_type)
    # --------------------------------
    # 5. Graph整体信息
    # --------------------------------
    print("\n===== GRAPH =====")
    print("graph name:",model.graph.name)
    print( "number of nodes:",len(model.graph.node))
    print("number of initializers:",len(model.graph.initializer))
    # --------------------------------
    # 6. 统计所有算子
    # --------------------------------
    counter = Counter(node.op_type for node in model.graph.node)
    print("\n===== OPERATORS =====")
    for op, count in counter.most_common():
        print(f"{op:20s} {count}")
    # --------------------------------
    # 7. 前20个Node
    # --------------------------------
    print("\n===== FIRST 20 NODES =====")
    for i, node in enumerate(model.graph.node[:20]):
        print("\nNode", i)
        print( "name:",node.name)
        print( "op:",node.op_type)
        print( "inputs:",list(node.input))
        print("outputs:",list(node.output))

if __name__ == "__main__":
    main()
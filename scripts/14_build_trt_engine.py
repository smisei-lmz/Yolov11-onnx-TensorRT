from pathlib import Path

import tensorrt as trt

ONNX_PATH = "/home/smisei/Yolov11-ONNX-TensorRT/models/yolo11s_static_Nodynamic_Nosimplify_Nonms.onnx"
ENGINE_PATH = "models/yolo11s_static_Nodynamic_Nosimplify_Nonms_fp32.engine"

def main():
    logger = trt.Logger(trt.Logger.INFO)
    Build = trt.Builder(logger)
    network = Build.create_network()
    paser = trt.OnnxParser(network,logger)
    with open(ONNX_PATH,"rb") as f:
        onnx_data = f.read()
    success = paser.parse(onnx_data)
    for i in range(network.num_inputs):
        input = network.get_input(i)
        print("name:",input.name)
        print("shape:",input.shape)
        print("type:",input.dtype)
    for i in range(network.num_outputs):
        output = network.get_output(i)
        print("name:",output.name)
        print("shape:",output.shape)
        print("type:",output.dtype)
    config = Build.create_builder_config()
    serialized_engine_modle = Build.build_serialized_network(network,config)
    engine_path = Path(ENGINE_PATH)
    with open(engine_path,"wb") as f:
        f.write(serialized_engine_modle)

if __name__ == "__main__":

    main()
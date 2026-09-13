import numpy as np
import torch
import tensorrt as trt

INPUT_PATH = (
    "data/preprocessed_image.npy"
)
ENGINE_PATH = "models/yolo11s_static_Nodynamic_Nosimplify_Nonms_fp32.engine"

def main():
    logger = trt.Logger(trt.Logger.INFO)
    with open(ENGINE_PATH,"rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()

    input_names = []
    output_names = []

    for i in range (engine.num_io_tensors):
        name = (engine.get_tensor_name(i))
        mode = (engine.get_tensor_mode(name))
        if (mode==trt.TensorIOMode.INPUT):
            input_names.append(name)
        else:
            output_names.append(name)
    input_name = (input_names[0])      
    x_np = np.load(INPUT_PATH).astype(np.float32)

    x_gpu = torch.from_numpy(x_np).contiguous().cuda()
    context.set_tensor_address(input_name,x_gpu.data_ptr()) #告诉context你要的数据在gpu的这块地址

    outputs = {}
    for output_name in output_names:
        output_shape = tuple(context.get_tensor_shape(output_name))
        output_dtype = (engine.get_tensor_dtype(output_name))

        output_gpu = torch.empty( output_shape,dtype=torch.float32,device="cuda")
        outputs[ output_name] = output_gpu

        context.set_tensor_address(output_name,output_gpu.data_ptr())

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        success = (context.execute_async_v3(stream_handle=stream.cuda_stream))
    if not success:
        raise RuntimeError("TensorRT inference failed")
    stream.synchronize()
    for name, tensor in outputs.items():
        output_np = (tensor .cpu() .numpy())
    np.save(f"results/"f"trt_{name}.npy",output_np)    


if __name__ == "__main__":
    main()
    
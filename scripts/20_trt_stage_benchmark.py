import csv

import numpy as np
import torch
import tensorrt as trt


ENGINE_PATH = (
    "/home/smisei/Yolov11-ONNX-TensorRT/models/yolo11s_static_Nodynamic_Nosimplify_Nonms_fp32.engine"
)
INPUT_PATH = (
    "data/preprocessed_image.npy"
)
RESULT_PATH = (
    "results/day8_trt_stage.csv"
)
WARMUP = 50
ITERATIONS = 300

def main():
    logger = trt.Logger(trt.Logger.WARNING)
    with open(ENGINE_PATH,"rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    input_names = []
    output_names = []
    for i in range(engine.num_io_tensors):
        name = (engine.get_tensor_name(i))
        mode = (engine.get_tensor_mode(name))
        if ( mode ==trt.TensorIOMode.INPUT):input_names.append(name)
        else:
            output_names.append(name)
    input_name = input_names[0]
    output_name = output_names[0]    
    #######CPU输入 
    x_np = np.load(INPUT_PATH).astype(np.float32)
    x_np = np.ascontiguousarray(x_np)
    x_pageable = torch.from_numpy(x_np)
    ######$提前申请Pinned CPU Input
    x_cpu = torch.empty(x_pageable.shape,dtype=torch.float32,pin_memory=True)
    x_cpu.copy_(x_pageable)
    x_gpu = torch.empty(x_cpu.shape,dtype=torch.float32,device="cuda")
    output_shape = tuple(context.get_tensor_shape(output_name))
    y_gpu = torch.empty(output_shape,dtype=torch.float32,device="cuda")
    y_cpu = torch.empty(output_shape,dtype=torch.float32,pin_memory=True)
    context.set_tensor_address(input_name,x_gpu.data_ptr())
    context.set_tensor_address(output_name,y_gpu.data_ptr())
    stream = torch.cuda.Stream()
    for _ in range(WARMUP):
        with torch.cuda.stream(stream):
            # H2D
            x_gpu.copy_(x_cpu,non_blocking=True)
            # TensorRT
            context.execute_async_v3(stream.cuda_stream)
            # D2H
            y_cpu.copy_(y_gpu,non_blocking=True)
    stream.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)
    e3 = torch.cuda.Event(enable_timing=True)
    records = []
    for _ in range(ITERATIONS):
        with torch.cuda.stream(stream):
            #start
            e0.record(stream)
            # -------------------------
            # H2D
            # -------------------------
            x_gpu.copy_(x_cpu,non_blocking=True)
            e1.record(stream)
            #infreance
            success = context.execute_async_v3(stream.cuda_stream)
            if not success:
                raise RuntimeError("TensorRT inference failed")
            e2.record(stream)
            # -------------------------
            # D2H
            # -------------------------
            y_cpu.copy_(y_gpu,non_blocking=True)
            e3.record(stream)
        stream.synchronize()
        h2d_ms = (e0.elapsed_time(e1))
        infer_ms = (e1.elapsed_time(e2))
        d2h_ms = (e2.elapsed_time(e3))
        total_ms = (e0.elapsed_time(e3))
        records.append(
            (
                h2d_ms,
                infer_ms,
                d2h_ms,
                total_ms
            )
        )
    records = np.asarray(
        records,
        dtype=np.float64
    )
    h2d = records[:, 0]
    infer = records[:, 1]
    d2h = records[:, 2]
    total = records[:, 3]

    print(
        "\n===== Day8 TensorRT Pipeline ====="
    )

    print(
        f"H2D Mean   : {h2d.mean():.3f} ms"
    )

    print(
        f"TRT Mean   : {infer.mean():.3f} ms"
    )

    print(
        f"D2H Mean   : {d2h.mean():.3f} ms"
    )

    print(
        f"Total Mean : {total.mean():.3f} ms"
    )

    print(
        f"Total P50  : "
        f"{np.percentile(total, 50):.3f} ms"
    )

    print(
        f"Total P90  : "
        f"{np.percentile(total, 90):.3f} ms"
    )

    print(
        f"Total P99  : "
        f"{np.percentile(total, 99):.3f} ms"
    )

    print(
        f"FPS        : "
        f"{1000.0 / total.mean():.2f}"
    )





if __name__ == "__main__":
    main()
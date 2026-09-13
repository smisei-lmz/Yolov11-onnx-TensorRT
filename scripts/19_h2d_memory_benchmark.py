import numpy as np
import torch

INPUT_PATH = "data/preprocessed_image.npy"
WARMUP = 20
ITERATIONS = 300

def benchmark_copy(src_cpu,name,non_blocking = True):
    # -----------------------------------
    # GPU Buffer只申请一次
    # -----------------------------------
    dst_gpu = torch.empty(size=src_cpu.shape,dtype=src_cpu.dtype,device="cuda")
    stream = torch.cuda.Stream()
    for _ in range (WARMUP):
        with torch.cuda.stream(stream):
            dst_gpu.copy_(src_cpu,non_blocking=non_blocking)
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(stream):
        start.record(stream)
        for _ in range(ITERATIONS):
            dst_gpu.copy_(src_cpu,non_blocking=non_blocking)
        end.record(stream)
    stream.synchronize()
    total_ms = start.elapsed_time(
        end
    )
    avg_ms = total_ms / ITERATIONS

    size_mb = src_cpu.numel() * src_cpu.element_size() / 1024 / 1024

    bandwidth_gbps  = size_mb / 1024 / (avg_ms/1000.0)
    print(
        f"{name:25s} "
        f"{avg_ms:.4f} ms "
        f"{bandwidth_gbps:.3f} GB/s"
    )


def main():
    x_np = np.load(INPUT_PATH).astype(np.float32)
    x_np = np.ascontiguousarray(x_np)
    pageable = torch.from_numpy(x_np).contiguous()
    pinned = torch.empty(pageable.shape,dtype=pageable.dtype,pin_memory=True)
    pinned.copy_(pageable)

    benchmark_copy(pageable,"pageable, blocking",False)
    benchmark_copy(pageable,"pageable, async",True)
    benchmark_copy(pinned,"pinned, blocking",False)
    benchmark_copy(pinned,"pinned, async",True)




if __name__ == "__main__":
    main()

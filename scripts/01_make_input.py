#该代码用于创建固定输入
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

x = torch.rand(1,3,640,640,dtype=torch.float32)

print("shape:",x.shape)
print("dtype:",x.dtype)
print("min:",x.min().item())
print("max:",x.max().item())
print("mean:", x.mean().item())
torch.save(x,"data/input_random.pt")
np.save("data/input_random.npy",x.numpy)

zero = torch.zeros(1,3,640,640,dtype=torch.float32)
print("shape:",zero.shape)
print("dtype:",zero.dtype)
print("min:",zero.min().item())
print("max:",zero.max().item())
print("mean:", zero.mean().item())
torch.save(zero,"data/input_zero.pt")
np.save("data/input_zero.npy",zero.numpy)
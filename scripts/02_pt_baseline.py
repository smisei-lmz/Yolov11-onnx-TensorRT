import torch
from ultralytics import YOLO

device = "cuda"
yolo_model = YOLO("./models/yolo11s.pt")

net = yolo_model.model
net.to(device)
net.eval()

x = torch.load("./data/input_random.pt",weights_only=True)
x = x.to(device)
print("Input shape:",x.shape)
print("Input dtype:",x.dtype)

with torch.inference_mode():
    output = net(x)
print("Output dtype:",type(output))

def inspect_output(obj, prefix="output"):
    if torch.is_tensor(obj):
        print(prefix,"Tensor","shape=",tuple(obj.shape),"dtype=",obj.dtype)
    elif isinstance(obj,(list, tuple)):
        print(prefix, type(obj).__name__,"len=",len(obj))
        for i, item in enumerate(obj):
            inspect_output(item,f"{prefix}[{i}]")
    elif isinstance(obj, dict):
        print(prefix,"dict")
        for key, value in obj.items():
            inspect_output(value,f"{prefix}[{key}]")
    else:
        print(prefix,type(obj))

inspect_output(output)

pred = output[0]
print(pred.shape) #torch.Size([1, 84, 8400])
#output[0]存放8400个候选框的坐标和80种类别的概率
'''
[batch, 4个框坐标 + 80个类别概率, 候选框数量]
pred[:, 0, :]      cx
pred[:, 1, :]      cy
pred[:, 2, :]      width
pred[:, 3, :]      height
pred[:, 4, :]      person 概率
pred[:, 5, :]      bicycle 概率
...
pred[:, 83, :]     toothbrush 概率
'''
'''
对于 640×640 输入：
| 检测层 | stride | 特征图 | 候选位置 |
|---|---:|---:|---:|
| P3 | 8 | 80×80 | 6400 |
| P4 | 16 | 40×40 | 1600 |
| P5 | 32 | 20×20 | 400 |
'''


torch.save(output,"./results/pt_reference_output.pt")
print("\nSaved reference output.")
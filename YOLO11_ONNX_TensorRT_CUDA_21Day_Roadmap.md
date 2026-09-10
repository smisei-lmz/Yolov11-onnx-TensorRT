# YOLO11 → ONNX → TensorRT → CUDA：21 天项目驱动学习路线

> 适用起点：已经在 WSL2 + Ubuntu 22.04 环境跑通 airockchip YOLO11 的 PyTorch 测试与验证，已安装 CUDA Toolkit 13.2。
> 主线：YOLO11 → ONNX → ONNX Runtime → TensorRT → C++ → CUDA → FP16 / INT8 → Nsight。
> 整理日期：2026-09-10。本文是学习任务书；代码、模型、性能和精度结果需要在你的机器上实际完成，不代表已经测试通过。

## 1. 使用方法与最终目标

围绕同一个模型、同一套输入和同一份后处理，建立 `yolo11_deploy` 工程。每天执行“阅读 → 编码 → 验证 → 记录”。不要把阅读完文档等同于掌握，也不要把生成 engine 等同于部署完成。

21 天是任务顺序，不是必须赶完的期限。建议每天投入 3～5 小时；C++、CUDA 基础薄弱或遇到版本兼容问题时，将对应一天拆成两天。未通过正确性验收，就暂停后续性能优化。

完成后，你应能交付：

- 一个可复现的 PT / ORT / TensorRT 数值与检测精度对比流程。
- 一个支持图片输入、CPU 后处理的 TensorRT C++ 推理程序。
- 一个与 CPU 参考实现对齐的 CUDA 图像预处理模块。
- 静态与动态输入实验，以及 FP32 图、FP16 图和显式 INT8 Q/DQ 图的构建与评估记录。
- 一份区分模型执行和端到端耗时的基准表，以及用 Nsight 证据支持的优化报告。

第一轮固定：检测模型 `yolo11n.pt`、batch=1、输入 `[1,3,640,640]`、FP32、关闭导出内置 NMS、暂不简化图。模型名称仅为约定；若你已有自训练权重，继续使用该权重并记录类别数。

## 2. 开始前必须固定的两个约定

### 2.1 先确认你用的是哪一种 YOLO11 输出

airockchip 分支不能直接当成上游 Ultralytics 使用。其说明明确记录了输出结构修改、DFL 移到外部后处理、增加置信度汇总分支等改动。因此，不能默认导出结果只有一个 `[1,84,8400]` 张量，也不能把 PyTorch 返回值机械地取 `[0]` 后比较。[Rockchip 导出说明](https://github.com/airockchip/ultralytics_yolo11/blob/main/RKOPT_README.zh-CN.md)

Day 1 必须在 `configs/model_contract.json` 写清楚：

| 字段 | 需要记录的内容 |
|---|---|
| 来源 | 仓库 URL、commit、权重 SHA256、实际导入的 ultralytics 路径 |
| 输入 | 名称、dtype、NCHW/NHWC、尺寸、RGB/BGR、归一化范围 |
| 输出 | 每个名称、shape、dtype、语义、对应尺度或分支 |
| 解码 | DFL 是否在图内、是否已变成框坐标、是否已 sigmoid |
| 后处理 | 类别数、置信度计算、NMS 类型、阈值、坐标还原方法 |

默认继续你的 Rockchip 分支。若其导出路径固定了尺寸，则动态形状阶段先检查并修改尺寸相关代码，重新验证；不能只改 ONNX 输入维度名字。若希望另外使用上游版本练習，单独创建环境和模型文件，并重新建立基线，不覆盖已经跑通的 Rockchip 环境。

### 2.2 版本以实际环境为准

`nvidia-smi` 显示的 CUDA 版本、`nvcc` 对应的 Toolkit、PyTorch 自带 CUDA runtime、ORT 所需 CUDA/cuDNN，并非同一件事。不要仅凭“装了 CUDA 13.2”认定全部库兼容。

截至本次核对，ORT 官方 CUDA EP 页面说明：从 ORT 1.27 起，PyPI GPU 包默认采用 CUDA 13.0 构建；具体安装仍需同时核对 CUDA、cuDNN 和驱动要求。`get_available_providers()` 只说明可用后端列表；必须实际创建会话、执行模型并检查 profiling，确认节点执行位置。[ORT CUDA EP 要求](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

本路线默认采用 TensorRT 11 的命名张量 API。TensorRT 11 强制强类型网络，旧的 `BuilderFlag.FP16`、`BuilderFlag.INT8`、`--fp16`、`--int8` 和隐式校准流程不能直接照搬；精度阶段改用图类型转换与显式 Q/DQ。不要把原对话中的某个“最新小版本号”作为安装约束。[Python 迁移指南](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-python-api-patterns.html)、[trtexec 迁移指南](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-trtexec.html)

TensorRT 的 pip 安装不包含 C++ 头文件；`trtexec` 来自非 pip 安装方式。本项目需要 C++ 开发文件与 CLI，Day 8 按支持矩阵选择匹配的 Linux 开发包或 tar 包，同时核对 GPU 架构与 Ubuntu 支持情况。[安装方式](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html)、[pip 安装限制](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html)

如果硬件或系统只能支持其他 TensorRT 主版本，固定该版本并使用对应文档；不要在同一份示例里混用 8.x bindings、10.x 精度开关和 11.x API。

## 3. 最终项目目录

以下是你在 WSL Ubuntu 中逐步建立的工程，不要求第一天写完所有文件。所有命令默认从 `yolo11_deploy/` 根目录执行；编号文件通过路径直接执行，共用模块从 `common` 导入。

```text
yolo11_deploy/
├── README.md
├── requirements-lock.txt
├── .gitignore
├── configs/
│   ├── model_contract.json
│   ├── export.json
│   ├── benchmark.json
│   └── quantization.json
├── models/
│   ├── yolo11n.pt
│   ├── yolo11n_static.onnx
│   ├── yolo11n_dynamic.onnx
│   ├── yolo11n_fp16.onnx
│   ├── yolo11n_int8_qdq.onnx
│   ├── toy.onnx
│   ├── toy_inferred.onnx
│   └── engines/                 # 分别保存 static/dynamic/fp16/int8 engine
├── common/
│   ├── __init__.py
│   ├── preprocess.py            # CPU 图像处理参考实现
│   ├── postprocess.py           # 分支解码、NMS、坐标还原
│   ├── output_adapter.py        # 对齐不同后端输出的语义与顺序
│   ├── metrics.py               # 数值误差与检测评估辅助函数
│   └── tensor_io.py             # 二进制张量及 JSON 元数据交换
├── 00_env/
│   └── collect_env.py
├── 01_onnx/
│   ├── 01_export_onnx.py
│   ├── 02_inspect_onnx.py
│   ├── 03_make_toy_onnx.py
│   ├── 04_ort_infer.py
│   ├── 05_compare_pt_ort.py
│   └── 06_dynamic_shape.py
├── 02_tensorrt/
│   ├── 01_build_engine.py
│   ├── 02_inspect_engine.py
│   ├── 03_trt_infer.py
│   ├── 04_benchmark.py
│   └── 05_dynamic_profile.py
├── 03_cpp/
│   ├── CMakeLists.txt
│   ├── main.cpp
│   ├── trt_engine.h
│   ├── trt_engine.cpp
│   ├── preprocess_cpu.cpp
│   ├── postprocess.h
│   └── postprocess.cpp
├── 04_cuda/
│   ├── CMakeLists.txt
│   ├── cuda_vector.cu
│   ├── memory_benchmark.cu
│   ├── preprocess.h
│   ├── preprocess.cu
│   └── benchmark.cpp
├── 05_precision/
│   ├── 01_convert_fp16.py
│   ├── 02_prepare_calibration.py
│   ├── 03_quantize_int8.py
│   └── 04_eval_precision.py
├── 06_profile/
│   ├── run_nsys.sh
│   ├── run_ncu.sh
│   └── analyze_benchmark.py
├── data/
│   ├── images/                  # 少量正确性样例
│   ├── tensors/                 # 固定输入、参考输出及元数据
│   ├── calibration.txt          # 校准图片清单
│   └── validation.txt           # 独立、有标注的验证集清单
├── results/
│   ├── environment.txt
│   ├── graph_summary.txt
│   ├── pt_onnx_error.txt
│   ├── ort_profile.json
│   ├── trt_error.json
│   ├── dynamic_shapes.csv
│   ├── trtexec_fp32.txt
│   ├── benchmark.csv
│   ├── precision.csv
│   ├── detections/
│   └── profiles/
└── docs/
    ├── daily_log.md
    ├── output_contract.md
    ├── onnx_notes.md
    ├── runtime_lifecycle.md
    ├── memory_notes.md
    ├── precision_report.md
    └── optimization_report.md
```

大数据集、权重、engine、构建目录和 Nsight 原始报告通常放入 `.gitignore`；README 保存获取方法、哈希和复现命令。Engine 按模型哈希、GPU、TensorRT 版本、精度和 profile 命名，默认在目标环境重建。

## 4. 21 天总览

| 天数 | 阶段 | 当天核心成果 |
|---|---|---|
| Day 1 | A：固定基线 | 环境记录、模型契约、PT 基准输入与输出 |
| Day 2 | B：ONNX | 静态导出与图检查 |
| Day 3 | B：ONNX | 手写 toy ONNX 与 NumPy 验证 |
| Day 4 | C：ORT | CPU/CUDA 会话与原始张量推理 |
| Day 5 | C：ORT | PT/ORT 同输入、同语义输出对齐 |
| Day 6 | C：ORT | 真实图片预处理、解码、NMS 与检测图 |
| Day 7 | C：ORT | 动态 ONNX 的 320/640/960 验证 |
| Day 8 | D：TensorRT | 完整开发环境与首个 CLI engine |
| Day 9 | D：TensorRT | Python Builder 和 engine 检查器 |
| Day 10 | D：TensorRT | Python Runtime 与 ORT 数值对齐 |
| Day 11 | D：TensorRT | 动态 profile 与首份统一基准 |
| Day 12 | E：C++/CUDA 基础 | CUDA vector、内存与异步拷贝实验 |
| Day 13 | E：C++ | C++ engine 加载和固定张量推理 |
| Day 14 | E：C++ | 图片检测与 Python/C++ 对齐 |
| Day 15 | F：CUDA 预处理 | 通道转换、归一化、布局转换 kernel |
| Day 16 | F：CUDA 预处理 | 双线性 resize、letterbox 与边界验证 |
| Day 17 | F：CUDA 预处理 | 预处理接入推理及端到端对比 |
| Day 18 | G：精度 | FP16 图、engine、误差和性能 |
| Day 19 | G：精度 | INT8 Q/DQ、校准、检测精度评估 |
| Day 20 | H：Nsight | 系统时间线与自写 kernel 分析 |
| Day 21 | H：交付 | 一个有证据的优化、复测、完整 README |

## 5. 阶段 A：固定环境与 PyTorch 基线（Day 1）

**目标：** 后续任何误差都能追溯到模型、输入、输出契约或运行环境。

**要看什么：** 阅读你当前仓库的导出说明、Detect 的 `forward` 与导出分支、已经跑通的验证入口；只追踪输入到输出的实际路径。

**要写什么：** `00_env/collect_env.py` 记录 Python 包版本、GPU、导入路径；`configs/model_contract.json` 保存接口约定；`common/tensor_io.py` 保存连续 FP32 张量为 `.bin`，另写 `.json` 记录名称、shape、dtype、字节序与布局。

### Day 1 执行清单

- [ ] 在独立项目目录保存现有权重的引用或副本，记录仓库 commit 和权重哈希。
- [ ] 收集以下输出，保存到 `results/environment.txt`。

```bash
nvidia-smi
nvcc --version
python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name())"
python -c "import ultralytics; print(ultralytics.__file__)"
python -m pip freeze > requirements-lock.txt
```

- [ ] 用固定随机种子生成 `[1,3,640,640]` FP32 输入并保存；优先用 `[0,1]` 随机值，再补充零输入。
- [ ] 设置模型 `eval()`，在 inference/no-grad 模式执行，打印返回结构及全部张量 shape。
- [ ] 保存可明确解释的参考输出；若普通 PT 返回与导出分支不同，记录后续需要的 adapter。
- [ ] 选择 5～10 张不同宽高的图片，并准备独立有标注的验证集清单。

**验收问题：** 实际导入的是哪个仓库？权重对应多少类别？PT 是否包含后处理？为什么 `nvidia-smi` 和 `torch.version.cuda` 不一定相同？

**产出物：** 环境记录、锁定依赖、模型契约、固定输入、PT 输出、数据清单。

## 6. 阶段 B：把 ONNX 当成计算图理解（Day 2～3）

**目标：** 能从 PyTorch 模块追踪到 ONNX 节点，并独立构造一个可执行的小图。

**要看什么：** ONNX 的 ModelProto/GraphProto、节点输入输出、initializer、attribute、opset 与 IR；阅读 helper、checker、shape inference 对应 API。重点看 Conv、MatMul、Add、Reshape、Transpose、Resize，而非通读所有算子。[ONNX Python 入门](https://onnx.ai/onnx/intro/python.html)

### Day 2：静态导出与检查

- [ ] 写 `01_onnx/01_export_onnx.py`：参数化权重、输出路径、imgsz、opset、dynamic；保存实际导出参数。
- [ ] 上游接口可按下面形式起步；Rockchip 分支先遵循其 exporter，逐项确认这些参数是否生效。

```python
from pathlib import Path
import shutil
from ultralytics import YOLO

root = Path(__file__).resolve().parents[1]
model = YOLO(str(root / "models/yolo11n.pt"))
exported = Path(model.export(
    format="onnx", imgsz=640, batch=1,
    dynamic=False, simplify=False, nms=False,
))
target = root / "models/yolo11n_static.onnx"
if exported.resolve() != target.resolve():
    shutil.copy2(exported, target)
```

以上为上游接口示例，不保证适用于每个 fork。第一次记录导出器实际选择的 opset；确认 ORT 与 TensorRT parser 支持后，将该值固定到配置，不盲目追求最大 opset。[Ultralytics 导出参数](https://docs.ultralytics.com/modes/export)

- [ ] 写 `02_inspect_onnx.py`：执行 checker；打印 IR/opset、输入输出、权重数、节点数、算子频次及前 20 个节点。
- [ ] 用 Netron 找到 backbone、neck、head；选一个 stride=2 的 Conv，手算输出空间尺寸并核对。
- [ ] 对图执行 shape inference，观察 `value_info`；记录仍无法推断的动态维度。

**验收：** 能说明图的 output 指向哪个节点、哪些数据是权重、哪些是运行时输入；checker 通过只表示图合法，不能证明检测结果正确。

**产出：** `yolo11n_static.onnx`、`configs/export.json`、`graph_summary.txt`、图结构笔记。

### Day 3：手写 toy ONNX

- [ ] 写 `03_make_toy_onnx.py`，构造 `Y = Relu(X @ W + B)`，形状分别为 `[1,4]`、`[4,3]`、`[3]`、`[1,3]`。
- [ ] W/B 使用 initializer；显式选择已验证受 ORT 支持的 opset 与 IR 组合，记录版本选择理由。
- [ ] 运行 checker、shape inference，保存原图与推断图。
- [ ] 用 NumPy 和 ORT CPU 执行同一输入，使用 `np.testing.assert_allclose` 验证，如先用 `rtol=1e-5, atol=1e-6`。
- [ ] 故意把 W 的维度改错一次，阅读报错，再修复。

**验收问题：** initializer 与 node 有何区别？opset 与 IR 有何区别？为什么 shape inference 不能补出所有 shape？ONNX 是否能仅靠权重表达完整推理过程？

**产出：** toy 脚本、`toy.onnx`、`toy_inferred.onnx`、NumPy/ORT 对比记录。toy 通过后再继续 YOLO。

## 7. 阶段 C：ORT 执行、正确性与动态输入（Day 4～7）

**目标：** 建立后续所有后端共享的正确性基线。

**要看什么：** ORT InferenceSession、输入输出元数据、ExecutionProvider、profiling；阅读自己的 YOLO 后处理实现，追踪 DFL、框解码、阈值筛选、NMS 和坐标还原。[ORT Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)、[Rockchip YOLO11 示例](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolo11)

### Day 4：先跑固定张量

- [ ] 按兼容表安装选定版本的 ONNX/ORT，避免在同一环境混装 CPU 与 GPU 两种 ORT 包。
- [ ] 写 `04_ort_infer.py`：支持模型路径、输入文件、CPU/CUDA 后端选择、输出目录。
- [ ] 先用 CPU session 跑通；再用 CUDA session，输出全部张量及其元数据。
- [ ] 开启 `SessionOptions.enable_profiling`；调用 `end_profiling()` 保存报告，检查实际节点分配和初始化警告。
- [ ] 记录 CPU fallback，不能仅凭列表里有 CUDA 就宣布全图在 GPU 上执行。

**验收：** 改错输入名、dtype 或 shape 后，能解释错误；能说明 `session.run()` 接收 NumPy 输入时与数据拷贝的关系。

**产出：** ORT 输出、CPU/CUDA 日志、`results/ort_profile.json`。

### Day 5：同输入、同输出语义的数值对齐

- [ ] 写 `common/output_adapter.py`：显式描述每个 PT 张量如何对应 ONNX 输出，必要时进行布局转换或同样的解码。
- [ ] 写 `05_compare_pt_ort.py`：读取 Day 1 同一输入，运行两后端，逐个输出验证 shape、dtype、有限值与误差。
- [ ] 若 PT 为已解码输出，而 Rockchip ONNX 为多尺度原始分支，先取 PT 对应导出分支，或给两边执行相同解码；不能直接相减。
- [ ] 写 `common/metrics.py`，计算 max/mean absolute error、相对误差分位数、allclose 失败比例；框坐标与类别分数分开统计。

```python
import numpy as np

def compare(ref, pred, atol, rtol):
    assert ref.shape == pred.shape
    assert np.isfinite(ref).all() and np.isfinite(pred).all()
    a, b = ref.astype(np.float64), pred.astype(np.float64)
    diff = np.abs(a - b)
    allowed = atol + rtol * np.abs(a)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "failed_ratio": float(np.mean(diff > allowed)),
    }
```

对 FP32 可从 `atol=1e-4, rtol=1e-3` 作为排查起点，但这不是所有 YOLO 分支的统一质量标准。根据数值尺度、TF32 设置和检测评估确定最终门槛，并提前写入配置。不要为了通过而反复放宽阈值；整体 cosine 很高也不能掩盖类别分数错误。

**验收：** 能证明两边输入一致、输出语义一致；发现差异时能优先排查导出分支、布局、sigmoid/DFL，而不是直接重装环境。

**产出：** `pt_onnx_error.txt`、输出映射说明、固定容差配置。

### Day 6：真实图片与共享后处理

- [ ] 写 `common/preprocess.py`：固定尺寸 letterbox、padding 值、BGR→RGB、HWC→CHW、FP32/255、增加 batch；返回实际缩放和 padding 元数据。
- [ ] 保存一张图片生成的最终输入 tensor，两后端直接读取它，避免重复预处理引入差异。
- [ ] 写 `common/postprocess.py`：按契约处理多输出；只在需要时执行 DFL 和 sigmoid，再做框解码、置信度筛选、按类 NMS。
- [ ] 统一阈值和 NMS 行为，将框从 letterbox 坐标还原到原图并裁剪到图像范围。
- [ ] 测试横图、竖图、奇数宽高、小图、无检出样例；保存图片和结构化检测结果。

**验收：** 能解释为何“输出张量对齐”与“画框看起来接近”是不同层次；知道为什么不能给已经归一化的数据再次 `/255`。

**产出：** 共享前后处理、检测图、`docs/output_contract.md`、真实图片误差记录。

### Day 7：动态 ONNX

- [ ] 写 `06_dynamic_shape.py`，导出动态模型并检查输入中的符号维度及图内尺寸计算。
- [ ] 依次运行 `[1,3,320,320]`、`[1,3,640,640]`、`[1,3,960,960]`；再加入合法非正方形尺寸，如 `[1,3,384,640]`。
- [ ] 每个尺寸都生成同一份 PT/ORT 输入并对齐；输出候选数变化应能从特征图尺度解释。
- [ ] 区分动态 batch 与动态分辨率；本轮 batch 仍固定为 1。

**验收问题：** 输入维度写成符号就一定支持动态执行吗？哪些解码常量可能固定尺寸？为什么输出 shape 会随分辨率改变？

**产出：** 动态 ONNX、`dynamic_shapes.csv`、各尺寸误差。静态正确性未通过，不进入 TensorRT。

## 8. 阶段 D：TensorRT Builder、Runtime 与基准（Day 8～11）

**目标：** 分清“编译 engine”与“执行 engine”，用自己的程序完成推理与对齐。

**要看什么：** TensorRT 架构、安装/支持矩阵、Python API 的 build/runtime 示例、动态 profile、trtexec 帮助。只围绕当前阶段查相应章节。[TensorRT 文档入口](https://docs.nvidia.com/deeplearning/tensorrt/latest/)、[Python API 流程](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html)

### Day 8：开发环境与第一个 engine

- [ ] 记录 TensorRT、CUDA、GPU 型号；确认 CLI、Python 包、C++ 库与头文件来自匹配版本。
- [ ] 验证 `import tensorrt` 和 `trtexec --help`，保存日志；若 CLI 不在 PATH，使用安装包中的实际绝对路径。
- [ ] 从静态 FP32 ONNX 构建 engine，记录 parser 的全部错误而非只看最后一行。

```bash
trtexec --onnx=models/yolo11n_static.onnx --saveEngine=models/engines/yolo11n_static.engine > results/trtexec_fp32.txt 2>&1
trtexec --loadEngine=models/engines/yolo11n_static.engine --dumpProfile --dumpLayerInfo
```

先创建 `models/engines` 和 `results` 目录。具体可选参数以当前 `--help` 为准；这些命令没有加入旧式精度开关。

**验收：** 能画出 `ONNX → Parser → Network → Builder → Engine` 与 `Engine → Runtime → Context → GPU` 两条流程。

**产出：** engine、构建日志、安装版本说明。这里的 FP32 指图类型；实际数学模式仍需记录，不能据此宣称严格 IEEE FP32。

### Day 9：自己写 Builder 与检查器

- [ ] `01_build_engine.py`：读取模型、构造 logger/builder/network/parser/config、打印全部 parser 错误、构建序列化结果、检测空返回、保存 engine。
- [ ] 提供 `--onnx`、`--engine` 参数；保存模型哈希、版本、构建配置和构建耗时。
- [ ] `02_inspect_engine.py`：枚举 `num_io_tensors`，打印每个张量的 name、mode、dtype、shape、location；列出 profile 信息。
- [ ] 写 `docs/runtime_lifecycle.md`：解释 engine 与 context 的职责及对象生命周期。

**验收：** 为什么 engine 不能默认跨 GPU/版本拷贝？为什么动态 engine 中的 `-1` 不能直接用于内存分配？

**产出：** 两个脚本、engine 元数据、生命周期笔记。

### Day 10：Python Runtime 真正推理

- [ ] 写 `03_trt_infer.py`：反序列化 engine、创建 context、准备所有输入输出 buffer、绑定地址、异步执行、同步后读取结果。
- [ ] 使用命名张量 `set_tensor_address()` 与 `execute_async_v3()`；枚举所有输出，不硬编码只有一个输出。
- [ ] 初版可用 PyTorch CUDA tensor 持有显存；按 engine dtype 分配连续 tensor，地址来自 `data_ptr()`。输入准备、执行和读取之间显式建立 stream 顺序，保证 tensor 生命周期覆盖 GPU 工作。
- [ ] 检查 API 返回值；按运行时实际 shape 分配输出，执行后与 ORT 使用同一 adapter 比较。
- [ ] 图片前后处理继续复用 Day 6，不在今天重写。

**验收：** 输入指针为什么必须仍然有效？异步函数返回是否说明结果已经算好？为什么 engine 有几个输出就必须处理几个输出？

**产出：** 可运行 Python Runtime、`trt_error.json`、图片验证结果。

### Day 11：动态 profile 与首份基准

- [ ] `05_dynamic_profile.py` 为真实输入名添加 MIN=`(1,3,320,320)`、OPT=`(1,3,640,640)`、MAX=`(1,3,960,960)`。
- [ ] 运行时设置 profile/输入 shape，再从 context 查询输出 shape，最后分配/绑定 buffer。不同版本的方法与返回值查当前 API。
- [ ] 验证 320/640/960，并主动测试超出范围的 shape；在应用入口给出清晰错误。
- [ ] 同 profile 中仍需满足网络尺寸约束，MIN/MAX 并不保证所有中间整数尺寸都合法。
- [ ] 写 `04_benchmark.py`：warmup 50 次、测量至少 500 次，重复 3 轮；保存每次样本，输出 mean/P50/P90/P99。
- [ ] 对比 PT、ORT、TRT；统一输入、batch、数学模式、计时边界，单独记录 H2D/D2H 是否包含。

**验收：** OPT 是主要优化尺寸而非数学中点；为什么 profile 更宽未必更快？为什么 CLI 成绩不能直接与 Python 端到端耗时相比？

**产出：** 动态 engine、尺寸正确性表、首份 `benchmark.csv`。

## 9. 阶段 E：CUDA 内存基础与 C++ Runtime（Day 12～14）

**目标：** 理解 CPU/GPU 数据流，独立写出有资源管理的 C++ 推理程序。主线进入 C++ 前，先用一天补齐必要的 CUDA 内存知识。

**要看什么：** CUDA Runtime 的 allocation、copy、stream、event；TensorRT C++ Runtime 示例；CMake 的 CUDA Toolkit 查找。重点追踪资源创建、使用、同步和释放。[CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)、[TensorRT C++ API 流程](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/c-api-docs.html)

### Day 12：脱离模型写 CUDA 小实验

- [ ] 写 `cuda_vector.cu`：100 万个 float，CPU→GPU，kernel 做 `x*2`，GPU→CPU，逐元素检查。
- [ ] 包含越界保护 `if (i < n)`，检查内存分配、拷贝、kernel launch 与同步错误。
- [ ] 写 `memory_benchmark.cu`：比较 pageable/pinned host buffer 的传输；观察同步与异步拷贝。
- [ ] 学习 `cudaMalloc/cudaFree`、`cudaMemcpyAsync`、`cudaStreamCreate`、`cudaEventRecord`；首次用一个 stream。
- [ ] 写 CPU RAM→H2D→GPU buffer→kernel→D2H→CPU RAM 数据流图。

**验收：** Async 是否必然与其他操作重叠？为什么 pinned memory 有助于异步传输？何时才可覆盖输入或读取输出？

**产出：** CUDA demo、拷贝实验数据、`memory_notes.md`。

### Day 13：固定张量的 C++ 推理

- [ ] 写 `trt_engine.h/.cpp`：封装 runtime、engine、context、stream 和 device buffers，采用 RAII，明确析构顺序。
- [ ] 写 `main.cpp`：加载 engine，读取 Day 1 `.bin + .json` 输入，绑定全部 IO，执行并保存输出。
- [ ] 数据流：H2D → `setTensorAddress` → `enqueueV3` → D2H → stream 同步。
- [ ] 写 `CMakeLists.txt`：找到 CUDA Toolkit、TensorRT include/library；仅 runtime 推理不需要无理由链接 ONNX parser。
- [ ] CLI 至少支持 `--engine`、`--input`、`--output-dir`；错误时返回非零退出码。

**验收：** C++ 和 Python 输入字节数完全一致；输出 shape/dtype 相同；同一 engine 的结果在约定容差内对齐；重复运行不持续增长显存。

**产出：** C++ 可执行程序、构建命令、二进制交换协议、对齐日志。

### Day 14：C++ 图片检测

- [ ] `preprocess_cpu.cpp` 实现与 Python 同样的 letterbox 与布局转换，先比较输入 tensor。
- [ ] `postprocess.h/.cpp` 移植已验证的解码/NMS，参数统一；不要凭通用 YOLO 示例猜你的输出格式。
- [ ] `main.cpp` 增加 `--image`、阈值、可视化及结果 JSON 输出。
- [ ] 用 Day 6 的宽图、竖图、奇数尺寸、无检出图片验证。
- [ ] 对 NMS 阈值边缘样例记录差异，按类别和 IoU 匹配检测框，不要求输出顺序绝对一致。

**验收问题：** 错框能否定位到预处理、engine 原始输出或后处理？连续处理多张图片是否重复分配显存？

**产出：** C++ 图片检测程序、检测图、Python/C++ 三层对齐记录。

## 10. 阶段 F：CUDA 图像预处理（Day 15～17）

**目标：** 把已经正确的 CPU 预处理搬到 GPU，证明数值行为和端到端收益。

**要看什么：** CUDA grid/block/thread、内存索引、连续访问、双线性插值；重点查看现有 CPU resize 的坐标规则与取整方式。[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### Day 15：先实现不改变尺寸的 kernel

- [ ] 写 `preprocess.h`，接口携带原图指针、width、height、row stride、目标指针及 stream。
- [ ] 写 `preprocess.cu`：BGR→RGB、uint8→float32、除以 255、HWC→CHW。
- [ ] 先用固定大小图片，直接对比 CPU 参考 tensor；测试 1×1、小图和行跨度不等于 width×3 的输入。
- [ ] 写 `benchmark.cpp`，分别测 kernel 与含输入拷贝的路径，预分配内存。

**验收：** 能从 `(x,y,c)` 推出原图和 CHW 输出索引；没有把颜色通道错误当成精度误差。

**产出：** 第一版 kernel、逐元素误差和 kernel 计时。

### Day 16：加入 resize 与 letterbox

- [ ] 在 CPU 明确计算实际 resize 后宽高、左右上下 padding 和坐标还原参数。
- [ ] 实现反向采样：从目标像素定位到源图，再做双线性插值；明确定义像素中心、边界钳制和 padding 分支。
- [ ] 核对 CPU 库是否先把插值结果舍入为 uint8；GPU 直接保留 float 可能产生小差异，需要记录并评估。
- [ ] 覆盖横竖图、极小图、奇数宽高、padding 不对称、边缘像素和纯色图片。
- [ ] 比较 tensor、原始检测输出和最终检测三层结果；出现边缘误差先画差异图定位。

**验收：** 能解释半像素坐标与舍入差异；CPU/GPU 使用同一组 letterbox 元数据还原坐标。

**产出：** 完整 CUDA 预处理、边界用例结果、容差及其理由。

### Day 17：接入 C++ 推理

- [ ] 流程改为原图 uint8 H2D → CUDA preprocess → TensorRT → D2H → CPU postprocess。
- [ ] 预处理输出直接写入 engine 输入 buffer；保持同一 stream 的执行顺序，避免不必要的中间回传。
- [ ] 将 host/device buffer 分配移出稳态循环；支持 `--preprocess cpu|cuda`。
- [ ] 复用同一模型和图片，测 CPU 与 CUDA 两种端到端路径；单独记原图上传、预处理和推理时间。

**验收问题：** kernel 快了，端到端一定快吗？上传高分辨率原图是否增加 H2D？优化节省了哪一段时间？

**产出：** 集成程序、更新的 `benchmark.csv`、正确性回归记录。只有实际测量支持时，才写“加速”。

## 11. 阶段 G：FP16 与 INT8（Day 18～19）

**目标：** 将精度视为可验证的工程选择，同时衡量检测质量、延迟和内存。

**要看什么：** TensorRT 11 precision/migration、ModelOpt 的 ONNX 类型转换与量化示例、显式 QuantizeLinear/DequantizeLinear、校准数据要求。[精度控制](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/precision-control.html)、[NVIDIA Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)

### Day 18：FP16 图与 engine

- [ ] 写 `01_convert_fp16.py`，调用与你锁定工具版本一致的官方类型转换流程；保存完整命令、配置和转换日志。
- [ ] 明确是内部 FP16 + FP32 IO，还是 IO 也变 FP16。Runtime 必须按 engine 实际 dtype 分配，不能只更改文件名。
- [ ] 保留数值敏感操作为更高精度的策略必须写入配置，并检查转换后的图。
- [ ] 构建独立 FP16 engine；先测 NaN/Inf 与分组数值误差，再用相同验证集评估检测精度。
- [ ] 写 `04_eval_precision.py`，复用共享后处理，输出 mAP50、mAP50-95、误差、模型/engine 大小和延迟。

**验收：** 为什么强类型 FP32 图不能仅靠旧 flag 变 FP16？哪些算子可能对精度敏感？精度下降与性能收益是否满足预先设定的要求？

**产出：** `yolo11n_fp16.onnx`、engine、`precision.csv` 的 FP16 行、转换配置。

### Day 19：显式 INT8 Q/DQ

- [ ] `02_prepare_calibration.py` 生成有代表性的校准数据：光照、尺度、类别、背景接近目标场景；例如先选 200～500 张，再做覆盖与收敛检查，数量不是通用保证。
- [ ] 校准集与有标注验证集分开；校准输入必须使用部署同款预处理，保存样本列表、随机种子和数据哈希。
- [ ] `03_quantize_int8.py` 调用锁定版本的 ModelOpt 显式量化流程，生成带 Q/DQ 的 ONNX；读取工具当前示例确定参数，不照搬旧 calibrator。
- [ ] 检查量化节点、scale、未量化层和 parser 日志；构建独立 engine，检查实际层实现，不能从文件名推断全网 INT8。
- [ ] 使用 Day 18 评估脚本比较 PT/FP32、FP16、INT8；加入易漏检的小目标和困难样例分析。
- [ ] 如果精度不达标，检查校准代表性、输出解码、量化覆盖、敏感层；逐项改变一个因素后复测。

**验收问题：** scale 与 clipping 改变什么？PTQ 与 QAT 有何区别？为什么不能拿随机噪声校准真实检测模型？为什么 INT8 不保证更快？

**产出：** Q/DQ ONNX、engine、校准清单与配置、`precision_report.md`。如果被硬件或算子支持阻塞，记录具体错误与复现方法，保留为待完成项，不能把失败日志当作已完成 INT8 部署。

精度门槛在测量前约定。例如实验可先设 mAP50-95 相对 FP32 基线下降不超过 0.5 个百分点，但最终应由你的任务要求决定。必须注明“百分点”还是“相对百分比”，以及样本数；少量图片目视检查不能替代精度评估。

## 12. 阶段 H：Nsight 分析与最终交付（Day 20～21）

**目标：** 用时间线和指标定位瓶颈，完成一个有可复现证据的优化。

**要看什么：** Nsight Systems 的 CUDA/NVTX 时间线，Nsight Compute 的 kernel 指标、memory throughput、occupancy 与瓶颈解释；先看整体流程，再看自己的 kernel。[Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/)、[Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/)

WSL2 的 profiling 能力依赖 GPU、Windows 驱动、系统和工具版本，并可能受性能计数器权限影响。按当前支持文档检查，不能假设安装了 CUDA 就能采集所有指标。[CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)、[Nsight Compute 支持与限制](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html)

### Day 20：先 Systems，再 Compute

- [ ] 给 C++ 程序加入 NVTX 区间：preprocess、H2D、infer、D2H、postprocess；warmup 和测量范围区分开。
- [ ] 写 `run_nsys.sh`，对少量稳态迭代采集 CUDA/NVTX 时间线。
- [ ] 找到 CPU 等待、GPU 空闲、重复分配、多余同步与数据回传；保存截图并标注位置。
- [ ] 写 `run_ncu.sh`，仅采集自己的预处理 kernel，控制 launch 数量和指标集合。
- [ ] 用访存吞吐、访问模式和执行效率解释可能瓶颈；occupancy 不能单独代表性能高低。

以下假设你已在 C++ CLI 实现对应参数，运行前按本机帮助核对工具选项：

```bash
nsys profile --trace=cuda,nvtx --output=results/profiles/pipeline ./build/yolo11_infer --engine models/engines/yolo11n_static.engine --image data/images/sample.jpg --warmup 50 --iterations 100
ncu --set basic --launch-count 5 --export results/profiles/preprocess ./build/preprocess_benchmark
```

将示例图片替换为实际文件；第一次采集保持样本小，后续用 kernel 过滤参数选择目标。Profiling 会扰动时间，最终基准必须在不采集的正常运行下重测。

**验收：** 能指出一段具体时间线，解释“这里慢在哪里”；如果权限不支持某指标，报告清楚限制，不填写推测数字。

**产出：** `.nsys-rep`、`.ncu-rep`（环境支持时）、关键截图、至少三个观察和一个优先优化项。

### Day 21：只做一个有依据的优化并交付

- [ ] 从 Day 20 选择一个问题：重复分配、多余同步、多余拷贝、预处理访存等。
- [ ] 先记录假设与预期改善的计时区间，再修改代码。
- [ ] 复跑正确性、精度和正常模式基准；至少重复 3 轮，说明波动与收益。
- [ ] 写 `06_profile/analyze_benchmark.py` 生成对比表；未测项保留空值并标明原因。
- [ ] 完成 README：环境、数据、导出、构建、运行、验证、性能、已知限制、复现步骤。
- [ ] 写 `optimization_report.md`：问题 → 证据 → 改动 → 正确性 → 性能变化 → 适用范围。
- [ ] 从干净构建目录按 README 重走关键路径，确认没有依赖未记录的手工步骤。

**验收问题：** 你的优化改善了哪项指标？为什么有效？何时可能无效？能否在同样条件下复现？

**产出：** 可复现项目、完整基准/精度表、优化报告和最终 README。

## 13. 统一验收与 Benchmark 规范

### 13.1 正确性分四层

| 层次 | 检查对象 | 通过依据 |
|---|---|---|
| 输入 | 布局、颜色、尺寸、padding、归一化 | 同一 tensor 或明确的逐元素容差 |
| 原始输出 | 每个分支的 shape、语义、数值 | 有限值、分组误差、约定阈值 |
| 检测结果 | 解码、类别、坐标、NMS | 按类别与 IoU 匹配、边界样例通过 |
| 数据集 | 独立有标注验证集 | mAP 与任务接受门槛 |

任何层失败，先修该层，再继续下游。随机输入适合测接口与数值，不用于宣称真实检测质量。

### 13.2 时间必须标明边界

- GPU 模型执行：优先同一 stream 上使用 CUDA events，完成同步后读时间。
- CPU 前后处理：使用高分辨率 wall clock。
- 端到端：明确是否含读盘/图像解码；建议主表从已解码图片开始到最终检测结果可用，另列含读盘的业务时间。
- 异步流水线的各段可能重叠，不能简单相加当成总时间。
- 单张顺序延迟可换算近似 `1000 / latency_ms`；真实吞吐用完成图片数/总墙钟时间计算。batch>1 时标明 batch latency 与 images/s。
- ORT 的普通 `session.run()` 与 IO Binding 的传输边界不同；没有可靠 device-only 计时就标为 host API 耗时，别与 TensorRT GPU event 时间混在同列。
- 固定 GPU、电源状态、输入、batch、精度、数学模式和 warmup；记录后台负载与温度影响。构建耗时不计入稳态推理。

推荐 CSV 字段：

```text
backend,precision_graph,math_mode,gpu,trt_version,input_shape,batch,
timing_scope,includes_h2d,includes_d2h,warmup,iterations,repeat,
mean_ms,p50_ms,p90_ms,p99_ms,images_per_second,map50,map50_95
```

至少保存下表，全部填写实测数值，不预填“通常能达到”的成绩：

| 后端/路径 | 图精度 | 输入 | 模型执行 ms | E2E P50/P99 ms | images/s | mAP50-95 |
|---|---|---|---|---|---|---|
| PyTorch | FP32 | 1×3×640×640 | 待测 | 待测 | 待测 | 待测 |
| ORT CUDA | FP32 | 同上 | 待测 | 待测 | 待测 | 待测 |
| TRT Python | FP32 | 同上 | 待测 | 待测 | 待测 | 待测 |
| TRT C++ / CPU preprocess | FP32 | 同上 | 待测 | 待测 | 待测 | 待测 |
| TRT C++ / CUDA preprocess | FP32 | 同上 | 待测 | 待测 | 待测 | 待测 |
| TRT C++ / CUDA preprocess | FP16 | 同上 | 待测 | 待测 | 待测 | 待测 |
| TRT C++ / CUDA preprocess | INT8 Q/DQ | 同上 | 待测 | 待测 | 待测 | 待测 |

## 14. 常见阻塞与排查顺序

| 现象 | 优先检查 | 应保存的证据 |
|---|---|---|
| ONNX 有多个输出 | Rockchip 导出分支、DFL 是否外移、置信度汇总分支 | 输出契约和节点截图 |
| PT/ORT shape 不同 | 是否比较相同语义、输出 adapter、动态解码 | 两边结构与导出参数 |
| ORT 看见 CUDA 但执行失败 | CUDA/cuDNN 兼容、动态库、session 初始化日志 | 版本表、完整错误 |
| ORT 有 CPU 节点 | EP 分配与不支持算子，分清少量 shape 节点与主计算 | profiling 文件 |
| ONNX 合法但 TRT parse 失败 | 算子/opset/dtype、动态 shape、parser 错误位置 | 全部 parser 错误 |
| Python 有 TRT，C++ 编译失败 | 开发头文件/库安装、链接路径、版本是否匹配 | CMake 与 linker 日志 |
| 动态输出分配错误 | 是否先设置运行 shape，再查询 context shape | shape、dtype、字节数 |
| FP16 出现 NaN/Inf | 敏感操作、溢出、转换配置、IO dtype | 首个异常输出和配置 |
| INT8 检测下降 | 校准数据、预处理、量化范围、后处理 | 校准清单和分场景指标 |
| CUDA 预处理边缘不一致 | 像素中心、插值舍入、padding、stride | 差异图与边界输入 |
| GPU 计时异常低 | 是否遗漏同步或仅测入队 | event/stream 代码和时间线 |
| Nsight 无计数器权限 | WSL/驱动/GPU 支持及计数器授权 | 环境诊断与工具报错 |

阻塞超过一段专注时间后，缩小为最小复现：一份输入、一个模型、一个命令、完整日志。保留现有正确基线，每次只改变一个变量。

## 15. 每日复盘模板

把下面模板复制到 `docs/daily_log.md`，每天写一条：

```markdown
### Day N：主题
- 今日目标：
- 阅读的具体章节/代码函数：
- 新增或修改的文件及职责：
- 实际运行命令：
- 输入/模型/版本标识：
- 正确性结果与阈值：
- 性能结果及计时边界：
- 今天能独立回答的问题：
- 未解决问题、最小复现与下一步：
- 产出文件位置：
- 验收：通过 / 未通过
```

## 16. 最终检查清单与后续方向

- [ ] 能解释自己模型的全部输入输出，不依赖网上的固定 shape。
- [ ] 能手写并验证 toy ONNX，能解释 graph/node/initializer/opset。
- [ ] PT、ORT、TRT 在同一输入和同一输出语义下完成对齐。
- [ ] C++ 程序可以从图片得到最终检测结果，并妥善管理资源。
- [ ] CUDA 预处理通过边界测试，端到端变化有实测数据。
- [ ] 动态输入有合法范围、实际 shape 与越界验证。
- [ ] FP16/INT8 有转换或量化配置、校准记录和独立精度评估。
- [ ] Benchmark 标注计时边界、统计方法、硬件与软件版本。
- [ ] Nsight 报告支持至少一个具体优化结论；受限项目明确标注。
- [ ] README 足以让别人复现，失败或待完成项没有伪装为成果。

完成后再扩展 CUDA Graph、多流与双缓冲、GPU NMS、TensorRT plugin、量化感知训练、视频解码接入。与 RK3588 对照学习时，可比较“RGA 预处理/RKNN 推理”和“CUDA 预处理/TensorRT 推理”的数据流；dma-buf 与 CUDA device memory 的机制不同，应比较职责而不是视为相同 API。

下一次继续学习时，可以直接指定：“从这份路线的 Day N 开始，先检查我的环境与上一阶段产出，再逐文件实现并验收。”

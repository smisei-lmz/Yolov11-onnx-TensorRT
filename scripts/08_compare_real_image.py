import numpy as np
import cv2
import torch
import onnxruntime as ort
from ultralytics import YOLO

def letterbox(image_bgr:np.ndarray,new_shape=(640,640),padding_value=114,scaleup=True):
    src_h , src_w = image_bgr.shape[:2]
    target_h , target_w = new_shape
    r = min(target_h/src_h,target_w/src_w)
    if not scaleup:
        r = min(r,1.0)
    new_w = int(round(src_w*r))
    new_h = int(round(src_h*r))
    dw = target_w - new_w
    dh = target_h - new_h
    dw /= 2
    dh /= 2
    if (src_w != new_w or src_h != new_h):
        resized_img = cv2.resize(image_bgr,(new_w,new_h),interpolation=cv2.INTER_LINEAR)
    else:
        resized_img = image_bgr.copy()

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized_img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=( padding_value,padding_value,padding_value))
    meta = {
        "original_shape": (
            src_h,
            src_w
        ),
        "target_shape": (
            target_h,
            target_w
        ),
        "resized_shape": (
            new_h,
            new_w
        ),
        "scale": r,
        "pad_left": left,
        "pad_right": right,
        "pad_top": top,
        "pad_bottom": bottom,
    }
    return padded, meta

def preprocess_yolo(image_bgr: np.ndarray,new_shape=(640, 640)):
    letterboxed_bgr, meta = letterbox(
        image_bgr,
        new_shape=new_shape,
        padding_value=114,
        scaleup=True
    )
    rgb = cv2.cvtColor(letterboxed_bgr,cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb,(2,0,1))
    chw = np.ascontiguousarray(chw)
    tensor = chw.astype(np.float32)
    tensor /= 255.0
    tensor = np.expand_dims(tensor,axis=0)
    return (tensor,letterboxed_bgr,meta)



PT_PATH = "/home/smisei/Yolov11-ONNX-TensorRT/models/yolo11s.pt"
ONNX_PATH = "/home/smisei/Yolov11-ONNX-TensorRT/models/yolo11s_static_Nodynamic_Nosimplify_Nonms.onnx"
IMAGE_PATH = "/home/smisei/Yolov11-ONNX-TensorRT/ultralytics_yolo11-main/datasets/coco/images/val2017/000000000785.jpg"

def get_pt_raw_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output,(tuple, list)):
        if (len(output) > 0 and torch.is_tensor(output[0])):
            return output[0]
    raise RuntimeError(
        "无法识别PT输出结构，"
        "请先打印output结构，"
        "然后修改adapter。"
    )

def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(IMAGE_PATH)    
    x_np, _, meta = preprocess_yolo(image,new_shape=(640, 640))
    print("shape:",x_np.shape)
    print("dtype:",x_np.dtype)
    print("min:",x_np.min())
    print( "max:",x_np.max())
    print("scale:",meta["scale"])
    print("padding:",(meta["pad_left"],meta["pad_top"]))
    # =====================================
    # pytorch 
    # =====================================
    model = YOLO(PT_PATH).model
    model.cuda()
    model.float()
    model.eval()
    x_pt = torch.from_numpy(x_np).cuda()
    with torch.inference_mode():
        pt_output = model(x_pt)
    pt_raw = get_pt_raw_output(pt_output)
    pt_raw = (pt_raw.detach().float().cpu().numpy())
    print("PT output shape:",pt_raw.shape)
    # =====================================
    # ONNX Runtime
    # =====================================
    session = ort.InferenceSession(ONNX_PATH,providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    print("providers:",session.get_providers())
    input_name = session.get_inputs()[0].name
    ort_output = session.run(None,input_feed={input_name:x_np})
    ort_raw = ort_output[0]
    print("ORT output shape:",ort_raw.shape)
    diff = np.abs(pt_raw-ort_raw)
    max_error = (diff.max())
    mean_error = (diff.mean())
    median_error = (np.median(diff))   
    pt_flat = (
        pt_raw
        .astype(np.float64)
        .reshape(-1)
    )

    ort_flat = (
        ort_raw
        .astype(np.float64)
        .reshape(-1)
    )

    cosine = (
        np.dot(
            pt_flat,
            ort_flat
        )
        /
        (
            np.linalg.norm(
                pt_flat
            )
            *
            np.linalg.norm(
                ort_flat
            )
            +
            1e-12
        )
    )
    # =====================================
    # 8. allclose
    # =====================================

    allclose = np.allclose(
        pt_raw,
        ort_raw,

        rtol=1e-4,
        atol=1e-4
    )

    # =====================================
    # 9. 打印
    # =====================================

    print(
        "\n===== Comparison ====="
    )

    print(
        "max abs error:",
        max_error
    )

    print(
        "mean abs error:",
        mean_error
    )

    print(
        "median abs error:",
        median_error
    )

    print(
        "cosine similarity:",
        cosine
    )

    print(
        "allclose:",
        allclose
    )


if __name__ == "__main__":
    main()




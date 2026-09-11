import cv2
import numpy as np
import json
from pathlib import Path

IMAGE_PATH = "/home/smisei/Yolov11-ONNX-TensorRT/ultralytics_yolo11-main/datasets/coco/images/val2017/000000000785.jpg"

OUTPUT_NPY = "data/preprocessed_image.npy"

OUTPUT_IMAGE = "results/letterbox_image.jpg"

OUTPUT_META = "results/preprocess_meta.json"

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


def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {IMAGE_PATH}")
    print("===== Original Image =====")
    print("shape:",image.shape)
    print("dtype:",image.dtype)
    print("OpenCV layout: HWC")
    print("OpenCV color: BGR")   
    tensor, letterbox_img, meta = (preprocess_yolo(image,new_shape=(640, 640)))
    print(
        "\n===== LetterBox ====="
    )
    for key, value in meta.items():
        print( f"{key}: {value}")
    # ==========================================
    # 4. Tensor信息
    # ==========================================
    print(
        "\n===== YOLO Input Tensor ====="
    )
    print(
        "shape:",
        tensor.shape
    )
    print(
        "dtype:",
        tensor.dtype
    )
    print(
        "min:",
        tensor.min()
    )
    print(
        "max:",
        tensor.max()
    )
    print(
        "mean:",
        tensor.mean()
    )
    print(
        "C_CONTIGUOUS:",
        tensor.flags[
            "C_CONTIGUOUS"
        ]
    )

    # ==========================================
    # 5. 保存Tensor
    # ==========================================

    Path(
        OUTPUT_NPY
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUTPUT_NPY,
        tensor
    )

    # ==========================================
    # 6. 保存LetterBox后的图
    # ==========================================

    Path(
        OUTPUT_IMAGE
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        OUTPUT_IMAGE,
        letterbox_img
    )

    # ==========================================
    # 7. 保存metadata
    # ==========================================

    json_meta = {
        key: (
            list(value)
            if isinstance(value, tuple)
            else value
        )
        for key, value
        in meta.items()
    }

    with open(
        OUTPUT_META,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            json_meta,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_NPY
    )

    print(
        OUTPUT_IMAGE
    )

    print(
        OUTPUT_META
    )


if __name__ == "__main__":
    main()    
   
import numpy as np 
import cv2
import onnxruntime as ort
from pathlib import Path

ONNX_PATH = "/home/smisei/Yolov11-ONNX-TensorRT/models/yolo11s_static_Nodynamic_Nosimplify_Nonms.onnx"
IMAGE_PATH = "/home/smisei/Yolov11-ONNX-TensorRT/ultralytics_yolo11-main/datasets/coco/images/val2017/000000000785.jpg"

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



def xywh2xyxy(box):
    x = box.copy()
    x[...,0] = box[...,0] - box[...,2]/2
    x[...,1] = box[...,1] - box[...,3]/2
    x[...,2] = box[...,0] + box[...,2]/2
    x[...,3] = box[...,1] + box[...,3]/2
    return x

def nms(
    boxes,
    scores,
    iou_threshold=0.45
):

    """
    非极大值抑制
    """

    boxes_xywh = boxes.copy()
    boxes_xywh[:, 2:] -= boxes_xywh[:, :2]
    indices = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(),
        scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold
    )

    if len(indices)==0:
        return []

    return indices.flatten()



def scale_boxes(
    boxes,
    meta
):

    """
    LetterBox坐标
    映射回原图
    """


    scale = meta["scale"]

    pad_x = meta["pad_left"]

    pad_y = meta["pad_top"]


    boxes[:,[0,2]] -= pad_x
    boxes[:,[1,3]] -= pad_y


    boxes /= scale
    height, width = meta["original_shape"]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)


    return boxes


def postprocess(
    output,
    meta,
    conf_threshold=0.25,
    iou_threshold=0.45
):


    # --------------------
    # 1.
    # [1,84,8400]
    # ->
    # [8400,84]
    # --------------------

    # ORT 输出列表 -> [1,84,N] -> [84,N] -> [N,84]
    if output[0].ndim != 3 or output[0].shape[:2] != (1, 84):
        raise ValueError(f"期望 [1,84,N]，实际为 {output[0].shape}")
    pred = output[0][0].transpose(
        1,0
    )


    # --------------------
    # 2. bbox
    # --------------------

    boxes = pred[:,:4]


    boxes = xywh2xyxy(
        boxes
    )


    # --------------------
    # 3. YOLO11 没有独立 objectness，先用最大类别概率筛选。
    # --------------------

    obj = pred[:,4:].max(axis=1)


    mask = obj > conf_threshold


    boxes = boxes[mask]

    obj = obj[mask]

    pred = pred[mask]



    # --------------------
    # 4. class
    # --------------------

    cls_score = pred[:,4:]


    class_ids = np.argmax(
        cls_score,
        axis=1
    )


    cls_conf = np.max(
        cls_score,
        axis=1
    )


    scores = cls_conf


    # 再过滤一次

    mask = (
        scores >
        conf_threshold
    )


    boxes = boxes[mask]

    scores = scores[mask]

    class_ids = class_ids[mask]



    # --------------------
    # 5. NMS
    # --------------------


    # 分类别抑制，OpenCV NMS 的输入为左上角 xy + wh。
    keep = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        selected = nms(boxes[indices], scores[indices], iou_threshold)
        keep.extend(indices[selected].tolist())
    keep = np.asarray(keep, dtype=np.int64)
    keep = keep[np.argsort(-scores[keep])][:300]


    boxes = boxes[keep]

    scores = scores[keep]

    class_ids = class_ids[keep]


    # --------------------
    # 6. 映射回原图
    # --------------------

    boxes = scale_boxes(
        boxes,
        meta
    )


    return (
        boxes,
        scores,
        class_ids
    )


def main():


    image=cv2.imread(
        IMAGE_PATH
    )


    if image is None:
        raise FileNotFoundError(f"无法读取图片：{IMAGE_PATH}")
    tensor,_,meta = preprocess_yolo(
        image
    )


    ort.preload_dlls(directory="")
    session=ort.InferenceSession(
        ONNX_PATH,
        providers=[
            "CUDAExecutionProvider"
        ]
    )


    print("Session providers:", session.get_providers())
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("CUDA 后端加载失败，请检查上方日志")
    input_name=session.get_inputs()[0].name


    outputs=session.run(
        None,
        {
            input_name:tensor
        }
    )


    boxes,scores,classes=postprocess(
        outputs,
        meta
    )


    print(
        "detections:",
        len(boxes)
    )


    for box,score,cls in zip(
        boxes,
        scores,
        classes
    ):


        x1,y1,x2,y2=box.astype(int)


        cv2.rectangle(
            image,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )


        cv2.putText(
            image,
            f"{cls}:{score:.2f}",
            (x1,y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )


    result_path = Path(__file__).resolve().parents[1] / "results/detection_result.jpg"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(result_path), image):
        raise OSError(f"无法保存结果：{result_path}")
    print("Saved:", result_path)



if __name__=="__main__":
    main()

import numpy as np 
import cv2

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

    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
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

    pred = output[0].transpose(
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
    # 3. objectness
    # --------------------

    obj = pred[:,4]


    mask = obj > conf_threshold


    boxes = boxes[mask]

    obj = obj[mask]

    pred = pred[mask]



    # --------------------
    # 4. class
    # --------------------

    cls_score = pred[:,5:]


    class_ids = np.argmax(
        cls_score,
        axis=1
    )


    cls_conf = np.max(
        cls_score,
        axis=1
    )


    scores = (
        obj *
        cls_conf
    )


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


    keep = nms(
        boxes,
        scores,
        iou_threshold
    )


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



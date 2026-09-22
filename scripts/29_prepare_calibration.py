from pathlib import Path

import cv2
import numpy as np


CALIB_DIR = Path(
    "ultralytics_yolo11-main/datasets/coco/images/calibration"
)

OUTPUT_PATH = (
    "data/calibration.npy"
)

INPUT_H = 640
INPUT_W = 640

MAX_IMAGES = 100


def letterbox(
    image,
    new_shape=(640, 640),
    color=(114, 114, 114)
):

    src_h, src_w = (
        image.shape[:2]
    )

    dst_h, dst_w = new_shape

    scale = min(
        dst_w / src_w,
        dst_h / src_h
    )

    new_w = int(
        round(
            src_w * scale
        )
    )

    new_h = int(
        round(
            src_h * scale
        )
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    dw = (
        dst_w - new_w
    )

    dh = (
        dst_h - new_h
    )

    left = int(
        round(
            dw / 2.0 - 0.1
        )
    )

    right = int(
        round(
            dw / 2.0 + 0.1
        )
    )

    top = int(
        round(
            dh / 2.0 - 0.1
        )
    )

    bottom = int(
        round(
            dh / 2.0 + 0.1
        )
    )

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded


def preprocess(
    image
):

    image = letterbox(
        image,
        (
            INPUT_H,
            INPUT_W
        )
    )

    # BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # HWC -> CHW
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    image = np.ascontiguousarray(
        image,
        dtype=np.float32
    )

    image /= 255.0

    return image


def main():

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    paths = sorted(
        p
        for p in CALIB_DIR.rglob("*")
        if p.suffix.lower()
        in extensions
    )

    if not paths:

        raise RuntimeError(
            "No calibration images found."
        )

    paths = paths[
        :MAX_IMAGES
    ]

    print(
        "Calibration images:",
        len(paths)
    )

    calibration_data = []


    for i, path in enumerate(
        paths
    ):

        image = cv2.imread(
            str(path)
        )

        if image is None:

            print(
                "skip:",
                path
            )

            continue

        tensor = preprocess(
            image
        )

        calibration_data.append(
            tensor
        )

        if (
            (i + 1) % 50
            == 0
        ):

            print(
                f"{i + 1}/"
                f"{len(paths)}"
            )


    calibration_data = np.stack(
        calibration_data,
        axis=0
    )


    # 最终：
    # [N,3,640,640]

    print(
        "\n===== Calibration ====="
    )

    print(
        "shape:",
        calibration_data.shape
    )

    print(
        "dtype:",
        calibration_data.dtype
    )

    print(
        "min:",
        calibration_data.min()
    )

    print(
        "max:",
        calibration_data.max()
    )

    print(
        "mean:",
        calibration_data.mean()
    )


    np.save(
        OUTPUT_PATH,
        calibration_data
    )


    print(
        "\nsaved:",
        OUTPUT_PATH
    )


if __name__ == "__main__":

    main()
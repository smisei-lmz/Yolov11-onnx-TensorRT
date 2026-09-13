import numpy as np


ORT_PATH = (
    "results/ort_output_0.npy"
)

TRT_PATH = (
    "results/trt_output0.npy"
)


def main():

    ort = np.load(
        ORT_PATH
    )

    trt = np.load(
        TRT_PATH
    )

    print(
        "ORT:",
        ort.shape,
        ort.dtype
    )

    print(
        "TRT:",
        trt.shape,
        trt.dtype
    )

    if ort.shape != trt.shape:

        raise RuntimeError(
            "ORT / TRT shape不同"
        )

    diff = np.abs(
        ort - trt
    )

    print(
        "\n===== Comparison ====="
    )

    print(
        "max abs error:",
        diff.max()
    )

    print(
        "mean abs error:",
        diff.mean()
    )

    print(
        "median abs error:",
        np.median(diff)
    )

    a = (
        ort
        .astype(np.float64)
        .reshape(-1)
    )

    b = (
        trt
        .astype(np.float64)
        .reshape(-1)
    )

    cosine = (
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
            +
            1e-12
        )
    )

    print(
        "cosine similarity:",
        cosine
    )

    print(
        "allclose:",
        np.allclose(
            ort,
            trt,
            rtol=1e-4,
            atol=1e-4
        )
    )


if __name__ == "__main__":

    main()
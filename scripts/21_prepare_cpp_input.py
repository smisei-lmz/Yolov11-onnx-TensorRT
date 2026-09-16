import numpy as np


INPUT = "data/preprocessed_image.npy"
OUTPUT = "data/preprocessed_image.bin"


x = np.load(INPUT)

x = np.ascontiguousarray(
    x,
    dtype=np.float32
)

print("shape:", x.shape)
print("dtype:", x.dtype)
print("elements:", x.size)
print("bytes:", x.nbytes)

x.tofile(OUTPUT)

print("saved:", OUTPUT)
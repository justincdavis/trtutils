# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License

from __future__ import annotations

SCALE_SWAP_TRANSPOSE_KERNEL_CODE = """\
extern "C" __global__
void scaleSwapTranspose(
    const unsigned char* __restrict__ inputArr,
    float* outputArr,
    const float scale,
    const float offset,
    const int height,
    const int width
) {
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    const int tz = blockIdx.z * blockDim.z + threadIdx.z;
    if (tx < height && ty < width && tz < 3) {
        const int inputIdx = (tx * width * 3) + (ty * 3) + tz;
        const float val = static_cast<float>(inputArr[inputIdx]);
        const float scaledVal = val * scale + offset;
        const int dstChannel = 2 - tz;
        const int outputIdx = (dstChannel * height * width) + (tx * width) + ty;
        outputArr[outputIdx] = scaledVal;
    }
}
"""

extern "C" __global__
void scaleSwapTranspose(
    const unsigned char* __restrict__ inImg,
    float* __restrict__ outImg,
    const float scale,
    const float offset,
    const int shape
) {
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx < shape && ty < shape) {
        const int inputBaseIdx = (tx * shape + ty) * 3;

        // scale BGR values to new range
        const float b = static_cast<float>(inImg[inputBaseIdx + 0]) * scale + offset;
        const float g = static_cast<float>(inImg[inputBaseIdx + 1]) * scale + offset;
        const float r = static_cast<float>(inImg[inputBaseIdx + 2]) * scale + offset;

        const int shape_squared = shape * shape;

        // swap BGR to RGB for YOLO inference
        const int outputBaseIdx = (tx * shape + ty);
        outImg[outputBaseIdx + 0 * shape_squared] = r;
        outImg[outputBaseIdx + 1 * shape_squared] = g;
        outImg[outputBaseIdx + 2 * shape_squared] = b;
    }
}

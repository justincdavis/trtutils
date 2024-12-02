extern "C" __global__
void scaleSwapTranspose(
    const unsigned char* __restrict__ inImg,
    float* __restrict__ outImg,
    const float scale,
    const float offset,
    const int height,
    const int width
) {
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx < height && ty < width) {
        const int inputBaseIdx = (tx * width + ty) * 3;

        // scale BGR values to new range
        float b = static_cast<float>(inImg[inputBaseIdx + 0]) * scale + offset;
        float g = static_cast<float>(inImg[inputBaseIdx + 1]) * scale + offset;
        float r = static_cast<float>(inImg[inputBaseIdx + 2]) * scale + offset;

        // swap BGR to RGB for YOLO inference
        int outputBaseIdx = (tx * width + ty);
        outImg[outputBaseIdx + 0 * height * width] = r;
        outImg[outputBaseIdx + 1 * height * width] = g;
        outImg[outputBaseIdx + 2 * height * width] = b;
    }
}

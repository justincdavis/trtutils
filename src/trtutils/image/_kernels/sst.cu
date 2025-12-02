extern "C" __global__
void scaleSwapTranspose(
    const unsigned char* __restrict__ inImg,
    float* __restrict__ outImg,
    const float scale,
    const float offset,
    const int height,
    const int width,
    const int batch_size
) {
    const int batch_idx = blockIdx.z;
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;

    // compute batch offsets
    const int pixels_per_image = height * width;
    const unsigned char* batchInImg = inImg + batch_idx * pixels_per_image * 3;
    float* batchOutImg = outImg + batch_idx * 3 * pixels_per_image;

    if (tx < height && ty < width) {
        const int inputBaseIdx = (tx * width + ty) * 3;

        // scale BGR values to new range
        const float b = static_cast<float>(batchInImg[inputBaseIdx + 0]) * scale + offset;
        const float g = static_cast<float>(batchInImg[inputBaseIdx + 1]) * scale + offset;
        const float r = static_cast<float>(batchInImg[inputBaseIdx + 2]) * scale + offset;

        // swap BGR to RGB for YOLO inference
        const int outputBaseIdx = (tx * width + ty);
        batchOutImg[outputBaseIdx + 0 * pixels_per_image] = r;
        batchOutImg[outputBaseIdx + 1 * pixels_per_image] = g;
        batchOutImg[outputBaseIdx + 2 * pixels_per_image] = b;
    }
}

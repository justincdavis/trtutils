extern "C" __global__
void rgbToBgr(
    const unsigned char* __restrict__ inImg,
    unsigned char* __restrict__ outImg,
    const int width,
    const int height,
    const int inPitch
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const unsigned char* src = inImg + (size_t)y * inPitch + (size_t)x * 3;
    unsigned char* dst = outImg + ((size_t)y * width + x) * 3;
    dst[0] = src[2];
    dst[1] = src[1];
    dst[2] = src[0];
}

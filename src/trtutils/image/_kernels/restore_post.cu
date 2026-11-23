// Restored images to uint8 BGR: one thread per output pixel.

// clamp to [0, 1], scale to 0-255 and round half to even (numpy's rint, as the CPU
// postprocessor does), RGB CHW to BGR HWC, keeping the top-left (outH, outW) of the image
extern "C" __global__
void restoreToBgr(
    const float* __restrict__ input,
    const int height,
    const int width,
    const int outH,
    const int outW,
    unsigned char* __restrict__ output
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= outW || y >= outH) return;
    const size_t plane = (size_t)height * width;
    const float* src = input + (size_t)y * width + x;
    unsigned char* dst = output + ((size_t)y * outW + x) * 3;
    for (int c = 0; c < 3; ++c) {
        const float v = fminf(fmaxf(src[(2 - c) * plane], 0.0f), 1.0f) * 255.0f;
        dst[c] = (unsigned char)__float2int_rn(v);
    }
}

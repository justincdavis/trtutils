// Antialiased resize: per axis, a tent (triangle) filter as wide as the downscale factor,
// renormalized at the borders, as PIL's BILINEAR and torchvision's antialias=True compute.
// On an axis that is not downscaled the weights are bilinear interpolation's.
// Same arguments and fill as letterboxResize: the region is where the resized image lands.

struct TentSpan {
    int first;
    int stop;
    float center;
    float inverseSupport;
    float norm;
};

__device__ __forceinline__ float tent(const float x) {
    return fmaxf(0.0f, 1.0f - fabsf(x));
}

// the source pixels [first, stop) of output pixel o, as PIL's precompute_coeffs
__device__ __forceinline__ TentSpan tentSpan(const int o, const int sizeIn, const float scale) {
    TentSpan span;
    const float support = fmaxf(scale, 1.0f);
    span.center = (o + 0.5f) * scale;
    // int() truncates toward zero, as in PIL
    span.first = max((int)(span.center - support + 0.5f), 0);
    span.stop = min((int)(span.center + support + 0.5f), sizeIn);
    span.inverseSupport = 1.0f / support;
    float sum = 0.0f;
    for (int i = span.first; i < span.stop; ++i) {
        sum += tent((i + 0.5f - span.center) * span.inverseSupport);
    }
    span.norm = sum > 0.0f ? 1.0f / sum : 0.0f;
    return span;
}

extern "C" __global__
void tentResize(
    const unsigned char* __restrict__ inImg,
    unsigned char* __restrict__ outImg,
    const int widthIn,
    const int heightIn,
    const int widthOut,
    const int heightOut,
    const int startX,
    const int startY,
    const int regionWidth,
    const int regionHeight,
    const int inputStride,
    const int outputStride
) {
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= widthOut || ty >= heightOut) return;

    const unsigned char* inputBase = inImg + (size_t)blockIdx.z * inputStride;
    unsigned char* output = outImg + (size_t)blockIdx.z * outputStride + (ty * widthOut + tx) * 3;

    if (tx < startX || tx >= startX + regionWidth || ty < startY || ty >= startY + regionHeight) {
        output[0] = 114;
        output[1] = 114;
        output[2] = 114;
        return;
    }

    const TentSpan sx = tentSpan(tx - startX, widthIn, widthIn / (float)regionWidth);
    const TentSpan sy = tentSpan(ty - startY, heightIn, heightIn / (float)regionHeight);

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    for (int y = sy.first; y < sy.stop; ++y) {
        const float wy = tent((y + 0.5f - sy.center) * sy.inverseSupport);
        const unsigned char* row = inputBase + (size_t)y * widthIn * 3;
        float rowR = 0.0f;
        float rowG = 0.0f;
        float rowB = 0.0f;
        for (int x = sx.first; x < sx.stop; ++x) {
            const float wx = tent((x + 0.5f - sx.center) * sx.inverseSupport);
            const unsigned char* pixel = row + x * 3;
            rowR += wx * pixel[0];
            rowG += wx * pixel[1];
            rowB += wx * pixel[2];
        }
        r += wy * rowR;
        g += wy * rowG;
        b += wy * rowB;
    }
    const float norm = sx.norm * sy.norm;
    output[0] = (unsigned char)fminf(fmaxf(r * norm + 0.5f, 0.0f), 255.0f);
    output[1] = (unsigned char)fminf(fmaxf(g * norm + 0.5f, 0.0f), 255.0f);
    output[2] = (unsigned char)fminf(fmaxf(b * norm + 0.5f, 0.0f), 255.0f);
}

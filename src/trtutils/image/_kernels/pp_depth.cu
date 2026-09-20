// Depth map min-max normalization to [0, 1]: (B, H*W) treated as 2D rows.
// depthMinMax: one block per image reduces per-image min/max.
// depthNormalize: in-place elementwise (x - min) / (max - min + 1e-8).

#define FLT_MAX 3.402823466e+38f

extern "C" __global__
void depthMinMax(
    const float* __restrict__ d,    // (B, H*W)
    float* __restrict__ minmax,     // (B, 2)
    const int hwp
) {
    const int b = blockIdx.x;
    const float* row = d + (size_t)b * hwp;

    __shared__ float s_min[1024];
    __shared__ float s_max[1024];

    float mn = FLT_MAX;
    float mx = -FLT_MAX;
    for (int j = threadIdx.x; j < hwp; j += blockDim.x) {
        const float v = row[j];
        mn = fminf(mn, v);
        mx = fmaxf(mx, v);
    }
    s_min[threadIdx.x] = mn;
    s_max[threadIdx.x] = mx;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_min[threadIdx.x] = fminf(s_min[threadIdx.x], s_min[threadIdx.x + stride]);
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        minmax[b * 2] = s_min[0];
        minmax[b * 2 + 1] = s_max[0];
    }
}

extern "C" __global__
void depthNormalize(
    float* __restrict__ d,          // (B, H*W), in/out
    const float* __restrict__ minmax,  // (B, 2)
    const int batch,
    const int hwp
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * hwp) return;

    const int b = idx / hwp;
    const float mn = minmax[b * 2];
    const float mx = minmax[b * 2 + 1];
    d[idx] = (d[idx] - mn) * (1.0f / (mx - mn + 1e-8f));
}

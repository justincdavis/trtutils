// Per-row softmax: in (B, C) -> out (B, C). One block per row, grid-stride
// over C so it also works for C > 1024.

#define FLT_MAX 3.402823466e+38f

extern "C" __global__
void softmaxRows(
    const float* __restrict__ in,  // (B, C)
    float* __restrict__ out,       // (B, C)
    const int c
) {
    const int b = blockIdx.x;
    const float* row_in = in + (size_t)b * c;
    float* row_out = out + (size_t)b * c;

    __shared__ float s_val[1024];

    float mx = -FLT_MAX;
    for (int j = threadIdx.x; j < c; j += blockDim.x) {
        mx = fmaxf(mx, row_in[j]);
    }
    s_val[threadIdx.x] = mx;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_val[threadIdx.x] = fmaxf(s_val[threadIdx.x], s_val[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_val = s_val[0];

    float sum = 0.0f;
    for (int j = threadIdx.x; j < c; j += blockDim.x) {
        sum += expf(row_in[j] - max_val);
    }
    s_val[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_val[threadIdx.x] += s_val[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv = 1.0f / s_val[0];

    for (int j = threadIdx.x; j < c; j += blockDim.x) {
        row_out[j] = expf(row_in[j] - max_val) * inv;
    }
}

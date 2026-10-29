// OCR postprocessing: CTC greedy decode and DB probability map binarization.

#define CTC_THREADS 256

// per (batch, timestep) row of C classes: first index of the max value and the value itself
extern "C" __global__
void ctcArgmax(
    const float* __restrict__ probs,
    const int C,
    int* __restrict__ outIdx,
    float* __restrict__ outProb
) {
    const size_t row = blockIdx.x;
    const float* p = probs + row * C;
    float best = p[0];
    int bestIdx = 0;
    for (int c = threadIdx.x; c < C; c += CTC_THREADS) {
        // strictly greater keeps the lowest index on ties, as np.argmax
        if (p[c] > best) {
            best = p[c];
            bestIdx = c;
        }
    }
    __shared__ float sVal[CTC_THREADS];
    __shared__ int sIdx[CTC_THREADS];
    sVal[threadIdx.x] = best;
    sIdx[threadIdx.x] = bestIdx;
    __syncthreads();
    for (int s = CTC_THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float v = sVal[threadIdx.x + s];
            const int i = sIdx[threadIdx.x + s];
            if (v > sVal[threadIdx.x] || (v == sVal[threadIdx.x] && i < sIdx[threadIdx.x])) {
                sVal[threadIdx.x] = v;
                sIdx[threadIdx.x] = i;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        outIdx[row] = sIdx[0];
        outProb[row] = sVal[0];
    }
}

// bitmap = probs > thresh as 0/1 bytes
extern "C" __global__
void dbBinarize(
    const float* __restrict__ probs,
    const size_t n,
    const float thresh,
    unsigned char* __restrict__ bitmap
) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) bitmap[i] = probs[i] > thresh;
}

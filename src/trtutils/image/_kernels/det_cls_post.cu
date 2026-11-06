// Detection unletterbox and classification softmax. One thread per image for
// detection (a few hundred boxes); one block per image for the class vector.

#include "postprocess.cuh"

// keep the rows scoring at least conf as unletterboxed (x1, y1, x2, y2, score, class) rows,
// in row order. Strided inputs serve both YOLO-v10 (batch, n, 6) rows and the EfficientNMS
// outputs; numDets (null for all n rows) bounds each image's rows
extern "C" __global__
void detUnletterbox(
    const float* __restrict__ boxes,
    const int boxStride,
    const float* __restrict__ scores,
    const int scoreStride,
    const void* __restrict__ classes,
    const int classStride,
    const int classKind,
    const void* __restrict__ numDets,
    const int numKind,
    const int n,
    const float* __restrict__ letterbox,
    const float conf,
    float* __restrict__ dets,
    int* __restrict__ counts
) {
    const int image = blockIdx.x;
    if (threadIdx.x != 0) return;
    const int limit = numDets ? min(max(loadIndex(numDets, numKind, image), 0), n) : n;
    const size_t row = (size_t)image * n;
    counts[image] = keepUnletterboxed(
        boxes + row * boxStride,
        boxStride,
        scores + row * scoreStride,
        scoreStride,
        classes,
        row * classStride,
        classStride,
        classKind,
        limit,
        loadLetterbox(letterbox, image),
        conf,
        dets + row * 6,
        nullptr
    );
}

#define CLS_THREADS 256

extern "C" __global__
void clsSoftmax(
    const float* __restrict__ logits,
    const int c,
    const int doSoftmax,
    float* __restrict__ probs
) {
    const float* row = logits + (size_t)blockIdx.x * c;
    float* out = probs + (size_t)blockIdx.x * c;
    __shared__ float red[CLS_THREADS];
    float tmax = -1e30f;
    for (int i = threadIdx.x; i < c; i += CLS_THREADS) tmax = fmaxf(tmax, row[i]);
    red[threadIdx.x] = tmax;
    __syncthreads();
    for (int s = CLS_THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    tmax = red[0];
    float sum = 0.f;
    if (doSoftmax) {
        for (int i = threadIdx.x; i < c; i += CLS_THREADS) sum += expf(row[i] - tmax);
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    if (doSoftmax) {
        for (int s = CLS_THREADS / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        sum = red[0];
        for (int i = threadIdx.x; i < c; i += CLS_THREADS) out[i] = expf(row[i] - tmax) / sum;
    } else {
        for (int i = threadIdx.x; i < c; i += CLS_THREADS) out[i] = row[i];
    }
}

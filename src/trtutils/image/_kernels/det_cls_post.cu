// Detection unletterbox and classification softmax. One thread per image for
// detection (a few hundred boxes); one block per image for the class vector.

extern "C" __global__
void unletterboxV10(
    const float* __restrict__ rows,
    const int n,
    const float* __restrict__ letter,
    const float conf,
    float* __restrict__ boxes,
    float* __restrict__ scores,
    int* __restrict__ classes,
    int* __restrict__ counts
) {
    const int image = blockIdx.x;
    if (threadIdx.x != 0) return;
    const float rx = letter[image * 6 + 0];
    const float ry = letter[image * 6 + 1];
    const float px = letter[image * 6 + 2];
    const float py = letter[image * 6 + 3];
    const float width = letter[image * 6 + 4];
    const float height = letter[image * 6 + 5];
    const float* in = rows + (size_t)image * n * 6;
    float* outb = boxes + (size_t)image * n * 4;
    float* outs = scores + (size_t)image * n;
    int* outc = classes + (size_t)image * n;
    int kept = 0;
    for (int j = 0; j < n; ++j) {
        const float score = in[j * 6 + 4];
        if (score < conf) continue;
        float x1 = (in[j * 6 + 0] - px) / rx;
        float y1 = (in[j * 6 + 1] - py) / ry;
        float x2 = (in[j * 6 + 2] - px) / rx;
        float y2 = (in[j * 6 + 3] - py) / ry;
        x1 = fminf(fmaxf(x1, 0.f), width);
        y1 = fminf(fmaxf(y1, 0.f), height);
        x2 = fminf(fmaxf(x2, 0.f), width);
        y2 = fminf(fmaxf(y2, 0.f), height);
        outb[kept * 4 + 0] = x1;
        outb[kept * 4 + 1] = y1;
        outb[kept * 4 + 2] = x2;
        outb[kept * 4 + 3] = y2;
        outs[kept] = score;
        outc[kept] = (int)in[j * 6 + 5];
        ++kept;
    }
    counts[image] = kept;
}

extern "C" __global__
void unletterboxEff(
    const void* __restrict__ numDets,
    const int numIs64,
    const float* __restrict__ inBoxes,
    const float* __restrict__ inScores,
    const void* __restrict__ inClasses,
    const int clsIsFloat,
    const int m,
    const float* __restrict__ letter,
    const float conf,
    float* __restrict__ boxes,
    float* __restrict__ scores,
    int* __restrict__ classes,
    int* __restrict__ counts
) {
    const int image = blockIdx.x;
    if (threadIdx.x != 0) return;
    int limit = numIs64 ? (int)((const long long*)numDets)[image] : ((const int*)numDets)[image];
    if (limit > m) limit = m;
    if (limit < 0) limit = 0;
    const float rx = letter[image * 6 + 0];
    const float ry = letter[image * 6 + 1];
    const float px = letter[image * 6 + 2];
    const float py = letter[image * 6 + 3];
    const float width = letter[image * 6 + 4];
    const float height = letter[image * 6 + 5];
    const float* ib = inBoxes + (size_t)image * m * 4;
    const float* isc = inScores + (size_t)image * m;
    float* outb = boxes + (size_t)image * m * 4;
    float* outs = scores + (size_t)image * m;
    int* outc = classes + (size_t)image * m;
    int kept = 0;
    for (int j = 0; j < limit; ++j) {
        const float score = isc[j];
        if (score < conf) continue;
        float x1 = (ib[j * 4 + 0] - px) / rx;
        float y1 = (ib[j * 4 + 1] - py) / ry;
        float x2 = (ib[j * 4 + 2] - px) / rx;
        float y2 = (ib[j * 4 + 3] - py) / ry;
        x1 = fminf(fmaxf(x1, 0.f), width);
        y1 = fminf(fmaxf(y1, 0.f), height);
        x2 = fminf(fmaxf(x2, 0.f), width);
        y2 = fminf(fmaxf(y2, 0.f), height);
        int cls;
        if (clsIsFloat) cls = (int)((const float*)inClasses)[(size_t)image * m + j];
        else cls = ((const int*)inClasses)[(size_t)image * m + j];
        outb[kept * 4 + 0] = x1;
        outb[kept * 4 + 1] = y1;
        outb[kept * 4 + 2] = x2;
        outb[kept * 4 + 3] = y2;
        outs[kept] = score;
        outc[kept] = cls;
        ++kept;
    }
    counts[image] = kept;
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

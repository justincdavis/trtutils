// YOLO-style head postprocessing: decode, sort, and class-aware NMS.
// Every kernel takes the batch index from the last grid dimension.
// The head is (B, C, N): rows 0-3 cxcywh, rows [clsStart, clsStart + nc) class scores.

#define NMS_MAX_WORDS 1024  // NMS supports up to 64 * NMS_MAX_WORDS candidates
// keep anchors whose best class score is above conf, in any order
extern "C" __global__
void yoloDecode(
    const float* __restrict__ out,
    const int C,
    const int N,
    const int clsStart,
    const int nc,
    const float conf,
    const int inclusive,  // keep scores >= conf instead of > conf
    int* __restrict__ count,
    int* __restrict__ candAnchor,
    float* __restrict__ candScore,
    int* __restrict__ candCls,
    int* __restrict__ candRank
) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (a >= N) return;
    const float* cls = out + ((size_t)b * C + clsStart) * N + a;
    float best = cls[0];
    int bestCls = 0;
    for (int c = 1; c < nc; ++c) {
        const float s = cls[(size_t)c * N];
        if (s > best) {
            best = s;
            bestCls = c;
        }
    }
    // opencv NMS keeps scores strictly above its threshold
    if (!(inclusive ? best >= conf : best > conf)) return;
    const int slot = atomicAdd(count + b, 1);
    candAnchor[(size_t)b * N + slot] = a;
    candScore[(size_t)b * N + slot] = best;
    candCls[(size_t)b * N + slot] = bestCls;
    candRank[(size_t)b * N + slot] = 0;
}

#define RANK_SLICES 16

// count the candidates that sort before each candidate (score, ties by anchor: a stable
// sort of the anchor order); a block per 256 candidates and slice of the others, so a few
// thousand candidates still fill the GPU
extern "C" __global__
void yoloRank(
    const int N,
    const int* __restrict__ count,
    const int* __restrict__ candAnchor,
    const float* __restrict__ candScore,
    int* __restrict__ candRank
) {
    __shared__ float tileScore[256];
    __shared__ int tileAnchor[256];
    const int tile = blockIdx.x / RANK_SLICES;
    const int slice = blockIdx.x % RANK_SLICES;
    const int b = blockIdx.y;
    const int n = count[b];
    if (tile * (int)blockDim.x >= n) return;
    const int perSlice = (n + RANK_SLICES - 1) / RANK_SLICES;
    const int sliceLen = (perSlice + blockDim.x - 1) / blockDim.x * blockDim.x;
    const int jStart = slice * sliceLen;
    if (jStart >= n) return;
    const int jEnd = min(n, jStart + sliceLen);
    const size_t base = (size_t)b * N;
    const int i = tile * blockDim.x + threadIdx.x;
    const bool active = i < n;
    const float s = active ? candScore[base + i] : 0.0f;
    const int a = active ? candAnchor[base + i] : 0;
    int rank = 0;
    for (int start = jStart; start < jEnd; start += blockDim.x) {
        const int j = start + threadIdx.x;
        if (j < jEnd) {
            tileScore[threadIdx.x] = candScore[base + j];
            tileAnchor[threadIdx.x] = candAnchor[base + j];
        }
        __syncthreads();
        const int len = min((int)blockDim.x, jEnd - start);
        for (int t = 0; t < len; ++t) {
            const float sj = tileScore[t];
            rank += (sj > s) || (sj == s && tileAnchor[t] < a);
        }
        __syncthreads();
    }
    if (active && rank) atomicAdd(candRank + base + i, rank);
}

// write each candidate at its rank, with its xyxy box
extern "C" __global__
void yoloScatter(
    const float* __restrict__ out,
    const int C,
    const int N,
    const int* __restrict__ count,
    const int* __restrict__ candAnchor,
    const float* __restrict__ candScore,
    const int* __restrict__ candCls,
    const int* __restrict__ candRank,
    int* __restrict__ sortAnchor,
    float* __restrict__ sortScore,
    int* __restrict__ sortCls,
    float* __restrict__ sortBox
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (i >= count[b]) return;
    const size_t base = (size_t)b * N;
    const int a = candAnchor[base + i];
    const int rank = candRank[base + i];
    sortAnchor[base + rank] = a;
    sortScore[base + rank] = candScore[base + i];
    sortCls[base + rank] = candCls[base + i];
    const float* box = out + (size_t)b * C * N + a;
    const float cx = box[0];
    const float cy = box[N];
    const float hw = box[2 * (size_t)N] / 2.0f;
    const float hh = box[3 * (size_t)N] / 2.0f;
    float* dst = sortBox + (base + rank) * 4;
    dst[0] = cx - hw;
    dst[1] = cy - hh;
    dst[2] = cx + hw;
    dst[3] = cy + hh;
}

// bit j of row i is set when sorted box j > i overlaps box i of the same class;
// a block per 64 rows and stripe of column blocks, so empty launches stay cheap.
// N is the row capacity of mask: images with more candidates are left to the reduce
extern "C" __global__
void yoloNmsMask(
    const int N,
    const int* __restrict__ count,
    const int* __restrict__ sortCls,
    const float* __restrict__ sortBox,
    const float iouThres,
    unsigned long long* __restrict__ mask
) {
    __shared__ float colBox[64 * 4];
    __shared__ int colCls[64];
    const int b = blockIdx.z;
    const int n = count[b];
    const int rowStart = blockIdx.x * 64;
    if (rowStart >= n || n > N) return;
    const size_t base = (size_t)b * N;
    const int words = (N + 63) / 64;
    const int i = rowStart + threadIdx.x;
    const bool active = i < n;
    // opencv computes IoU on double (x, y, w, h) rects built from float32 xywh
    double ax = 0.0, ay = 0.0, aw = 0.0, ah = 0.0;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    int ci = -1;
    if (active) {
        const float* bi = sortBox + (base + i) * 4;
        a0 = bi[0];
        a1 = bi[1];
        a2 = bi[2];
        a3 = bi[3];
        ax = bi[0];
        ay = bi[1];
        aw = (double)(bi[2] - bi[0]);
        ah = (double)(bi[3] - bi[1]);
        ci = sortCls[base + i];
    }
    // column blocks at or after the row block, striped over blockIdx.y
    for (int colStart = rowStart + 64 * blockIdx.y; colStart < n; colStart += 64 * gridDim.y) {
        const int cols = min(64, n - colStart);
        __syncthreads();
        if ((int)threadIdx.x < cols) {
            const int j = colStart + threadIdx.x;
            for (int k = 0; k < 4; ++k) colBox[threadIdx.x * 4 + k] = sortBox[(base + j) * 4 + k];
            colCls[threadIdx.x] = sortCls[base + j];
        }
        __syncthreads();
        if (!active) continue;
        unsigned long long bits = 0;
        const int first = (colStart == rowStart) ? threadIdx.x + 1 : 0;
        for (int t = first; t < cols; ++t) {
            if (colCls[t] != ci) continue;
            const float* bj = colBox + t * 4;
            // float64 is slow on GeForce parts: decide in float32 when the pair is clearly
            // apart or its IoU clearly off the threshold. The margins sit far above the
            // rounding between these corners and opencv's double rects; the rest go exact.
            const float mx = 1e-4f * (fabsf(a2) + fabsf(bj[2]) + 1.0f);
            const float my = 1e-4f * (fabsf(a3) + fabsf(bj[3]) + 1.0f);
            const float fiw = fminf(a2, bj[2]) - fmaxf(a0, bj[0]);
            const float fih = fminf(a3, bj[3]) - fmaxf(a1, bj[1]);
            if (fiw < -mx || fih < -my) continue;
            if (fiw > mx && fih > my) {
                const float inter = fiw * fih;
                const float iou = inter / ((a2 - a0) * (a3 - a1) + (bj[2] - bj[0]) * (bj[3] - bj[1]) - inter);
                if (iou > iouThres + 2e-3f) {
                    bits |= 1ULL << t;
                    continue;
                }
                if (iou < iouThres - 2e-3f) continue;
            }
            const double bx = bj[0];
            const double by = bj[1];
            const double bw = (double)(bj[2] - bj[0]);
            const double bh = (double)(bj[3] - bj[1]);
            const double iw = fmin(ax + aw, bx + bw) - fmax(ax, bx);
            const double ih = fmin(ay + ah, by + bh) - fmax(ay, by);
            if (iw <= 0.0 || ih <= 0.0) continue;
            const double inter = iw * ih;
            const double uni = aw * ah + bw * bh - inter;
            if (uni > 0.0 && inter / uni > iouThres) bits |= 1ULL << t;
        }
        mask[(base + i) * words + colStart / 64] = bits;
    }
}

// greedy pass over the sorted boxes, one warp per image (the loop syncs once per kept box);
// keepCount is -1 when the image has more candidates than mask holds (N rows)
extern "C" __global__
void yoloNmsReduce(
    const int N,
    const int maxDet,
    const int* __restrict__ count,
    const unsigned long long* __restrict__ mask,
    int* __restrict__ keep,
    int* __restrict__ keepCount
) {
    __shared__ unsigned long long removed[NMS_MAX_WORDS];
    const int b = blockIdx.x;
    const int n = count[b];
    if (n > N) {
        if (threadIdx.x == 0) keepCount[b] = -1;
        return;
    }
    const int words = (N + 63) / 64;
    const int used = (n + 63) / 64;
    for (int w = threadIdx.x; w < used; w += blockDim.x) removed[w] = 0;
    __syncwarp();
    const size_t base = (size_t)b * N;
    int kept = 0;
    for (int w = 0; w < used && kept < maxDet; ++w) {
        // live boxes of word w, jumping straight from one survivor to the next
        const unsigned long long tail = (w == used - 1 && n % 64) ? (1ULL << (n % 64)) - 1 : ~0ULL;
        unsigned long long live = ~removed[w] & tail;
        while (live && kept < maxDet) {
            const int bit = __ffsll(live) - 1;
            const int i = w * 64 + bit;
            if (threadIdx.x == 0) keep[(size_t)b * maxDet + kept] = i;
            ++kept;
            const unsigned long long* row = mask + (base + i) * words;
            // only words at or after i hold bits for later boxes
            for (int x = w + threadIdx.x; x < used; x += blockDim.x) removed[x] |= row[x];
            __syncwarp();
            live = ~removed[w] & tail & ~((2ULL << bit) - 1);
        }
    }
    if (threadIdx.x == 0) keepCount[b] = kept;
}

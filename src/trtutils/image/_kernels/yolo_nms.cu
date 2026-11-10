// YOLO-style head postprocessing: decode, sort, class-aware NMS, and gather of the kept rows.
// Every kernel takes the batch index from the last grid dimension.
// The head is (B, C, N): rows 0-3 cxcywh, rows [clsStart, clsStart + nc) class scores.

#include "postprocess.cuh"

#define NMS_MAX_ROWS 4096  // _NMS_ROWS in _cuda_nms.py

// whether box b overlaps box a (both x1, y1, x2, y2) by IoU above iouThres, as opencv's NMS,
// which computes IoU on double (x, y, w, h) rects built from float32 xywh
__device__ __forceinline__ bool nmsOverlap(const float* a, const float* b, const float iouThres) {
    // float64 is slow on GeForce parts: decide in float32 when the pair is clearly
    // apart or its IoU clearly off the threshold. The margins sit far above the
    // rounding between these corners and opencv's double rects; the rest go exact.
    const float mx = 1e-4f * (fabsf(a[2]) + fabsf(b[2]) + 1.0f);
    const float my = 1e-4f * (fabsf(a[3]) + fabsf(b[3]) + 1.0f);
    const float fiw = fminf(a[2], b[2]) - fmaxf(a[0], b[0]);
    const float fih = fminf(a[3], b[3]) - fmaxf(a[1], b[1]);
    if (fiw < -mx || fih < -my) return false;
    if (fiw > mx && fih > my) {
        const float inter = fiw * fih;
        const float iou = inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
        if (iou > iouThres + 2e-3f) return true;
        if (iou < iouThres - 2e-3f) return false;
    }
    const double ax = a[0];
    const double ay = a[1];
    const double aw = (double)(a[2] - a[0]);
    const double ah = (double)(a[3] - a[1]);
    const double bx = b[0];
    const double by = b[1];
    const double bw = (double)(b[2] - b[0]);
    const double bh = (double)(b[3] - b[1]);
    const double iw = fmin(ax + aw, bx + bw) - fmax(ax, bx);
    const double ih = fmin(ay + ah, by + bh) - fmax(ay, by);
    if (iw <= 0.0 || ih <= 0.0) return false;
    const double inter = iw * ih;
    const double uni = aw * ah + bw * bh - inter;
    return uni > 0.0 && inter / uni > iouThres;
}

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
// N is the row capacity of mask: images with more candidates are left to the reduce.
// stride is the batch stride of sortCls/sortBox (the anchor count), which can exceed N.
extern "C" __global__
void yoloNmsMask(
    const int N,
    const int stride,
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
    const size_t sortBase = (size_t)b * stride;
    const size_t maskBase = (size_t)b * N;
    const int words = (N + 63) / 64;
    const int i = rowStart + threadIdx.x;
    const bool active = i < n;
    float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int ci = -1;
    if (active) {
        for (int k = 0; k < 4; ++k) a[k] = sortBox[(sortBase + i) * 4 + k];
        ci = sortCls[sortBase + i];
    }
    // column blocks at or after the row block, striped over blockIdx.y
    for (int colStart = rowStart + 64 * blockIdx.y; colStart < n; colStart += 64 * gridDim.y) {
        const int cols = min(64, n - colStart);
        __syncthreads();
        if ((int)threadIdx.x < cols) {
            const int j = colStart + threadIdx.x;
            for (int k = 0; k < 4; ++k) colBox[threadIdx.x * 4 + k] = sortBox[(sortBase + j) * 4 + k];
            colCls[threadIdx.x] = sortCls[sortBase + j];
        }
        __syncthreads();
        if (!active) continue;
        unsigned long long bits = 0;
        const int first = (colStart == rowStart) ? threadIdx.x + 1 : 0;
        for (int t = first; t < cols; ++t) {
            if (colCls[t] == ci && nmsOverlap(a, colBox + t * 4, iouThres)) bits |= 1ULL << t;
        }
        mask[(maskBase + i) * words + colStart / 64] = bits;
    }
}

// greedy pass over the sorted boxes, one warp per image (the loop syncs once per kept box).
// An image with more candidates than mask holds (N rows) instead tests each candidate
// against the boxes kept so far, slower but rare (a very low conf_thres)
extern "C" __global__
void yoloNmsReduce(
    const int N,
    const int stride,
    const int maxDet,
    const int* __restrict__ count,
    const int* __restrict__ sortCls,
    const float* __restrict__ sortBox,
    const float iouThres,
    const unsigned long long* __restrict__ mask,
    int* __restrict__ keep,
    int* __restrict__ keepCount
) {
    __shared__ unsigned long long removed[NMS_MAX_ROWS / 64];
    const int b = blockIdx.x;
    const int n = count[b];
    int* kept_ = keep + (size_t)b * maxDet;
    int kept = 0;
    if (n > N) {
        const size_t sortBase = (size_t)b * stride;
        for (int i = 0; i < n && kept < maxDet; ++i) {
            const float* box = sortBox + (sortBase + i) * 4;
            const int cls = sortCls[sortBase + i];
            bool hit = false;
            for (int u = threadIdx.x; u < kept && !hit; u += blockDim.x) {
                const int k = kept_[u];
                hit = sortCls[sortBase + k] == cls && nmsOverlap(sortBox + (sortBase + k) * 4, box, iouThres);
            }
            if (!__any_sync(0xffffffffu, hit)) {
                if (threadIdx.x == 0) kept_[kept] = i;
                ++kept;
                __syncwarp();
            }
        }
        if (threadIdx.x == 0) keepCount[b] = kept;
        return;
    }
    const int words = (N + 63) / 64;
    const int used = (n + 63) / 64;
    for (int w = threadIdx.x; w < used; w += blockDim.x) removed[w] = 0;
    __syncwarp();
    const size_t base = (size_t)b * N;
    for (int w = 0; w < used && kept < maxDet; ++w) {
        // live boxes of word w, jumping straight from one survivor to the next
        const unsigned long long tail = (w == used - 1 && n % 64) ? (1ULL << (n % 64)) - 1 : ~0ULL;
        unsigned long long live = ~removed[w] & tail;
        while (live && kept < maxDet) {
            const int bit = __ffsll(live) - 1;
            const int i = w * 64 + bit;
            if (threadIdx.x == 0) kept_[kept] = i;
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

// the kept detections as unletterboxed (x1, y1, x2, y2, score, class) rows, plus nExtra head
// channels from extraStart per detection (mask coefficients, keypoints). With points the
// extras are (x, y, visibility) triplets whose x and y are unletterboxed like the boxes.
// A block per kept slot; counts gets each image's kept count
extern "C" __global__
void yoloGather(
    const float* __restrict__ out,
    const int C,
    const int N,
    const int extraStart,
    const int nExtra,
    const int points,
    const int maxDet,
    const int* __restrict__ keep,
    const int* __restrict__ keepCount,
    const int* __restrict__ sortAnchor,
    const float* __restrict__ sortScore,
    const int* __restrict__ sortCls,
    const float* __restrict__ sortBox,
    const float* __restrict__ letterbox,
    float* __restrict__ dets,
    float* __restrict__ extras,
    int* __restrict__ counts
) {
    const int k = blockIdx.x;
    const int b = blockIdx.y;
    const int kept = keepCount[b];
    if (k == 0 && threadIdx.x == 0) counts[b] = kept;
    if (k >= kept) return;
    const size_t base = (size_t)b * N;
    const size_t row = (size_t)b * maxDet + k;
    const int s = keep[row];
    const Letterbox lb = loadLetterbox(letterbox, b);
    if (threadIdx.x == 0) {
        const float* box = sortBox + (base + s) * 4;
        float* det = dets + row * 6;
        det[0] = unletterX(box[0], lb);
        det[1] = unletterY(box[1], lb);
        det[2] = unletterX(box[2], lb);
        det[3] = unletterY(box[3], lb);
        det[4] = sortScore[base + s];
        det[5] = (float)sortCls[base + s];
    }
    const float* head = out + ((size_t)b * C + extraStart) * N + sortAnchor[base + s];
    float* dst = extras + row * nExtra;
    for (int m = threadIdx.x; m < nExtra; m += blockDim.x) {
        const float v = head[(size_t)m * N];
        const int axis = points ? m % 3 : 2;
        dst[m] = axis == 0 ? unletterX(v, lb) : axis == 1 ? unletterY(v, lb) : v;
    }
}

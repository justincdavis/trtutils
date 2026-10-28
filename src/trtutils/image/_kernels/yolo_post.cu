// YOLO-style head postprocessing: decode, sort, class-aware NMS, segmentation masks.
// Every kernel takes the batch index from the last grid dimension.
// The head is (B, C, N): rows 0-3 cxcywh, rows [clsStart, clsStart + nc) class scores.

#define NMS_MAX_WORDS 1024  // NMS supports up to 64 * NMS_MAX_WORDS candidates
#define SEG_MAX_COEFS 256

// keep anchors whose best class score is above conf, in any order
extern "C" __global__
void yoloDecode(
    const float* __restrict__ out,
    const int C,
    const int N,
    const int clsStart,
    const int nc,
    const float conf,
    int* __restrict__ count,
    int* __restrict__ candAnchor,
    float* __restrict__ candScore,
    int* __restrict__ candCls
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
    if (!(best > conf)) return;
    const int slot = atomicAdd(count + b, 1);
    candAnchor[(size_t)b * N + slot] = a;
    candScore[(size_t)b * N + slot] = best;
    candCls[(size_t)b * N + slot] = bestCls;
}

// sort candidates by score, ties by anchor (a stable sort of the anchor order),
// writing their xyxy boxes in sorted order
extern "C" __global__
void yoloRank(
    const float* __restrict__ out,
    const int C,
    const int N,
    const int* __restrict__ count,
    const int* __restrict__ candAnchor,
    const float* __restrict__ candScore,
    const int* __restrict__ candCls,
    int* __restrict__ sortAnchor,
    float* __restrict__ sortScore,
    int* __restrict__ sortCls,
    float* __restrict__ sortBox
) {
    __shared__ float tileScore[256];
    __shared__ int tileAnchor[256];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    const int n = count[b];
    if ((int)(blockIdx.x * blockDim.x) >= n) return;
    const size_t base = (size_t)b * N;
    const bool active = i < n;
    const float s = active ? candScore[base + i] : 0.0f;
    const int a = active ? candAnchor[base + i] : 0;
    int rank = 0;
    for (int start = 0; start < n; start += blockDim.x) {
        const int j = start + threadIdx.x;
        if (j < n) {
            tileScore[threadIdx.x] = candScore[base + j];
            tileAnchor[threadIdx.x] = candAnchor[base + j];
        }
        __syncthreads();
        const int len = min((int)blockDim.x, n - start);
        for (int t = 0; t < len; ++t) {
            const float sj = tileScore[t];
            rank += (sj > s) || (sj == s && tileAnchor[t] < a);
        }
        __syncthreads();
    }
    if (!active) return;
    sortAnchor[base + rank] = a;
    sortScore[base + rank] = s;
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

// bit j of row i is set when sorted box j > i overlaps box i of the same class
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
    const int rowStart = blockIdx.y * 64;
    const int colStart = blockIdx.x * 64;
    if (rowStart >= n || colStart >= n || blockIdx.x < blockIdx.y) return;
    const size_t base = (size_t)b * N;
    const int words = (N + 63) / 64;
    const int cols = min(64, n - colStart);
    if ((int)threadIdx.x < cols) {
        const int j = colStart + threadIdx.x;
        for (int k = 0; k < 4; ++k) colBox[threadIdx.x * 4 + k] = sortBox[(base + j) * 4 + k];
        colCls[threadIdx.x] = sortCls[base + j];
    }
    __syncthreads();
    const int i = rowStart + threadIdx.x;
    if (i >= n) return;
    const float* bi = sortBox + (base + i) * 4;
    // opencv computes IoU on double (x, y, w, h) rects built from float32 xywh
    const double ax = bi[0];
    const double ay = bi[1];
    const double aw = (double)(bi[2] - bi[0]);
    const double ah = (double)(bi[3] - bi[1]);
    const int ci = sortCls[base + i];
    unsigned long long bits = 0;
    const int first = (colStart == rowStart) ? threadIdx.x + 1 : 0;
    for (int t = first; t < cols; ++t) {
        if (colCls[t] != ci) continue;
        const float* bj = colBox + t * 4;
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
    mask[(base + i) * words + blockIdx.x] = bits;
}

// greedy pass over the sorted boxes, one block per image
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
    const int words = (N + 63) / 64;
    const int used = (n + 63) / 64;
    for (int w = threadIdx.x; w < used; w += blockDim.x) removed[w] = 0;
    __syncthreads();
    const size_t base = (size_t)b * N;
    int kept = 0;
    for (int i = 0; i < n && kept < maxDet; ++i) {
        if (removed[i / 64] & (1ULL << (i % 64))) continue;
        if (threadIdx.x == 0) keep[(size_t)b * maxDet + kept] = i;
        ++kept;
        const unsigned long long* row = mask + (base + i) * words;
        // only words at or after i hold bits for later boxes
        for (int w = i / 64 + threadIdx.x; w < used; w += blockDim.x) removed[w] |= row[w];
        __syncthreads();
    }
    if (threadIdx.x == 0) keepCount[b] = kept;
}

// gather the kept detections as unletterboxed, clipped (x1, y1, x2, y2, score, class)
// rows and their mask coefficients
extern "C" __global__
void yoloSegGather(
    const float* __restrict__ out,
    const int C,
    const int N,
    const int coefStart,
    const int nm,
    const int maxDet,
    const int* __restrict__ keep,
    const int* __restrict__ keepCount,
    const int* __restrict__ sortAnchor,
    const float* __restrict__ sortScore,
    const int* __restrict__ sortCls,
    const float* __restrict__ sortBox,
    const float* __restrict__ letterbox,  // per image: ratio x, ratio y, pad x, pad y, width, height
    float* __restrict__ dets,
    float* __restrict__ coefs
) {
    const int k = blockIdx.x;
    const int b = blockIdx.y;
    if (k >= keepCount[b]) return;
    const size_t base = (size_t)b * N;
    const int s = keep[(size_t)b * maxDet + k];
    const size_t row = (size_t)b * maxDet + k;
    if (threadIdx.x == 0) {
        const float* lb = letterbox + b * 6;
        const float* box = sortBox + (base + s) * 4;
        float* det = dets + row * 6;
        det[0] = fminf(fmaxf((box[0] - lb[2]) / lb[0], 0.0f), lb[4]);
        det[1] = fminf(fmaxf((box[1] - lb[3]) / lb[1], 0.0f), lb[5]);
        det[2] = fminf(fmaxf((box[2] - lb[2]) / lb[0], 0.0f), lb[4]);
        det[3] = fminf(fmaxf((box[3] - lb[3]) / lb[1], 0.0f), lb[5]);
        det[4] = sortScore[base + s];
        det[5] = (float)sortCls[base + s];
    }
    const int a = sortAnchor[base + s];
    for (int m = threadIdx.x; m < nm; m += blockDim.x) {
        coefs[row * nm + m] = out[((size_t)b * C + coefStart + m) * N + a];
    }
}

__device__ __forceinline__ float protoProb(
    const float* __restrict__ proto,
    const float* __restrict__ coef,
    const int nm,
    const int plane,
    const int idx
) {
    float logit = 0.0f;
    for (int m = 0; m < nm; ++m) logit += coef[m] * proto[(size_t)m * plane + idx];
    return 1.0f / (1.0f + expf(-logit));
}

// write the binary mask of each kept detection inside its box, matching
// cv2.warpAffine(INTER_LINEAR | WARP_INVERSE_MAP, BORDER_REPLICATE) over the proto crop
extern "C" __global__
void yoloSegMasks(
    const float* __restrict__ proto,  // (nm, mh, mw) of one image
    const int nm,
    const int mh,
    const int mw,
    const float* __restrict__ dets,  // (n, 6) of one image
    const float* __restrict__ coefs,  // (n, nm) of one image
    const int n,
    const int width,
    const int height,
    const double scaleX,
    const double scaleY,
    const double offX,
    const double offY,
    const float maskThres,
    unsigned char* __restrict__ masks,  // (n, height, width)
    int* __restrict__ regions  // (n, 4) written box: x1, y1, x2, y2
) {
    __shared__ float coefShared[SEG_MAX_COEFS];
    const int k = blockIdx.x;
    if (k >= n) return;
    const float* det = dets + (size_t)k * 6;
    const int bx1 = max((int)floorf(det[0]), 0);
    const int by1 = max((int)floorf(det[1]), 0);
    const int bx2 = min((int)ceilf(det[2]), width);
    const int by2 = min((int)ceilf(det[3]), height);
    const int cx1 = max((int)floor(scaleX * bx1 + offX), 0);
    const int cx2 = min((int)ceil(scaleX * (bx2 - 1) + offX) + 1, mw);
    const int cy1 = max((int)floor(scaleY * by1 + offY), 0);
    const int cy2 = min((int)ceil(scaleY * (by2 - 1) + offY) + 1, mh);
    const bool valid = bx2 > bx1 && by2 > by1 && cx2 > cx1 && cy2 > cy1;
    if (blockIdx.y == 0 && threadIdx.x == 0) {
        int* region = regions + k * 4;
        region[0] = valid ? bx1 : 0;
        region[1] = valid ? by1 : 0;
        region[2] = valid ? bx2 : 0;
        region[3] = valid ? by2 : 0;
    }
    if (!valid) return;
    for (int m = threadIdx.x; m < nm; m += blockDim.x) coefShared[m] = coefs[(size_t)k * nm + m];
    __syncthreads();

    // opencv's fixed point warp: 10 bit coordinates rounded to 1/32 of a cell
    const double m0 = (double)(float)scaleX;
    const double m2 = (double)(float)(scaleX * bx1 + offX - cx1);
    const double m4 = (double)(float)scaleY;
    const double m5 = (double)(float)(scaleY * by1 + offY - cy1);
    const int cw = cx2 - cx1;
    const int ch = cy2 - cy1;
    const int bw = bx2 - bx1;
    const int total = bw * (by2 - by1);
    const int plane = mh * mw;
    unsigned char* dst = masks + (size_t)k * height * width;
    for (int p = blockIdx.y * blockDim.x + threadIdx.x; p < total; p += gridDim.y * blockDim.x) {
        const int xd = p % bw;
        const int yd = p / bw;
        const int X = (__double2int_rn(m2 * 1024.0) + 16 + __double2int_rn(m0 * xd * 1024.0)) >> 5;
        const int Y = (__double2int_rn((m4 * yd + m5) * 1024.0) + 16) >> 5;
        const int ix = X >> 5;
        const int iy = Y >> 5;
        const float fx = (X & 31) / 32.0f;
        const float fy = (Y & 31) / 32.0f;
        const int x0 = cx1 + min(max(ix, 0), cw - 1);
        const int x1 = cx1 + min(max(ix + 1, 0), cw - 1);
        const int y0 = cy1 + min(max(iy, 0), ch - 1);
        const int y1 = cy1 + min(max(iy + 1, 0), ch - 1);
        const float v00 = protoProb(proto, coefShared, nm, plane, y0 * mw + x0);
        const float v01 = protoProb(proto, coefShared, nm, plane, y0 * mw + x1);
        const float v10 = protoProb(proto, coefShared, nm, plane, y1 * mw + x0);
        const float v11 = protoProb(proto, coefShared, nm, plane, y1 * mw + x1);
        const float v = v00 * ((1.0f - fy) * (1.0f - fx)) + v01 * ((1.0f - fy) * fx)
            + v10 * (fy * (1.0f - fx)) + v11 * (fy * fx);
        dst[(size_t)(by1 + yd) * width + bx1 + xd] = v >= maskThres;
    }
}

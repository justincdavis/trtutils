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
// a block per 64 rows and stripe of column blocks, so empty launches stay cheap
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
    if (rowStart >= n) return;
    const size_t base = (size_t)b * N;
    const int words = (N + 63) / 64;
    const int i = rowStart + threadIdx.x;
    const bool active = i < n;
    // opencv computes IoU on double (x, y, w, h) rects built from float32 xywh
    double ax = 0.0, ay = 0.0, aw = 0.0, ah = 0.0;
    int ci = -1;
    if (active) {
        const float* bi = sortBox + (base + i) * 4;
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
,
    int* __restrict__ keepCountOut  // mapped host copy of keepCount
) {
    const int k = blockIdx.x;
    const int b = blockIdx.y;
    if (k == 0 && threadIdx.x == 0) keepCountOut[b] = keepCount[b];
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

// a detection's mask box in the image and the proto cells it samples, as _decode_masks
struct SegBox {
    int bx1, by1, bx2, by2, cx1, cx2, cy1, cy2;
    bool valid;
};

__device__ __forceinline__ SegBox segBox(
    const float* __restrict__ det,
    const int width,
    const int height,
    const int mh,
    const int mw,
    const double scaleX,
    const double scaleY,
    const double offX,
    const double offY
) {
    SegBox r;
    r.bx1 = max((int)floorf(det[0]), 0);
    r.by1 = max((int)floorf(det[1]), 0);
    r.bx2 = min((int)ceilf(det[2]), width);
    r.by2 = min((int)ceilf(det[3]), height);
    r.cx1 = max((int)floor(scaleX * r.bx1 + offX), 0);
    r.cx2 = min((int)ceil(scaleX * (r.bx2 - 1) + offX) + 1, mw);
    r.cy1 = max((int)floor(scaleY * r.by1 + offY), 0);
    r.cy2 = min((int)ceil(scaleY * (r.by2 - 1) + offY) + 1, mh);
    r.valid = r.bx2 > r.bx1 && r.by2 > r.by1 && r.cx2 > r.cx1 && r.cy2 > r.cy1;
    return r;
}

// segBox of a block's detection, computed once: float64 math is slow on GeForce parts
#define BLOCK_SEG_BOX(r, det, width, height, mh, mw, scaleX, scaleY, offX, offY) \
    __shared__ SegBox r##Shared; \
    if (threadIdx.x == 0) r##Shared = segBox(det, width, height, mh, mw, scaleX, scaleY, offX, offY); \
    __syncthreads(); \
    const SegBox r = r##Shared

// opencv's fixed point warp coordinates: 10 bit, rounded to 1/32 of a cell
__device__ __forceinline__ int warpX(const double m0, const double m2, const int xd) {
    return (__double2int_rn(m2 * 1024.0) + 16 + __double2int_rn(m0 * xd * 1024.0)) >> 5;
}
__device__ __forceinline__ int warpY(const double m4, const double m5, const int yd) {
    return (__double2int_rn((m4 * yd + m5) * 1024.0) + 16) >> 5;
}

// Masks are written to mapped host memory, where byte stores over PCIe are slow, so rows go
// out in aligned 16 byte chunks. Bytes of a chunk outside the box are zero: the previous
// frame's boxes are cleared first and everything else in a mask is already zero. Bytes outside
// the mask's own plane are left alone (byte stores), so neighbouring masks are never touched.
#define CHUNK 16
// chunkStart's marker for a chunk past the row's end (a real offset can be negative)
#define NO_CHUNK (-(1LL << 62))

// for each row of [x1, x2) the aligned chunks overlapping it, chunks per row is an upper bound
__device__ __forceinline__ int chunksPerRow(const int x1, const int x2) { return (x2 - x1 + 2 * CHUNK - 2) / CHUNK; }

// plane offset of the aligned chunk c of row y overlapping [x1, x2), or NO_CHUNK
__device__ __forceinline__ long long chunkStart(
    const unsigned char* plane, const int width, const int y, const int x1, const int x2, const int c
) {
    const unsigned long long row = (unsigned long long)(plane + (size_t)y * width);
    const unsigned long long first = (row + x1) & ~(unsigned long long)(CHUNK - 1);
    const unsigned long long start = first + (unsigned long long)c * CHUNK;
    return start < row + x2 ? (long long)start - (long long)plane : NO_CHUNK;
}

// store 16 bytes at plane offset off, byte by byte where the chunk leaves the plane
__device__ __forceinline__ void storeChunk(
    unsigned char* plane, const long long off, const long long size, const uint4 v
) {
    if (off >= 0 && off + CHUNK <= size) {
        *reinterpret_cast<uint4*>(plane + off) = v;
        return;
    }
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&v);
    for (int i = 0; i < CHUNK; ++i) {
        if (off + i >= 0 && off + i < size) plane[off + i] = bytes[i];
    }
}

// zero the boxes the previous frame wrote into each mask, skipping chunks the new box overlaps
extern "C" __global__
void yoloSegClear(
    unsigned char* __restrict__ masks,  // (>= n, height, width)
    const int* __restrict__ regions,  // (n, 4) old x1, y1, x2, y2
    const int n,
    const int width,
    const int height,
    const float* __restrict__ dets,  // (newN, 6) this frame's boxes
    const int newN,
    const int mh,
    const int mw,
    const double scaleX,
    const double scaleY,
    const double offX,
    const double offY
) {
    const int k = blockIdx.x;
    if (k >= n) return;
    // regions and dets are mapped host memory, read once per block
    __shared__ int r[4];
    __shared__ SegBox nb;
    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; ++i) r[i] = regions[k * 4 + i];
        nb.valid = false;
        if (k < newN) nb = segBox(dets + (size_t)k * 6, width, height, mh, mw, scaleX, scaleY, offX, offY);
    }
    __syncthreads();
    if (r[2] <= r[0] || r[3] <= r[1]) return;
    unsigned char* plane = masks + (size_t)k * height * width;
    const long long size = (long long)height * width;
    const int cpr = chunksPerRow(r[0], r[2]);
    const int total = cpr * (r[3] - r[1]);
    const uint4 zero = make_uint4(0, 0, 0, 0);
    for (int p = blockIdx.y * blockDim.x + threadIdx.x; p < total; p += gridDim.y * blockDim.x) {
        const int y = r[1] + p / cpr;
        const long long off = chunkStart(plane, width, y, r[0], r[2], p % cpr);
        if (off == NO_CHUNK) continue;
        if (nb.valid && y >= nb.by1 && y < nb.by2) {
            // the mask kernel rewrites every chunk of row y overlapping the new box, zeros
            // outside it; chunks reaching into a neighbouring row are cleared here
            const long long fx = off - (long long)y * width;
            if (fx >= 0 && fx + CHUNK <= width && fx + CHUNK > nb.bx1 && fx < nb.bx2) continue;
        }
        storeChunk(plane, off, size, zero);
    }
}

// sigmoid(coef . proto) of every proto cell a detection samples, (n, mh * mw) of one image
extern "C" __global__
void yoloSegProbs(
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
    float* __restrict__ probs
) {
    __shared__ float coefShared[SEG_MAX_COEFS];
    const int k = blockIdx.x;
    if (k >= n) return;
    BLOCK_SEG_BOX(r, dets + (size_t)k * 6, width, height, mh, mw, scaleX, scaleY, offX, offY);
    if (!r.valid) return;
    for (int m = threadIdx.x; m < nm; m += blockDim.x) coefShared[m] = coefs[(size_t)k * nm + m];
    __syncthreads();
    const int cw = r.cx2 - r.cx1;
    const int total = cw * (r.cy2 - r.cy1);
    const int plane = mh * mw;
    for (int p = blockIdx.y * blockDim.x + threadIdx.x; p < total; p += gridDim.y * blockDim.x) {
        const int idx = (r.cy1 + p / cw) * mw + r.cx1 + p % cw;
        float logit = 0.0f;
        for (int m = 0; m < nm; ++m) logit += coefShared[m] * proto[(size_t)m * plane + idx];
        probs[(size_t)k * plane + idx] = 1.0f / (1.0f + expf(-logit));
    }
}

// write the binary mask of each kept detection inside its box, matching
// cv2.warpAffine(INTER_LINEAR | WARP_INVERSE_MAP, BORDER_REPLICATE) over the proto crop
extern "C" __global__
void yoloSegMasks(
    const float* __restrict__ probs,  // (n, mh * mw) from yoloSegProbs
    const int mh,
    const int mw,
    const float* __restrict__ dets,  // (n, 6) of one image
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
    const int k = blockIdx.x;
    if (k >= n) return;
    BLOCK_SEG_BOX(r, dets + (size_t)k * 6, width, height, mh, mw, scaleX, scaleY, offX, offY);
    if (blockIdx.y == 0 && threadIdx.x == 0) {
        int* region = regions + k * 4;
        region[0] = r.valid ? r.bx1 : 0;
        region[1] = r.valid ? r.by1 : 0;
        region[2] = r.valid ? r.bx2 : 0;
        region[3] = r.valid ? r.by2 : 0;
    }
    if (!r.valid) return;

    // the float32 affine matrix cv2.warpAffine sees
    const double m0 = (double)(float)scaleX;
    const double m2 = (double)(float)(scaleX * r.bx1 + offX - r.cx1);
    const double m4 = (double)(float)scaleY;
    const double m5 = (double)(float)(scaleY * r.by1 + offY - r.cy1);
    const int cw = r.cx2 - r.cx1;
    const int ch = r.cy2 - r.cy1;
    const float* prob = probs + (size_t)k * mh * mw;
    unsigned char* plane = masks + (size_t)k * height * width;
    const long long size = (long long)height * width;
    const int cpr = chunksPerRow(r.bx1, r.bx2);
    const int total = cpr * (r.by2 - r.by1);
    for (int p = blockIdx.y * blockDim.x + threadIdx.x; p < total; p += gridDim.y * blockDim.x) {
        const int row = r.by1 + p / cpr;
        const long long off = chunkStart(plane, width, row, r.bx1, r.bx2, p % cpr);
        if (off == NO_CHUNK) continue;
        const long long rowOff = (long long)row * width;
        uint4 v = make_uint4(0, 0, 0, 0);
        unsigned char* bytes = reinterpret_cast<unsigned char*>(&v);
        for (int i = 0; i < CHUNK; ++i) {
            const long long o = off + i;
            if (o < 0 || o >= size) continue;
            // bytes of a chunk lie in its own row unless it straddles a row boundary
            int y = row;
            int x = (int)(o - rowOff);
            if (x < 0 || x >= width) {
                y = (int)(o / width);
                x = (int)(o % width);
            }
            if (y < r.by1 || y >= r.by2 || x < r.bx1 || x >= r.bx2) continue;
            const int xd = x - r.bx1;
            const int yd = y - r.by1;
            const int X = warpX(m0, m2, xd);
            const int Y = warpY(m4, m5, yd);
            const int ix = X >> 5;
            const int iy = Y >> 5;
            const float fx = (X & 31) / 32.0f;
            const float fy = (Y & 31) / 32.0f;
            const int x0 = r.cx1 + min(max(ix, 0), cw - 1);
            const int x1 = r.cx1 + min(max(ix + 1, 0), cw - 1);
            const int y0 = r.cy1 + min(max(iy, 0), ch - 1);
            const int y1 = r.cy1 + min(max(iy + 1, 0), ch - 1);
            const float v00 = prob[y0 * mw + x0];
            const float v01 = prob[y0 * mw + x1];
            const float v10 = prob[y1 * mw + x0];
            const float v11 = prob[y1 * mw + x1];
            const float val = v00 * ((1.0f - fy) * (1.0f - fx)) + v01 * ((1.0f - fy) * fx)
                + v10 * (fy * (1.0f - fx)) + v11 * (fy * fx);
            bytes[i] = val >= maskThres;
        }
        storeChunk(plane, off, size, v);
    }
}

// gather the kept pose detections as unletterboxed, clipped (x1, y1, x2, y2, score, class)
// rows and their (x, y, visibility) keypoints, x and y unletterboxed and clipped like boxes
extern "C" __global__
void yoloPoseGather(
    const float* __restrict__ out,
    const int C,
    const int N,
    const int kptStart,
    const int nk,
    const int maxDet,
    const int* __restrict__ keep,
    const int* __restrict__ keepCount,
    const int* __restrict__ sortAnchor,
    const float* __restrict__ sortScore,
    const int* __restrict__ sortCls,
    const float* __restrict__ sortBox,
    const float* __restrict__ letterbox,  // per image: ratio x, ratio y, pad x, pad y, width, height
    float* __restrict__ dets,
    float* __restrict__ kpts
,
    int* __restrict__ keepCountOut  // mapped host copy of keepCount
) {
    const int k = blockIdx.x;
    const int b = blockIdx.y;
    if (k == 0 && threadIdx.x == 0) keepCountOut[b] = keepCount[b];
    if (k >= keepCount[b]) return;
    const size_t base = (size_t)b * N;
    const int s = keep[(size_t)b * maxDet + k];
    const size_t row = (size_t)b * maxDet + k;
    const float* lb = letterbox + b * 6;
    if (threadIdx.x == 0) {
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
    const float* head = out + (size_t)b * C * N + a;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) {
        const float* src = head + (size_t)(kptStart + j * 3) * N;
        float* dst = kpts + (row * nk + j) * 3;
        dst[0] = fminf(fmaxf((src[0] - lb[2]) / lb[0], 0.0f), lb[4]);
        dst[1] = fminf(fmaxf((src[N] - lb[3]) / lb[1], 0.0f), lb[5]);
        dst[2] = src[2 * (size_t)N];
    }
}

// oriented boxes: float32 math in the operation order of the numpy path (no fused multiply-adds)
__device__ __forceinline__ float mul(const float a, const float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float add(const float a, const float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float sub(const float a, const float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float quo(const float a, const float b) { return __fdiv_rn(a, b); }

// decode the sorted candidates into rotated boxes (cx, cy, w, h, angle) in image coordinates and
// their gaussian covariance (a, b, c), 8 floats per box
extern "C" __global__
void obbPrep(
    const float* __restrict__ out,
    const int C,
    const int N,
    const int* __restrict__ count,
    const int* __restrict__ sortAnchor,
    const float* __restrict__ letterbox,  // per image: ratio x, ratio y, pad x, pad y
    float* __restrict__ rb
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (i >= count[b]) return;
    const size_t base = (size_t)b * N;
    const float* box = out + (size_t)b * C * N + sortAnchor[base + i];
    const float* lb = letterbox + b * 4;
    const float cx = quo(sub(box[0], lb[2]), lb[0]);
    const float cy = quo(sub(box[N], lb[3]), lb[1]);
    const float w = quo(box[2 * (size_t)N], lb[0]);
    const float h = quo(box[3 * (size_t)N], lb[1]);
    const float angle = box[(size_t)(C - 1) * N];
    const float varW = quo(mul(w, w), 12.0f);
    const float varH = quo(mul(h, h), 12.0f);
    const float cs = cosf(angle);
    const float sn = sinf(angle);
    const float cs2 = mul(cs, cs);
    const float sn2 = mul(sn, sn);
    float* dst = rb + (base + i) * 8;
    dst[0] = cx;
    dst[1] = cy;
    dst[2] = w;
    dst[3] = h;
    dst[4] = angle;
    dst[5] = add(mul(varW, cs2), mul(varH, sn2));
    dst[6] = add(mul(varW, sn2), mul(varH, cs2));
    dst[7] = mul(mul(sub(varW, varH), cs), sn);
}

// probiou between sorted boxes i and j, as _probiou_terms
__device__ float probiou(const float* __restrict__ p, const float* __restrict__ q) {
    const float eps = 1e-7f;
    const float dx = sub(p[0], q[0]);
    const float dy = sub(p[1], q[1]);
    const float sa = add(p[5], q[5]);
    const float sb = add(p[6], q[6]);
    const float sc = add(p[7], q[7]);
    const float det = sub(mul(sa, sb), mul(sc, sc));
    const float denom = add(det, eps);
    const float t1 = mul(quo(add(mul(sa, mul(dy, dy)), mul(sb, mul(dx, dx))), denom), 0.25f);
    const float t2 = mul(quo(mul(mul(sc, -dx), dy), denom), 0.5f);
    const float det1 = fmaxf(sub(mul(p[5], p[6]), mul(p[7], p[7])), 0.0f);
    const float det2 = fmaxf(sub(mul(q[5], q[6]), mul(q[7], q[7])), 0.0f);
    const float t3 = mul(logf(add(quo(det, add(mul(4.0f, sqrtf(mul(det1, det2))), eps)), eps)), 0.5f);
    const float bd = fminf(fmaxf(add(add(t1, t2), t3), eps), 100.0f);
    return sub(1.0f, sqrtf(add(sub(1.0f, expf(-bd)), eps)));
}

// ultralytics fast-nms: sorted box j survives unless any higher ranked box of its class
// overlaps it by probiou >= iouThres, whether or not that box survives itself. One warp per
// box; pairs whose centers are farther apart than reach * (trace_i + trace_j) cannot reach
// the threshold (bd >= |d|^2 / (4 * trace)) and skip the full formula, as the CPU path does
extern "C" __global__
void obbSuppress(
    const int N,
    const int* __restrict__ count,
    const int* __restrict__ sortCls,
    const float* __restrict__ rb,
    const float iouThres,
    const float reach,
    unsigned char* __restrict__ alive
) {
    const int j = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    const int b = blockIdx.y;
    if (j >= count[b]) return;
    const size_t base = (size_t)b * N;
    const int cj = sortCls[base + j];
    const float* q = rb + (base + j) * 8;
    const float traceJ = q[5] + q[6];
    bool hit = false;
    for (int start = 0; start < j; start += 32) {
        const int i = start + lane;
        if (i < j && sortCls[base + i] == cj) {
            const float* p = rb + (base + i) * 8;
            const float dx = p[0] - q[0];
            const float dy = p[1] - q[1];
            if (dx * dx + dy * dy <= reach * (p[5] + p[6] + traceJ)) {
                hit = probiou(p, q) >= iouThres;
            }
        }
        if (__any_sync(0xffffffffu, hit)) {
            hit = true;
            break;
        }
    }
    if (lane == 0) alive[base + j] = hit ? 0 : 1;
}

// first maxDet survivors in rank order as rows of (enclosing xyxy, score, class, cx, cy, w, h,
// angle), one block per image
extern "C" __global__
void obbKeep(
    const int N,
    const int maxDet,
    const int* __restrict__ count,
    const unsigned char* __restrict__ alive,
    const int* __restrict__ sortCls,
    const float* __restrict__ sortScore,
    const float* __restrict__ rb,
    float* __restrict__ dets,
    int* __restrict__ keepCount
) {
    __shared__ int warpSums[32];
    const int b = blockIdx.x;
    const int t = threadIdx.x;
    const int n = count[b];
    const size_t base = (size_t)b * N;
    const int chunk = (n + blockDim.x - 1) / blockDim.x;
    const int lo = min(t * chunk, n);
    const int hi = min(lo + chunk, n);
    int mine = 0;
    for (int k = lo; k < hi; ++k) mine += alive[base + k];
    // exclusive block scan of the counts: shuffles within warps, then over the warp totals
    // (blockDim is a multiple of 32)
    const int lane = t & 31;
    const int warp = t >> 5;
    const int warps = blockDim.x >> 5;
    int inc = mine;
    for (int d = 1; d < 32; d <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, inc, d);
        if (lane >= d) inc += v;
    }
    if (lane == 31) warpSums[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        int w = lane < warps ? warpSums[lane] : 0;
        for (int d = 1; d < 32; d <<= 1) {
            const int v = __shfl_up_sync(0xffffffffu, w, d);
            if (lane >= d) w += v;
        }
        warpSums[lane] = w;
    }
    __syncthreads();
    if (t == 0) keepCount[b] = min(warpSums[warps - 1], maxDet);
    int pos = inc - mine + (warp > 0 ? warpSums[warp - 1] : 0);
    for (int k = lo; k < hi && pos < maxDet; ++k) {
        if (!alive[base + k]) continue;
        const float* r = rb + (base + k) * 8;
        const float cs = cosf(r[4]);
        const float sn = sinf(r[4]);
        const float halfW = mul(add(fabsf(mul(r[2], cs)), fabsf(mul(r[3], sn))), 0.5f);
        const float halfH = mul(add(fabsf(mul(r[2], sn)), fabsf(mul(r[3], cs))), 0.5f);
        float* row = dets + ((size_t)b * maxDet + pos) * 11;
        row[0] = fmaxf(sub(r[0], halfW), 0.0f);
        row[1] = fmaxf(sub(r[1], halfH), 0.0f);
        row[2] = fmaxf(add(r[0], halfW), 0.0f);
        row[3] = fmaxf(add(r[1], halfH), 0.0f);
        row[4] = sortScore[base + k];
        row[5] = (float)sortCls[base + k];
        for (int m = 0; m < 5; ++m) row[6 + m] = r[m];
        ++pos;
    }
}

// YOLO segmentation: binary masks of the detections yoloGather keeps.
// Masks match cv2.warpAffine over the proto crop, see postprocess_yolo_seg.

#define SEG_MAX_COEFS 256

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
    const int* __restrict__ newCount,  // this frame's kept count on the device, at most newN used
    const int mh,
    const int mw,
    const double scaleX,
    const double scaleY,
    const double offX,
    const double offY
) {
    const int k = blockIdx.x;
    if (k >= n) return;
    // thread 0 reads this block's region and new box into shared memory
    __shared__ int r[4];
    __shared__ SegBox nb;
    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; ++i) r[i] = regions[k * 4 + i];
        nb.valid = false;
        if (k < min(newN, newCount[0])) nb = segBox(dets + (size_t)k * 6, width, height, mh, mw, scaleX, scaleY, offX, offY);
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
    const int* __restrict__ count,  // kept count on the device, at most n used
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
    if (k >= min(n, count[0])) return;
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
    const int* __restrict__ count,  // kept count on the device, at most n used
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
    if (k >= min(n, count[0])) return;
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

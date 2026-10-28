// OCR preprocessing on the device, bit-exact with OpenCV 4.11 on x86:
//   - cv2.resize(INTER_LINEAR) of 8UC3 (fixed point, its SIMD vertical pass)
//   - cv2.warpPerspective(INTER_CUBIC, BORDER_REPLICATE) of 8UC3 (INTER_BITS=5 coordinates,
//     15 bit cubic weights, see imgwarp.cpp)
// Double and float math uses the __*_rn intrinsics so the compiler never fuses a multiply-add.

#define RESIZE_BITS 11
#define RESIZE_SCALE 2048
#define TAB_SIZE 32

struct Coef {
    int s;
    int a0;
    int a1;
};

// resizeGeneric_ coefficients of destination index d; edges of x are pinned (fx = 0), y rows
// are clamped when read instead
__device__ __forceinline__ Coef linearCoef(const int d, const int nSrc, const double scale, const bool pinEdges) {
    float f = (float)__dsub_rn(__dmul_rn((double)d + 0.5, scale), 0.5);
    int s = (int)floorf(f);
    f = __fsub_rn(f, (float)s);
    if (pinEdges) {
        if (s < 0) {
            f = 0.0f;
            s = 0;
        }
        if (s >= nSrc - 1) {
            f = 0.0f;
            s = nSrc - 1;
        }
    }
    Coef c;
    c.s = s;
    c.a0 = __float2int_rn(__fmul_rn(__fsub_rn(1.0f, f), (float)RESIZE_SCALE));
    c.a1 = __float2int_rn(__fmul_rn(f, (float)RESIZE_SCALE));
    return c;
}

// one channel of one resized pixel: src is (sh, sw, 3) uint8, returns the cv2.resize value
__device__ __forceinline__ int resizePixel(
    const unsigned char* __restrict__ src,
    const int sh,
    const int sw,
    const Coef cx,
    const Coef cy,
    const int c
) {
    const int x1 = min(cx.s + 1, sw - 1);
    const int r0 = min(max(cy.s, 0), sh - 1);
    const int r1 = min(max(cy.s + 1, 0), sh - 1);
    const unsigned char* p0 = src + (size_t)r0 * sw * 3;
    const unsigned char* p1 = src + (size_t)r1 * sw * 3;
    const int h0 = p0[cx.s * 3 + c] * cx.a0 + p0[x1 * 3 + c] * cx.a1;
    const int h1 = p1[cx.s * 3 + c] * cx.a0 + p1[x1 * 3 + c] * cx.a1;
    // VResizeLinearVec_32s8u: high halves of the 16 bit products
    const int v = (((cy.a0 * (h0 >> 4)) >> 16) + ((cy.a1 * (h1 >> 4)) >> 16) + 2) >> 2;
    return min(max(v, 0), 255);
}

// detector input: resize image b to (dh, dw) and normalize through lut (3, 256) into the
// (B, 3, outH, outW) tensor; the pixels beyond (dh, dw) are left as they are
extern "C" __global__
void ocrDetResize(
    const unsigned char* __restrict__ src,
    const int sh,
    const int sw,
    const int dh,
    const int dw,
    const float* __restrict__ lut,
    const int outH,
    const int outW,
    const int batchIdx,
    float* __restrict__ out
) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= dh * dw) return;
    const int dy = p / dw;
    const int dx = p % dw;
    const double scaleX = 1.0 / ((double)dw / (double)sw);
    const double scaleY = 1.0 / ((double)dh / (double)sh);
    const Coef cx = linearCoef(dx, sw, scaleX, true);
    const Coef cy = linearCoef(dy, sh, scaleY, false);
    float* o = out + (size_t)batchIdx * 3 * outH * outW + (size_t)dy * outW + dx;
    for (int c = 0; c < 3; ++c) {
        o[(size_t)c * outH * outW] = lut[c * 256 + resizePixel(src, sh, sw, cx, cy, c)];
    }
}

// cv2.warpPerspective(INTER_CUBIC, BORDER_REPLICATE) of every box, written to crops as
// (rows, cols, 3) uint8; with rot set the crop is np.rot90'd, as crop_text_region does
//   mats: (n, 9) float64 inverse matrices, geom: (n, 4) int32 [width, height, rot, byte offset]
//   tab:  (1024, 16) int16 cubic weights
extern "C" __global__
void ocrWarpCrops(
    const unsigned char* __restrict__ frame,
    const int fh,
    const int fw,
    const double* __restrict__ mats,
    const int* __restrict__ geom,
    const short* __restrict__ tab,
    unsigned char* __restrict__ crops
) {
    const int box = blockIdx.y;
    const int cw = geom[box * 4];
    const int ch = geom[box * 4 + 1];
    const int rot = geom[box * 4 + 2];
    unsigned char* dst = crops + geom[box * 4 + 3];
    const double* M = mats + box * 9;
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= cw * ch) return;
    const int x = p % cw;
    const int y = p / cw;

    // warpPerspective evaluates a row from the origin of its 32x32 pixel block
    int bh0 = min(TAB_SIZE / 2, ch);
    const int bw0 = min(TAB_SIZE * TAB_SIZE / bh0, cw);
    const int bx = (x / bw0) * bw0;
    const int x1 = x - bx;
    const double X0 = __dadd_rn(__dadd_rn(__dmul_rn(M[0], bx), __dmul_rn(M[1], y)), M[2]);
    const double Y0 = __dadd_rn(__dadd_rn(__dmul_rn(M[3], bx), __dmul_rn(M[4], y)), M[5]);
    const double W0 = __dadd_rn(__dadd_rn(__dmul_rn(M[6], bx), __dmul_rn(M[7], y)), M[8]);
    double W = __dadd_rn(W0, __dmul_rn(M[6], x1));
    W = W != 0.0 ? __ddiv_rn((double)TAB_SIZE, W) : 0.0;
    const double fX = fmax(-2147483648.0, fmin(2147483647.0, __dmul_rn(__dadd_rn(X0, __dmul_rn(M[0], x1)), W)));
    const double fY = fmax(-2147483648.0, fmin(2147483647.0, __dmul_rn(__dadd_rn(Y0, __dmul_rn(M[3], x1)), W)));
    const int X = __double2int_rn(fX);
    const int Y = __double2int_rn(fY);
    const int sx = min(max(X >> 5, -32768), 32767) - 1;
    const int sy = min(max(Y >> 5, -32768), 32767) - 1;
    const short* w = tab + ((Y & (TAB_SIZE - 1)) * TAB_SIZE + (X & (TAB_SIZE - 1))) * 16;

    int xs[4];
    int ys[4];
    for (int i = 0; i < 4; ++i) {
        xs[i] = min(max(sx + i, 0), fw - 1);
        ys[i] = min(max(sy + i, 0), fh - 1);
    }
    // np.rot90 maps pixel (x, y) of the warp to row cw - 1 - x, column y
    const int cols = rot ? ch : cw;
    const size_t o = rot ? ((size_t)(cw - 1 - x) * cols + y) : ((size_t)y * cols + x);
    for (int c = 0; c < 3; ++c) {
        int sum = 0;
        for (int i = 0; i < 4; ++i) {
            const unsigned char* row = frame + (size_t)ys[i] * fw * 3 + c;
            sum += row[xs[0] * 3] * w[i * 4] + row[xs[1] * 3] * w[i * 4 + 1]
                 + row[xs[2] * 3] * w[i * 4 + 2] + row[xs[3] * 3] * w[i * 4 + 3];
        }
        dst[o * 3 + c] = (unsigned char)min(max((sum + (1 << 14)) >> 15, 0), 255);
    }
}

// recognizer input: crop j of the batch resized to (48, rw) and normalized through lut
// (3, 256) into (nb, 3, 48, bw), zero past rw
//   info: (n, 4) int32 [byte offset, rows, cols, resized width], the batch is info[start:start+nb]
extern "C" __global__
void ocrRecInput(
    const unsigned char* __restrict__ crops,
    const int* __restrict__ info,
    const int start,
    const int height,
    const int bw,
    const float* __restrict__ lut,
    float* __restrict__ out
) {
    const int j = blockIdx.y;
    const int* row = info + (start + j) * 4;
    const int sh = row[1];
    const int sw = row[2];
    const int rw = row[3];
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= height * bw) return;
    const int dy = p / bw;
    const int dx = p % bw;
    float* o = out + (size_t)j * 3 * height * bw + (size_t)dy * bw + dx;
    if (dx >= rw) {
        for (int c = 0; c < 3; ++c) o[(size_t)c * height * bw] = 0.0f;
        return;
    }
    const unsigned char* src = crops + row[0];
    const double scaleX = 1.0 / ((double)rw / (double)sw);
    const double scaleY = 1.0 / ((double)height / (double)sh);
    const Coef cx = linearCoef(dx, sw, scaleX, true);
    const Coef cy = linearCoef(dy, sh, scaleY, false);
    for (int c = 0; c < 3; ++c) {
        o[(size_t)c * height * bw] = lut[c * 256 + resizePixel(src, sh, sw, cx, cy, c)];
    }
}

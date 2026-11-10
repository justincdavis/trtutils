// Oriented-box decode and probiou NMS. Rank comes from yolo_nms.cu.
// float32 math in the operation order of the numpy path (no fused multiply-adds).
// Every kernel takes the batch index from the last grid dimension.

#include "postprocess.cuh"

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
    const float* __restrict__ letterbox,
    float* __restrict__ rb
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (i >= count[b]) return;
    const size_t base = (size_t)b * N;
    const float* box = out + (size_t)b * C * N + sortAnchor[base + i];
    const Letterbox lb = loadLetterbox(letterbox, b);
    const float cx = quo(sub(box[0], lb.px), lb.rx);
    const float cy = quo(sub(box[N], lb.py), lb.ry);
    const float w = quo(box[2 * (size_t)N], lb.rx);
    const float h = quo(box[3 * (size_t)N], lb.ry);
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

// probiou between sorted boxes i and j, as _probiou
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
// the threshold (bd >= |d|^2 / (4 * trace)) and skip the full formula
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
// angle). One thread per image: the list is the conf-filtered candidates, small next to the NMS
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
    const int b = blockIdx.y;
    const int n = count[b];
    const size_t base = (size_t)b * N;
    int pos = 0;
    for (int k = 0; k < n && pos < maxDet; ++k) {
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
    keepCount[b] = pos;
}

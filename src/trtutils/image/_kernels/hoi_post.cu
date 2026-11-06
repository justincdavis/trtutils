// Hand-interaction postprocess: unletterbox, drop scores below conf, class-aware
// NMS, then gather pair probabilities and side.
// ponytail: one thread per image. Fine for the fixed top-k (about 100); split
// the pair gather if K grows past a few hundred.
//
// NMS matches cv2.dnn.NMSBoxesBatched (the Rect2d overload): scores strictly
// above conf, stable score order, IoU on double xywh, suppress when overlap
// is strictly above the threshold. Class offsets use max_coord starting at 0.

#include "postprocess.cuh"

struct DRect {
    double x, y, w, h;
};

__device__ bool rectEmpty(DRect r) { return r.w <= 0.0 || r.h <= 0.0; }

__device__ double rectArea(DRect r) { return r.w * r.h; }

// opencv Rect::operator&=
__device__ DRect rectAnd(DRect a, DRect b) {
    DRect z = {0.0, 0.0, 0.0, 0.0};
    if (rectEmpty(a) || rectEmpty(b)) return z;
    DRect rxMin = (a.x < b.x) ? a : b;
    DRect rxMax = (a.x < b.x) ? b : a;
    DRect ryMin = (a.y < b.y) ? a : b;
    DRect ryMax = (a.y < b.y) ? b : a;
    if ((rxMin.x < 0.0 && rxMin.x + rxMin.w < rxMax.x) ||
        (ryMin.y < 0.0 && ryMin.y + ryMin.h < ryMax.y))
        return z;
    a.w = fmin(rxMin.w - (rxMax.x - rxMin.x), rxMax.w);
    a.h = fmin(ryMin.h - (ryMax.y - ryMin.y), ryMax.h);
    a.x = rxMax.x;
    a.y = ryMax.y;
    if (rectEmpty(a)) return z;
    return a;
}

// 1 - jaccardDistance, rounded the way opencv's rectOverlap does
__device__ float rectOverlap(DRect a, DRect b) {
    double aa = rectArea(a);
    double ab = rectArea(b);
    if ((aa + ab) <= 2.2204460492503131e-16) return 1.f;
    double inter = rectArea(rectAnd(a, b));
    double dist = 1.0 - inter / (aa + ab - inter);
    return 1.f - (float)dist;
}

// a (x1, y1, x2, y2, score, label) row as an xywh rect, offset by its label
__device__ DRect nmsRect(const float* row, double maxCoord) {
    double offset = (double)(int)row[5] * (maxCoord + 1.0);
    DRect r;
    r.x = (double)row[0] + offset;
    r.y = (double)row[1] + offset;
    r.w = (double)(row[2] - row[0]);
    r.h = (double)(row[3] - row[1]);
    return r;
}

extern "C" __global__
void hoiPost(
    const float* __restrict__ boxes,
    const float* __restrict__ scores,
    const void* __restrict__ labels,
    const int labelKind,
    const float* __restrict__ pairs,
    const void* __restrict__ side,
    const int sideKind,
    const int k,
    const int c,
    const float* __restrict__ letterbox,
    const float conf,
    const float iou,
    float* __restrict__ rows,
    int* __restrict__ src,
    int* __restrict__ order,
    int* __restrict__ keep,
    float* __restrict__ dets,
    float* __restrict__ outPairs,
    long long* __restrict__ outSide,
    int* __restrict__ counts
) {
    const int image = blockIdx.x;
    const size_t idBase = (size_t)image * k;
    float* fr = rows + idBase * 6;
    int* origin = src + idBase;
    const int m = keepUnletterboxed(
        boxes + idBase * 4,
        4,
        scores + idBase,
        1,
        labels,
        idBase,
        1,
        labelKind,
        k,
        loadLetterbox(letterbox, image),
        conf,
        fr,
        origin
    );

    // opencv starts max_coord at 0, so negative corners never lower it
    double maxCoord = 0.0;
    for (int i = 0; i < m; ++i) {
        const float* row = fr + i * 6;
        maxCoord = fmax(maxCoord, (double)row[0]);
        maxCoord = fmax(maxCoord, (double)row[1]);
        maxCoord = fmax(maxCoord, (double)row[0] + (double)(row[2] - row[0]));
        maxCoord = fmax(maxCoord, (double)row[1] + (double)(row[3] - row[1]));
    }

    int* ord = order + idBase;
    int p = 0;
    for (int i = 0; i < m; ++i) {
        if (fr[i * 6 + 4] > conf) ord[p++] = i;
    }
    for (int i = 1; i < p; ++i) {
        int key = ord[i];
        float ks = fr[key * 6 + 4];
        int j = i - 1;
        while (j >= 0 && fr[ord[j] * 6 + 4] < ks) {
            ord[j + 1] = ord[j];
            --j;
        }
        ord[j + 1] = key;
    }

    int* kept = keep + idBase;
    int n = 0;
    for (int t = 0; t < p; ++t) {
        int ci = ord[t];
        DRect box = nmsRect(fr + ci * 6, maxCoord);
        bool ok = true;
        for (int u = 0; u < n; ++u) {
            if (rectOverlap(box, nmsRect(fr + kept[u] * 6, maxCoord)) > iou) {
                ok = false;
                break;
            }
        }
        if (ok) kept[n++] = ci;
    }

    float* od = dets + idBase * 6;
    long long* os = outSide + idBase;
    float* op = outPairs + idBase * k * c;
    const float* ip = pairs + idBase * k * c;
    for (int a = 0; a < n; ++a) {
        int ca = kept[a];
        int ia = origin[ca];
        for (int t = 0; t < 6; ++t) od[a * 6 + t] = fr[ca * 6 + t];
        if (side) os[a] = loadIndex(side, sideKind, idBase + ia);
        for (int b = 0; b < n; ++b) {
            int ib = origin[kept[b]];
            const float* sp = ip + ((size_t)ia * k + ib) * c;
            float* dp = op + ((size_t)a * k + b) * c;
            for (int t = 0; t < c; ++t) dp[t] = sp[t];
        }
    }
    counts[image] = n;
}

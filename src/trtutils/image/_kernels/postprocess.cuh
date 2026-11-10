// Device helpers shared by the postprocess kernels.
#pragma once

// a row of the (batch, 6) letterbox table built by letterbox_table() in postprocessors/_cuda.py: resize ratio,
// padding, and the original image size boxes clip to (infinity when unknown)
struct Letterbox {
    float rx, ry, px, py, width, height;
};

__device__ __forceinline__ Letterbox loadLetterbox(const float* __restrict__ table, const int image) {
    const float* row = table + image * 6;
    return {row[0], row[1], row[2], row[3], row[4], row[5]};
}

// letterboxed x and y back in the original image, clipped as clip_boxes
__device__ __forceinline__ float unletterX(const float x, const Letterbox& lb) {
    return fminf(fmaxf((x - lb.px) / lb.rx, 0.0f), lb.width);
}

__device__ __forceinline__ float unletterY(const float y, const Letterbox& lb) {
    return fminf(fmaxf((y - lb.py) / lb.ry, 0.0f), lb.height);
}

// index tensors (class ids, counts) come as int32, float32 or int64, see index_kind() in image/kernels.py
__device__ __forceinline__ int loadIndex(const void* __restrict__ base, const int kind, const size_t i) {
    if (kind == 1) return (int)((const float*)base)[i];
    if (kind == 2) return (int)((const long long*)base)[i];
    return ((const int*)base)[i];
}

// compact rows j < limit scoring at least conf into (x1, y1, x2, y2, score, id) rows with the
// box unletterboxed, keeping row order; src (optional) gets each kept row's index. Returns the
// kept count. Row j reads boxes[j * boxStride], scores[j * scoreStride], ids[idBase + j * idStride]
__device__ int keepUnletterboxed(
    const float* __restrict__ boxes,
    const int boxStride,
    const float* __restrict__ scores,
    const int scoreStride,
    const void* __restrict__ ids,
    const size_t idBase,
    const int idStride,
    const int idKind,
    const int limit,
    const Letterbox lb,
    const float conf,
    float* __restrict__ rows,
    int* __restrict__ src
) {
    int kept = 0;
    for (int j = 0; j < limit; ++j) {
        const float score = scores[(size_t)j * scoreStride];
        if (!(score >= conf)) continue;
        const float* box = boxes + (size_t)j * boxStride;
        float* row = rows + (size_t)kept * 6;
        row[0] = unletterX(box[0], lb);
        row[1] = unletterY(box[1], lb);
        row[2] = unletterX(box[2], lb);
        row[3] = unletterY(box[3], lb);
        row[4] = score;
        row[5] = (float)loadIndex(ids, idKind, idBase + (size_t)j * idStride);
        if (src) src[kept] = j;
        ++kept;
    }
    return kept;
}

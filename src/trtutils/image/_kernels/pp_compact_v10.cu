// YOLO-v10 compaction: input (B, N, 6) rows (x1, y1, x2, y2, score, class).
// Confidence filter + letterbox remap, compact survivors per image.

extern "C" __global__
void compactV10(
    const float* __restrict__ v10,       // (B, N, 6)
    const float* __restrict__ ratios,     // (B, 2)
    const float* __restrict__ pads,      // (B, 2)
    float* __restrict__ out_boxes,       // (B, N, 4)
    float* __restrict__ out_scores,      // (B, N)
    int* __restrict__ out_classes,       // (B, N)
    int* __restrict__ counts,            // (B,)
    const int batch,
    const int n,
    const float conf_thres,
    const int use_conf
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * n) return;

    const int b = idx / n;

    const float* row = v10 + (size_t)idx * 6;
    const float s = row[4];
    if (use_conf && s < conf_thres) return;

    float x1 = row[0];
    float y1 = row[1];
    float x2 = row[2];
    float y2 = row[3];

    const float rw = ratios[b * 2];
    const float rh = ratios[b * 2 + 1];
    const float px = pads[b * 2];
    const float py = pads[b * 2 + 1];
    x1 = (x1 - px) / rw;
    y1 = (y1 - py) / rh;
    x2 = (x2 - px) / rw;
    y2 = (y2 - py) / rh;

    const int pos = atomicAdd(&counts[b], 1);
    float* box_out = out_boxes + (size_t)(b * n + pos) * 4;
    box_out[0] = fmaxf(x1, 0.0f);
    box_out[1] = fmaxf(y1, 0.0f);
    box_out[2] = fmaxf(x2, 0.0f);
    box_out[3] = fmaxf(y2, 0.0f);
    out_scores[b * n + pos] = s;
    out_classes[b * n + pos] = (int)row[5];
}

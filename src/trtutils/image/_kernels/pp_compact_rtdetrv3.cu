// RT-DETR v3 (PaddlePaddle export) compaction: combined_dets (T, 6) rows
// (class_id, score, x1, y1, x2, y2) in original image coords, plus
// num_dets_per_image (B,). Slices rows per image, confidence filters
// (+ finite check), and compacts survivors.

extern "C" __global__
void compactRtdetrV3(
    const float* __restrict__ dets,      // (T, 6)
    const void* __restrict__ num_dets,   // (B,) int32 or float
    float* __restrict__ out_boxes,       // (B, K, 4)
    float* __restrict__ out_scores,      // (B, K)
    int* __restrict__ out_classes,       // (B, K)
    int* __restrict__ counts,            // (B,)
    const int batch,
    const int total,
    const int k,
    const float conf_thres,
    const int use_conf,
    const int num_dets_is_float
) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;

    // find the image this row belongs to (B is tiny)
    const int* nd_i = (const int*)num_dets;
    const float* nd_f = (const float*)num_dets;
    int prefix = 0;
    int b = -1;
    for (int i = 0; i < batch; i++) {
        const int n_i = num_dets_is_float ? (int)nd_f[i] : (int)nd_i[i];
        if (t < prefix + n_i) {
            b = i;
            break;
        }
        prefix += n_i;
    }
    if (b < 0) return;  // padding rows beyond sum(num_dets)

    const float* row = dets + (size_t)t * 6;
    const float s = row[1];
    if (use_conf && s < conf_thres) return;

    float x1 = row[2];
    float y1 = row[3];
    float x2 = row[4];
    float y2 = row[5];
    if (!isfinite(x1) || !isfinite(y1) || !isfinite(x2) || !isfinite(y2)) return;

    const int pos = atomicAdd(&counts[b], 1);
    float* box_out = out_boxes + (size_t)(b * k + pos) * 4;
    box_out[0] = fmaxf(x1, 0.0f);
    box_out[1] = fmaxf(y1, 0.0f);
    box_out[2] = fmaxf(x2, 0.0f);
    box_out[3] = fmaxf(y2, 0.0f);
    out_scores[b * k + pos] = s;
    out_classes[b * k + pos] = (int)row[0];
}

// Batched detection compaction: confidence filter (+ optional finite check and
// letterbox remap), then compact surviving candidates to the front of per-image
// output rows. One thread per (batch, candidate). Output order among survivors
// is nondeterministic (atomic), which is fine for detections.

extern "C" __global__
void compactBoxes(
    const float* __restrict__ boxes,      // (B, K, 4)
    const float* __restrict__ scores,     // (B, K)
    const void* __restrict__ classes,     // (B, K) int32 or float
    const float* __restrict__ ratios,     // (B, 2) (width_scale, height_scale)
    const float* __restrict__ pads,       // (B, 2) (width_offset, height_offset)
    const void* __restrict__ num_dets,    // (B,) int32 or float, or nullptr
    float* __restrict__ out_boxes,        // (B, K, 4)
    float* __restrict__ out_scores,       // (B, K)
    int* __restrict__ out_classes,        // (B, K)
    int* __restrict__ counts,             // (B,)
    int* __restrict__ out_indices,        // (B, K) original candidate index per compacted slot, or nullptr
    const int batch,
    const int k,
    const float conf_thres,
    const int use_conf,
    const int use_finite,
    const int use_rescale,
    const int classes_is_float,
    const int num_dets_is_float
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * k) return;

    const int b = idx / k;
    const int i = idx % k;

    if (num_dets != nullptr) {
        const int* nd_i = (const int*)num_dets;
        const float* nd_f = (const float*)num_dets;
        const int n_i = num_dets_is_float ? (int)nd_f[b] : (int)nd_i[b];
        if (i >= n_i) return;
    }

    const float s = scores[idx];
    if (use_conf && s < conf_thres) return;

    const float* box_in = boxes + (size_t)idx * 4;
    float x1 = box_in[0];
    float y1 = box_in[1];
    float x2 = box_in[2];
    float y2 = box_in[3];
    if (use_finite && (!isfinite(x1) || !isfinite(y1) || !isfinite(x2) || !isfinite(y2))) return;

    if (use_rescale) {
        const float rw = ratios[b * 2];
        const float rh = ratios[b * 2 + 1];
        const float px = pads[b * 2];
        const float py = pads[b * 2 + 1];
        x1 = (x1 - px) / rw;
        y1 = (y1 - py) / rh;
        x2 = (x2 - px) / rw;
        y2 = (y2 - py) / rh;
    }

    const int* cls_in = (const int*)classes;
    const float* cls_in_f = (const float*)classes;
    const int cls = classes_is_float ? (int)cls_in_f[idx] : cls_in[idx];

    float* box_out = out_boxes + (size_t)(b * k) * 4;
    const int pos = atomicAdd(&counts[b], 1);
    box_out[pos * 4 + 0] = fmaxf(x1, 0.0f);
    box_out[pos * 4 + 1] = fmaxf(y1, 0.0f);
    box_out[pos * 4 + 2] = fmaxf(x2, 0.0f);
    box_out[pos * 4 + 3] = fmaxf(y2, 0.0f);
    out_scores[b * k + pos] = s;
    out_classes[b * k + pos] = cls;
    if (out_indices != nullptr) {
        out_indices[b * k + pos] = i;
    }
}

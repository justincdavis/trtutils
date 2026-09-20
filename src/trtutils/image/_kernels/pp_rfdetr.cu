// RF-DETR postprocessing. Logits (B, Q, C) -> sigmoid probs, top-Q over the
// Q*C flattened scores per image, then denormalize selected boxes (cx, cy, w,
// h) * input size to xyxy, unletterbox, and compact survivors.
//
// top-Q is done by iterative block-argmax (one block per image): each
// iteration is a block reduction over the N = Q*C values, which is fast
// (L2-resident) and bounded by Q iterations.

#define FLT_MAX 3.402823466e+38f

extern "C" __global__
void rfdetrSigmoid(
    const float* __restrict__ logits,  // (B, Q, C)
    float* __restrict__ probs,         // (B, Q, C)
    const int total
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    probs[idx] = 1.0f / (1.0f + expf(-logits[idx]));
}

extern "C" __global__
void rfdetrTopk(
    const float* __restrict__ probs,   // (B, N), N = Q*C
    float* __restrict__ topk_scores,   // (B, Q)
    int* __restrict__ topk_idx,        // (B, Q)
    const int num_queries,
    const int n
) {
    // one block per image
    float* p = (float*)(probs + (size_t)blockIdx.x * n);
    float* ts = topk_scores + blockIdx.x * num_queries;
    int* ti = topk_idx + blockIdx.x * num_queries;

    __shared__ float red[1024];
    __shared__ int red_idx[1024];

    for (int sel = 0; sel < num_queries; sel++) {
        float bv = -FLT_MAX;
        int bi = -1;
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            const float v = p[j];
            if (v > bv || (v == bv && bi < 0)) {
                bv = v;
                bi = j;
            }
        }
        red[threadIdx.x] = bv;
        red_idx[threadIdx.x] = bi;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                const int o = threadIdx.x + stride;
                if (red[o] > red[threadIdx.x] || (red[o] == red[threadIdx.x] && red_idx[threadIdx.x] < 0)) {
                    red[threadIdx.x] = red[o];
                    red_idx[threadIdx.x] = red_idx[o];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            ts[sel] = red[0];
            ti[sel] = red_idx[0];
            if (red_idx[0] >= 0) {
                p[red_idx[0]] = -FLT_MAX;
            }
        }
        __syncthreads();
    }
}

extern "C" __global__
void rfdetrGather(
    const float* __restrict__ dets,        // (B, Q, 4) normalized (cx, cy, w, h)
    const float* __restrict__ topk_scores, // (B, Q)
    const int* __restrict__ topk_idx,      // (B, Q)
    const float* __restrict__ ratios,      // (B, 2)
    const float* __restrict__ pads,       // (B, 2)
    float* __restrict__ out_boxes,        // (B, Q, 4)
    float* __restrict__ out_scores,       // (B, Q)
    int* __restrict__ out_classes,        // (B, Q)
    int* __restrict__ counts,             // (B,)
    const int num_queries,
    const int num_classes,
    const float input_w,
    const float input_h,
    const float conf_thres,
    const int use_conf
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    if (idx >= total) return;

    const int b = idx / num_queries;
    const float s = topk_scores[idx];
    if (use_conf && s < conf_thres) return;

    const int flat = topk_idx[idx];
    const int q = flat / num_classes;
    const int c = flat % num_classes;

    const float* box = dets + ((size_t)b * num_queries + q) * 4;
    const float cx = box[0] * input_w;
    const float cy = box[1] * input_h;
    const float w = box[2] * input_w;
    const float h = box[3] * input_h;

    const float rw = ratios[b * 2];
    const float rh = ratios[b * 2 + 1];
    const float px = pads[b * 2];
    const float py = pads[b * 2 + 1];

    float x1 = ((cx - w * 0.5f) - px) / rw;
    float y1 = ((cy - h * 0.5f) - py) / rh;
    float x2 = ((cx + w * 0.5f) - px) / rw;
    float y2 = ((cy + h * 0.5f) - py) / rh;

    const int pos = atomicAdd(&counts[b], 1);
    float* out = out_boxes + ((size_t)b * num_queries + pos) * 4;
    out[0] = fmaxf(x1, 0.0f);
    out[1] = fmaxf(y1, 0.0f);
    out[2] = fmaxf(x2, 0.0f);
    out[3] = fmaxf(y2, 0.0f);
    out_scores[b * num_queries + pos] = s;
    out_classes[b * num_queries + pos] = c - 1;  // 1-indexed in model
}

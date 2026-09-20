// Class-aware NMS for hand-object interaction candidates, run on the
// confidence-filtered survivors written by compactBoxes (counts[b] = n).
//
// One block per image:
//   1. bitonic-sort (score, idx) pairs descending in shared memory
//   2. greedy NMS in score order: suppress if IoU > threshold with a kept
//      box of the same class (matches cv2.dnn.NMSBoxesBatched semantics)
//   3. compact kept boxes/scores/labels (deterministic, score order) and
//      gather the pair_probs sub-matrix and side for the kept candidates
//
// Requires k <= 1024 (shared memory budget).

#define FLT_MAX 3.402823466e+38f

extern "C" __global__
void handNms(
    const float* __restrict__ boxes,      // (B, K, 4) remapped
    const float* __restrict__ scores,     // (B, K)
    const int* __restrict__ labels,       // (B, K)
    const float* __restrict__ pair_probs, // (B, K, K, C)
    const int* __restrict__ side,         // (B, K) or nullptr
    const int* __restrict__ orig_idx,     // (B, K) original candidate index per compacted slot
    float* __restrict__ out_boxes,       // (B, K, 4)
    float* __restrict__ out_scores,      // (B, K)
    int* __restrict__ out_labels,        // (B, K)
    float* __restrict__ out_pairs,       // (B, K, K, C)
    int* __restrict__ out_side,          // (B, K)
    int* __restrict__ counts,            // (B,) in: survivors, out: kept
    const int k,
    const int num_pair_classes,
    const float iou_thres,
    const float conf_thres,
    const int has_side
) {
    const int b = blockIdx.x;
    const int n = counts[b];
    if (n == 0) return;

    const float* b_boxes = boxes + (size_t)b * k * 4;
    const float* b_scores = scores + (size_t)b * k;
    const int* b_labels = labels + (size_t)b * k;
    const float* b_pairs = pair_probs + (size_t)b * k * k * num_pair_classes;

    // next power of two >= k (k <= 1024)
    int p = 1;
    while (p < k) p <<= 1;

    __shared__ float s_score[1024];
    __shared__ int s_idx[1024];
    __shared__ float s_kb[1024 * 4];
    __shared__ int s_kl[1024];
    __shared__ int s_kidx[1024];
    __shared__ int s_nkept;
    __shared__ float s_red[1024];

    if (threadIdx.x == 0) {
        s_nkept = 0;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < p; i += blockDim.x) {
        // only the first n (confidence-filtered survivors) are valid candidates
        const bool valid = i < n;
        s_score[i] = valid ? b_scores[i] : -FLT_MAX;
        s_idx[i] = valid ? i : -1;
    }
    __syncthreads();

    // bitonic sort, descending
    for (int size = 2; size <= p; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < p; i += blockDim.x) {
                const int j = i ^ stride;
                if (j < i) continue;
                // note: bitonic direction bit is inverted for descending order
                const bool descending = (i & size) == 0;
                const bool swap = descending ? (s_score[i] < s_score[j]) : (s_score[i] > s_score[j]);
                if (swap) {
                    const float fs = s_score[i];
                    const int fi = s_idx[i];
                    s_score[i] = s_score[j];
                    s_idx[i] = s_idx[j];
                    s_score[j] = fs;
                    s_idx[j] = fi;
                }
            }
            __syncthreads();
        }
    }

    // greedy NMS in descending score order
    for (int ci = 0; ci < n; ci++) {
        // cv2's GetMaxScoreIndex uses a strict >: at-or-below-conf candidates
        // are dropped (the compact stage keeps >=)
        if (s_score[ci] <= conf_thres) continue;
        const int cand = s_idx[ci];
        const int cand_label = b_labels[cand];
        const float cb0 = b_boxes[cand * 4 + 0];
        const float cb1 = b_boxes[cand * 4 + 1];
        const float cb2 = b_boxes[cand * 4 + 2];
        const float cb3 = b_boxes[cand * 4 + 3];
        const float cand_area = (cb2 - cb0) * (cb3 - cb1);

        float max_iou = 0.0f;
        for (int t2 = threadIdx.x; t2 < s_nkept; t2 += blockDim.x) {
            const int k4 = s_kidx[t2] * 4;
            const float kb0 = b_boxes[k4 + 0];
            const float kb1 = b_boxes[k4 + 1];
            const float kb2 = b_boxes[k4 + 2];
            const float kb3 = b_boxes[k4 + 3];

            const float iw = fmaxf(0.0f, fminf(cb2, kb2) - fmaxf(cb0, kb0));
            const float ih = fmaxf(0.0f, fminf(cb3, kb3) - fmaxf(cb1, kb1));
            const float inter = iw * ih;
            // cv2's jaccard is 0/0 = NaN for two zero-area boxes, and
            // `NaN <= threshold` is false, so any two degenerate boxes
            // suppress each other (even across classes: the class-offset
            // trick only zeroes the intersection, not the zero union).
            if (cand_area == 0.0f && (kb2 - kb0) * (kb3 - kb1) == 0.0f) {
                max_iou = FLT_MAX;
            } else if (s_kl[t2] == cand_label) {
                const float union_area = cand_area + (kb2 - kb0) * (kb3 - kb1) - inter;
                max_iou = fmaxf(max_iou, inter / union_area);
            }
        }

        s_red[threadIdx.x] = max_iou;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                s_red[threadIdx.x] = fmaxf(s_red[threadIdx.x], s_red[threadIdx.x + stride]);
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            if (s_red[0] <= iou_thres) {
                const int o4 = s_nkept * 4;
                s_kb[o4 + 0] = b_boxes[cand * 4 + 0];
                s_kb[o4 + 1] = b_boxes[cand * 4 + 1];
                s_kb[o4 + 2] = b_boxes[cand * 4 + 2];
                s_kb[o4 + 3] = b_boxes[cand * 4 + 3];
                s_kl[s_nkept] = cand_label;
                s_kidx[s_nkept] = cand;
                s_nkept++;
            }
        }
        __syncthreads();
    }

    // compact outputs (score order)
    const int nkept = s_nkept;

    // compact outputs (score order)
    for (int i = threadIdx.x; i < nkept; i += blockDim.x) {
        const int o4 = i * 4;
        const int orig = orig_idx[(size_t)b * k + s_kidx[i]];
        out_boxes[(size_t)b * k * 4 + o4 + 0] = s_kb[o4 + 0];
        out_boxes[(size_t)b * k * 4 + o4 + 1] = s_kb[o4 + 1];
        out_boxes[(size_t)b * k * 4 + o4 + 2] = s_kb[o4 + 2];
        out_boxes[(size_t)b * k * 4 + o4 + 3] = s_kb[o4 + 3];
        out_scores[(size_t)b * k + i] = b_scores[s_kidx[i]];
        out_labels[(size_t)b * k + i] = s_kl[i];
        if (has_side) {
            out_side[(size_t)b * k + i] = side[(size_t)b * k + orig];
        }
    }

    // gather pair_probs[i, j, c] = pair_probs[kept[i], kept[j], c] (original indices)
    const int total_pairs = nkept * nkept * num_pair_classes;
    for (int t3 = threadIdx.x; t3 < total_pairs; t3 += blockDim.x) {
        const int c3 = t3 % num_pair_classes;
        const int r = t3 / num_pair_classes;
        const int j = r % nkept;
        const int i = r / nkept;
        const int oi = orig_idx[(size_t)b * k + s_kidx[i]];
        const int oj = orig_idx[(size_t)b * k + s_kidx[j]];
        out_pairs[(size_t)b * k * k * num_pair_classes + (size_t)(i * k + j) * num_pair_classes + c3] =
            b_pairs[((size_t)oi * k + oj) * num_pair_classes + c3];
    }

    if (threadIdx.x == 0) {
        counts[b] = nkept;
    }
}

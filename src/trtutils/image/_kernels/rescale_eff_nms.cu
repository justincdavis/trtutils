extern "C" __global__
void rescaleEffNMSDetections(
    float* __restrict__ boxes,
    const float* __restrict__ scores,
    bool* __restrict__ mask,
    const int topk,
    const float confThres,
    const float widthScale,
    const float heightScale,
    const float widthOffset,
    const float heightOffset
) {
    // EfficientNMS output format: (Batch, TopK, 4)
    // final dimension is 4, representing:
    // [x1, y1, x2, y2]

    // EfficientNMS has dynamic topk and bboxSize of 4

    // algorithm:
    // for bbox
    // x1 = (x1 - x_offset) / width_scale
    // y1 = (y1 - y_offset) / height_scale
    // x2 = (x2 - x_offset) / width_scale
    // y2 = (y2 - y_offset) / height_scale
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= topk) return;

    // check if score is above threshold
    float score = scores[idx];
    if (score < confThres) {
        mask[idx] = false;
        return;
    }
    mask[idx] = true;

    // x1
    float x = boxes[idx * 4];
    x = (x - widthOffset) / widthScale;
    boxes[idx * 4] = fmaxf(x, 0.0f);

    // y1
    float y = boxes[idx * 4 + 1];
    y = (y - heightOffset) / heightScale;
    boxes[idx * 4 + 1] = fmaxf(y, 0.0f);

    // x2
    float x2 = boxes[idx * 4 + 2];
    x2 = (x2 - widthOffset) / widthScale;
    boxes[idx * 4 + 2] = fmaxf(x2, 0.0f);

    // y2
    float y2 = boxes[idx * 4 + 3];
    y2 = (y2 - heightOffset) / heightScale;
    boxes[idx * 4 + 3] = fmaxf(y2, 0.0f);
}

extern "C" __global__
void rescaleV10Detections(
    float* __restrict__ boxes,
    bool* __restrict__ mask,
    const float confThres,
    const float widthScale,
    const float heightScale,
    const float widthOffset,
    const float heightOffset
) {
    // YOLOv10 output format: (Batch, 300, 6)
    // final dimension is 6, representing:
    // [x1, y1, x2, y2, score, class]

    // YoloV10 has fixed topk of 300, and bboxSize of 6

    // algorithm:
    // for bbox
    // x1 = (x1 - x_offset) / width_scale
    // y1 = (y1 - y_offset) / height_scale
    // x2 = (x2 - x_offset) / width_scale
    // y2 = (y2 - y_offset) / height_scale
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 300) return;

    // check if score is above threshold
    float score = boxes[idx * 6 + 4];
    if (score < confThres) {
        mask[idx] = false;
        return;
    }
    mask[idx] = true;

    // x1
    float x = boxes[idx * 6];
    x = (x - widthOffset) / widthScale;
    boxes[idx * 6] = fmaxf(x, 0.0f);

    // y1
    float y = boxes[idx * 6 + 1];
    y = (y - heightOffset) / heightScale;
    boxes[idx * 6 + 1] = fmaxf(y, 0.0f);

    // x2
    float x2 = boxes[idx * 6 + 2];
    x2 = (x2 - widthOffset) / widthScale;
    boxes[idx * 6 + 2] = fmaxf(x2, 0.0f);

    // y2
    float y2 = boxes[idx * 6 + 3];
    y2 = (y2 - heightOffset) / heightScale;
    boxes[idx * 6 + 3] = fmaxf(y2, 0.0f);
}

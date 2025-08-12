extern "C" __global__
void rescaleDetections(
    const float* __restrict__ boxes,
    float* __restrict__ output,
    bool* __restrict__ mask,
    const int topk,
    const int bboxSize,
    const float confThres,
    const float widthScale,
    const float heightScale,
    const float widthOffset,
    const float heightOffset
) {
    // YOLOv10 output format: (Batch, 300, 6)
    // final dimension is 6, representing:
    // [x1, y1, x2, y2, score, class]
    // EfficientNMS output format: (Batch, TopK, 4)
    // final dimension is 4, representing:
    // [x1, y1, x2, y2]

    // YoloV10 has fixed topk of 300, and bboxSize of 6
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
    float score = boxes[idx * bboxSize + 4];
    if (score < confThres) {
        mask[idx] = false;
        return;
    }
    mask[idx] = true;

    // x1
    float x = boxes[idx * bboxSize];
    x = (x - widthOffset) / widthScale;
    output[idx * bboxSize] = x;

    // y1
    float y = boxes[idx * bboxSize + 1];
    y = (y - heightOffset) / heightScale;
    output[idx * bboxSize + 1] = y;

    // x2
    float x2 = boxes[idx * bboxSize + 2];
    x2 = (x2 - widthOffset) / widthScale;
    output[idx * bboxSize + 2] = x2;

    // y2
    float y2 = boxes[idx * bboxSize + 3];
    y2 = (y2 - heightOffset) / heightScale;
    output[idx * bboxSize + 3] = y2;
}

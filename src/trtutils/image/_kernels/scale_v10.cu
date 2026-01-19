extern "C" __global__
void rescaleV10(
    float* __restrict__ boxes,
    const float widthScale,
    const float heightScale,
    const float widthOffset,
    const float heightOffset
) {
    // YOLOv10 output format: (Batch, 300, 6)
    // final dimension is 6, representing:
    // [x, y, w, h, score, class]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 300) return;
}

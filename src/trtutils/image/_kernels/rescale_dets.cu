// Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
//
// MIT License

// Rescale detection boxes from letterboxed-input coordinates back to
// original-image coordinates, in place, for a whole batch.
//
// boxes:      (batchSize, numDets, boxStride) float tensor where the first
//             four values per detection are (x1, y1, x2, y2) in the
//             letterboxed input coordinate space. boxStride is 4 for
//             EfficientNMS det_boxes and 6 for YOLOv10 (x1,y1,x2,y2,score,class).
// transforms: (batchSize, 4) float tensor of per-image
//             (ratio_w, ratio_h, pad_x, pad_y) from preprocessing.
//
// Applies the same affine as the CPU postprocessors:
//   coord = max((coord - pad) / ratio, 0)
extern "C" __global__
void rescaleDetections(
    float* __restrict__ boxes,
    const float* __restrict__ transforms,
    const int batchSize,
    const int numDets,
    const int boxStride
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batchSize * numDets;
    if (idx >= total) return;

    const int b = idx / numDets;
    const float ratioW = transforms[b * 4 + 0];
    const float ratioH = transforms[b * 4 + 1];
    const float padX = transforms[b * 4 + 2];
    const float padY = transforms[b * 4 + 3];

    float* box = boxes + (long long)idx * boxStride;
    box[0] = fmaxf((box[0] - padX) / ratioW, 0.0f);
    box[1] = fmaxf((box[1] - padY) / ratioH, 0.0f);
    box[2] = fmaxf((box[2] - padX) / ratioW, 0.0f);
    box[3] = fmaxf((box[3] - padY) / ratioH, 0.0f);
}

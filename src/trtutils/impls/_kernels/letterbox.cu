extern "C" __global__
void letterboxResize(
    const unsigned char* __restrict__ inImg,
    unsigned char* __restrict__ outImg,
    const int inWidth,
    const int inHeight,
    const int outWidth,
    const int outHeight,
    const float scaleX,
    const float scaleY,
    const int startX,
    const int startY,
    const int newWidth,
    const int newHeight
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= outWidth || y >= outHeight) return;

    int outIdx = (y * outWidth + x) * 3;

    // always fill the value
    outImg[outIdx + 0] = 114;
    outImg[outIdx + 1] = 114;
    outImg[outIdx + 2] = 114;

    // if pixel is valid location for resizing input
    if (x >= startX && x < startX + newWidth && y >= startY && y < startY + newHeight) {
        // get source pixel locations
        float srcX = (x - startX) * scaleX;
        float srcY = (y - startY) * scaleY;

        int x0 = static_cast<int>(srcX);
        int y0 = static_cast<int>(srcY);

        int x1 = min(x0 + 1, inWidth - 1);
        int y1 = min(y0 + 1, inHeight - 1);

        // interpolation weights
        float dx = srcX - x0;
        float dy = srcY - y0;

        // compute adjacent four pixels
        int idx00 = (y0 * inWidth + x0) * 3;
        int idx01 = (y0 * inWidth + x1) * 3;
        int idx10 = (y1 * inWidth + x0) * 3;
        int idx11 = (y1 * inWidth + x1) * 3;

        // perform bilinear interpolation for pixel (on 3 channels)
        for (int c = 0; c < 3; ++c) {
            outImg[outIdx + c] = static_cast<unsigned char>(
                (1 - dx) * (1 - dy) * inImg[idx00 + c] +
                dx * (1 - dy) * inImg[idx01 + c] +
                (1 - dx) * dy * inImg[idx10 + c] +
                dx * dy * inImg[idx11 + c]);
        }
    }
}

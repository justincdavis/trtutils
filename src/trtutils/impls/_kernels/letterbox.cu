extern "C" __global__
void letterboxResize(
    const unsigned char* __restrict__ inImg,
    unsigned char* __restrict__ outImg,
    const int widthIn,
    const int heightIn,
    const int widthOut,
    const int heightOut,
    const int startX,
    const int startY,
    const int regionWidth,
    const int regionHeight
) {
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= widthOut || ty >= heightOut) return;

    const int outputIdx = (ty * widthOut + tx) * 3;
    outImg[outputIdx + 0] = 114;
    outImg[outputIdx + 1] = 114;
    outImg[outputIdx + 2] = 114;

    if (tx >= startX && tx < startX + regionWidth &&
        ty >= startY && ty < startY + regionHeight) {
        const float scaleX = widthIn / (float)regionWidth;
        const float scaleY = heightIn / (float)regionHeight;

        const float inputX = (tx - startX) * scaleX;
        const float inputY = (ty - startY) * scaleY;

        // get four surrounding pixels
        const int x0 = floor(inputX);
        const int y0 = floor(inputY);
        const int x1 = min(x0 + 1, widthIn - 1);
        const int y1 = min(y0 + 1, heightIn - 1);

        // interpolation weights
        const float dx = inputX - x0;
        const float dy = inputY - y0;
        const float w00 = (1.0f - dx) * (1.0f - dy);
        const float w01 = dx * (1.0f - dy);
        const float w10 = (1.0f - dx) * dy;
        const float w11 = dx * dy;

        const int inputIdx00 = (y0 * widthIn + x0) * 3;
        const int inputIdx01 = (y0 * widthIn + x1) * 3;
        const int inputIdx10 = (y1 * widthIn + x0) * 3;
        const int inputIdx11 = (y1 * widthIn + x1) * 3;

        // bilinear interpolation for each color channel
        for (int c = 0; c < 3; ++c) {
            const float interpolatedValue = 
                inImg[inputIdx00 + c] * w00 +
                inImg[inputIdx01 + c] * w01 +
                inImg[inputIdx10 + c] * w10 +
                inImg[inputIdx11 + c] * w11;

            outImg[outputIdx + c] = static_cast<unsigned char>(interpolatedValue);
        }
    }
}

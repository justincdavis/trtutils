extern "C" __global__
void letterboxResize(
    const unsigned char* __restrict__ inImg, // Input image
    unsigned char* __restrict__ outImg,      // Output image
    const int widthIn,                       // Input image width
    const int heightIn,                      // Input image height
    const int widthOut,                      // Output image width
    const int heightOut,                     // Output image height
    const int startX,                        // X-offset for the letterboxed region
    const int startY,                        // Y-offset for the letterboxed region
    const int regionWidth,                   // Width of the scaled region
    const int regionHeight                   // Height of the scaled region
) {
    // Get thread coordinates
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx == 0 && ty == 0) {
        printf("%d, %d, %d, %d, %d, %d, %d, %d\n", widthIn, heightIn, widthOut, heightOut, startX, startY, regionWidth, regionHeight);
    }

    // // Bounds check: Ensure valid threads
    // if (tx < widthOut && ty < heightOut) {
    //     const int outputIdx = (ty * widthOut + tx) * 3;
    //     outImg[outputIdx + 0] = 0;
    //     outImg[outputIdx + 1] = 0;
    //     outImg[outputIdx + 2] = 0;
    // }

    return;


    // Check if the thread falls within the active (non-padded) region
    if (tx >= startX && tx < startX + regionWidth &&
        ty >= startY && ty < startY + regionHeight) {
    
        // Scale factors for input to active region mapping
        const float scaleX = widthIn / (float)regionWidth;
        const float scaleY = heightIn / (float)regionHeight;

        // Calculate input image coordinates
        const float inputX = (tx - startX) * scaleX;
        const float inputY = (ty - startY) * scaleY;

        // Get four surrounding pixels (ensure indices are within bounds)
        const int x0 = max(0, min((int)floor(inputX), widthIn - 1));
        const int y0 = max(0, min((int)floor(inputY), heightIn - 1));
        const int x1 = max(0, min(x0 + 1, widthIn - 1));
        const int y1 = max(0, min(y0 + 1, heightIn - 1));

        // Interpolation weights
        const float dx = inputX - x0;
        const float dy = inputY - y0;
        const float w00 = (1.0f - dx) * (1.0f - dy);
        const float w01 = dx * (1.0f - dy);
        const float w10 = (1.0f - dx) * dy;
        const float w11 = dx * dy;

        // Input and output indices
        const int inputIdx00 = (y0 * widthIn + x0) * 3;
        const int inputIdx01 = (y0 * widthIn + x1) * 3;
        const int inputIdx10 = (y1 * widthIn + x0) * 3;
        const int inputIdx11 = (y1 * widthIn + x1) * 3;
        const int outputIdx = (ty * widthOut + tx) * 3;

        printf("Thread (%d, %d): x0=%d, y0=%d, x1=%d, y1=%d, scaleX=%.2f, scaleY=%.2f\n",
            tx, ty, x0, y0, x1, y1, scaleX, scaleY);

        return;

        // Perform bilinear interpolation for each color channel
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

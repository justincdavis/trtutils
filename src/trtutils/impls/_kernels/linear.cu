// extern "C" __global__
// void linearResize(
//     const unsigned char* __restrict__ inImg,
//     unsigned char* __restrict__ outImg,
//     const int inWidth,
//     const int inHeight,
//     const int outWidth,
//     const int outHeight
// ) {
//     const int tx = blockIdx.x * blockDim.x + threadIdx.x;
//     const int ty = blockIdx.y * blockDim.y + threadIdx.y;

//     const float scaleX = (float)outWidth / inWidth;
//     const float scaleY = (float)outHeight / inHeight;

//     if (tx < inHeight && ty < inWidth) {
//         const int inputBaseIdx = (tx * width + ty) * 3;

//         const int outX = static_cast<int>(outWidth * scaleX);
//         const int outY = static_cast<int>(outHeight * scaleY);

//         // Ensure input indices are within bounds
//         const int clampedInX = min(inX, inWidth - 1);
//         const int clampedInY = min(inY, inHeight - 1);

//         // Input and output indices
//         const int inputBaseIdx = (clampedInY * inWidth + clampedInX) * 3;
//         const int outputBaseIdx = (outY * outWidth + outX) * 3;

//         // // Copy pixel value directly (nearest neighbor)
//         // outImg[outputBaseIdx + 0] = inImg[inputBaseIdx + 0]; // B
//         // outImg[outputBaseIdx + 1] = inImg[inputBaseIdx + 1]; // G
//         // outImg[outputBaseIdx + 2] = inImg[inputBaseIdx + 2]; // R
//         for (int c = 0; c < 3; ++c) {
//             outImg[outputBaseIdx + c] = 114;
//         }
//     }
// }

extern "C" __global__
__global__ void linearResize( unsigned char* pIn, unsigned char* pOut, int widthIn, int heightIn, int widthOut, int heightOut)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if( tx < widthOut && ty < heightOut )
    {
        const int outputIdx = (tx * widthOut + ty) * 3;
        
        int txIn = tx * heightIn / heightOut;
        int tyIn = ty * widthIn / widthOut;
        int inputIdx = (txIn * widthIn + tyIn) * 3;

        printf("Thread (%d, %d), txIn = %d, tyIn = %d, inputIdx = %d, outputIdx = %d\n",
               tx, ty, txIn, tyIn, inputIdx, outputIdx);


        for(int c = 0; c < 3; c++)
            pOut[outputIdx + c] = pIn[inputIdx + c];
    }
}

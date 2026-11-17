#include <cuda_fp16.h>

extern "C" __global__
void bgrToBgrxInt8(
    const unsigned char* __restrict__ inImg,
    uint4* __restrict__ outImg,
    const int numPixels
) {
    const int quad = blockIdx.x * blockDim.x + threadIdx.x;
    if (quad * 4 >= numPixels) return;
    const unsigned char* px = inImg + quad * 12;
    unsigned int bgrx[4];
    for (int i = 0; i < 4; ++i) {
        bgrx[i] = (px[3 * i] | (px[3 * i + 1] << 8) | (px[3 * i + 2] << 16)) ^ 0x808080;
    }
    outImg[quad] = make_uint4(bgrx[0], bgrx[1], bgrx[2], bgrx[3]);
}

extern "C" __global__
void halfToFloat(
    const __half2* __restrict__ inData,
    float4* __restrict__ outData,
    const int numValues
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 4 >= numValues) return;
    const float2 low = __half22float2(inData[2 * idx]);
    const float2 high = __half22float2(inData[2 * idx + 1]);
    outData[idx] = make_float4(low.x, low.y, high.x, high.y);
}

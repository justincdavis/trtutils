#include <cuda_fp16.h>

#define TILE_DIM 32

extern "C" __global__
void scaleSwapTransposeImagenet(
    const unsigned char* __restrict__ inImg,
    float* __restrict__ outImg,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    const int height,
    const int width,
    const int batch_size
) {
    __shared__ float tileR[TILE_DIM][TILE_DIM + 1];
    __shared__ float tileG[TILE_DIM][TILE_DIM + 1];
    __shared__ float tileB[TILE_DIM][TILE_DIM + 1];

    int batch_idx = blockIdx.z;
    int in_col = blockIdx.x * TILE_DIM + threadIdx.x;
    int in_row = blockIdx.y * TILE_DIM + threadIdx.y;

    // compute batch offsets
    const int pixels_per_image = height * width;
    const unsigned char* batchInImg = inImg + batch_idx * pixels_per_image * 3;
    float* batchOutImg = outImg + batch_idx * 3 * pixels_per_image;

    if (in_row < height && in_col < width) {
        const int inputBaseIdx = (in_row * width + in_col) * 3;
        // convert to float in [0,1]
        float b = static_cast<float>(batchInImg[inputBaseIdx + 0]) / 255.0f;
        float g = static_cast<float>(batchInImg[inputBaseIdx + 1]) / 255.0f;
        float r = static_cast<float>(batchInImg[inputBaseIdx + 2]) / 255.0f;

        // normalize per channel: (x - mean) / std
        tileR[threadIdx.y][threadIdx.x] = (r - mean[0]) / std[0];
        tileG[threadIdx.y][threadIdx.x] = (g - mean[1]) / std[1];
        tileB[threadIdx.y][threadIdx.x] = (b - mean[2]) / std[2];
    }

    __syncthreads();

    int out_row = blockIdx.y * TILE_DIM + threadIdx.y;
    int out_col = blockIdx.x * TILE_DIM + threadIdx.x;

    if (out_row < height && out_col < width) {
        int outIdx = out_row * width + out_col;
        batchOutImg[outIdx + 0 * pixels_per_image] = tileR[threadIdx.y][threadIdx.x];
        batchOutImg[outIdx + 1 * pixels_per_image] = tileG[threadIdx.y][threadIdx.x];
        batchOutImg[outIdx + 2 * pixels_per_image] = tileB[threadIdx.y][threadIdx.x];
    }
}

extern "C" __global__
void scaleSwapTransposeImagenet_f16(
    const unsigned char* __restrict__ inImg,
    __half* __restrict__ outImg,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    const int height,
    const int width,
    const int batch_size
) {
    __shared__ float tileR[TILE_DIM][TILE_DIM + 1];
    __shared__ float tileG[TILE_DIM][TILE_DIM + 1];
    __shared__ float tileB[TILE_DIM][TILE_DIM + 1];

    int batch_idx = blockIdx.z;
    int in_col = blockIdx.x * TILE_DIM + threadIdx.x;
    int in_row = blockIdx.y * TILE_DIM + threadIdx.y;

    // compute batch offsets
    const int pixels_per_image = height * width;
    const unsigned char* batchInImg = inImg + batch_idx * pixels_per_image * 3;
    __half* batchOutImg = outImg + batch_idx * 3 * pixels_per_image;

    if (in_row < height && in_col < width) {
        const int inputBaseIdx = (in_row * width + in_col) * 3;
        // convert to float in [0,1]
        float b = static_cast<float>(batchInImg[inputBaseIdx + 0]) / 255.0f;
        float g = static_cast<float>(batchInImg[inputBaseIdx + 1]) / 255.0f;
        float r = static_cast<float>(batchInImg[inputBaseIdx + 2]) / 255.0f;

        // normalize per channel: (x - mean) / std
        tileR[threadIdx.y][threadIdx.x] = (r - mean[0]) / std[0];
        tileG[threadIdx.y][threadIdx.x] = (g - mean[1]) / std[1];
        tileB[threadIdx.y][threadIdx.x] = (b - mean[2]) / std[2];
    }

    __syncthreads();

    int out_row = blockIdx.y * TILE_DIM + threadIdx.y;
    int out_col = blockIdx.x * TILE_DIM + threadIdx.x;

    if (out_row < height && out_col < width) {
        int outIdx = out_row * width + out_col;
        batchOutImg[outIdx + 0 * pixels_per_image] = __float2half(tileR[threadIdx.y][threadIdx.x]);
        batchOutImg[outIdx + 1 * pixels_per_image] = __float2half(tileG[threadIdx.y][threadIdx.x]);
        batchOutImg[outIdx + 2 * pixels_per_image] = __float2half(tileB[threadIdx.y][threadIdx.x]);
    }
}

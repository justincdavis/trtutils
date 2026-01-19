#include <cuda_fp16.h>

#define TILE_DIM 32

extern "C" __global__
void scaleSwapTranspose_opt(
    const unsigned char* __restrict__ inImg,
    float* __restrict__ outImg,
    const float scale,
    const float offset,
    const int height,
    const int width,
    const int batch_size
) {
    // use a set of shared memory tiles to store intermediate values
    // much more effective use of memory bandwidth, since we stream in memory
    // then we write memory in two separate chunks
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

    // identical to the reading setup in the simple sst.cu kernel
    if (in_row < height && in_col < width) {
        const int inputBaseIdx = (in_row * width + in_col) * 3;
        float b = static_cast<float>(batchInImg[inputBaseIdx + 0]) * scale + offset;
        float g = static_cast<float>(batchInImg[inputBaseIdx + 1]) * scale + offset;
        float r = static_cast<float>(batchInImg[inputBaseIdx + 2]) * scale + offset;

        tileR[threadIdx.y][threadIdx.x] = r;
        tileG[threadIdx.y][threadIdx.x] = g;
        tileB[threadIdx.y][threadIdx.x] = b;
    }

    // wait for all threads so we have read then write
    __syncthreads();

    // compute extra output coords, then use same writing as the simple sst.cu kernel
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
void scaleSwapTranspose_opt_f16(
    const unsigned char* __restrict__ inImg,
    __half* __restrict__ outImg,
    const float scale,
    const float offset,
    const int height,
    const int width,
    const int batch_size
) {
    // use a set of shared memory tiles to store intermediate values
    // much more effective use of memory bandwidth, since we stream in memory
    // then we write memory in two separate chunks
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

    // identical to the reading setup in the simple sst.cu kernel
    if (in_row < height && in_col < width) {
        const int inputBaseIdx = (in_row * width + in_col) * 3;
        float b = static_cast<float>(batchInImg[inputBaseIdx + 0]) * scale + offset;
        float g = static_cast<float>(batchInImg[inputBaseIdx + 1]) * scale + offset;
        float r = static_cast<float>(batchInImg[inputBaseIdx + 2]) * scale + offset;

        tileR[threadIdx.y][threadIdx.x] = r;
        tileG[threadIdx.y][threadIdx.x] = g;
        tileB[threadIdx.y][threadIdx.x] = b;
    }

    // wait for all threads so we have read then write
    __syncthreads();

    // compute extra output coords, then use same writing as the simple sst.cu kernel
    int out_row = blockIdx.y * TILE_DIM + threadIdx.y;
    int out_col = blockIdx.x * TILE_DIM + threadIdx.x;

    if (out_row < height && out_col < width) {
        int outIdx = out_row * width + out_col;
        batchOutImg[outIdx + 0 * pixels_per_image] = __float2half(tileR[threadIdx.y][threadIdx.x]);
        batchOutImg[outIdx + 1 * pixels_per_image] = __float2half(tileG[threadIdx.y][threadIdx.x]);
        batchOutImg[outIdx + 2 * pixels_per_image] = __float2half(tileB[threadIdx.y][threadIdx.x]);
    }
}

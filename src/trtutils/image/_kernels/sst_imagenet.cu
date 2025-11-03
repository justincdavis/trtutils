#define TILE_DIM 32

extern "C" __global__
void scaleSwapTransposeImagenet(
    const unsigned char* __restrict__ inImg,
    float* __restrict__ outImg,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    const int shape
) {
    __shared__ float tileR[TILE_DIM][TILE_DIM + 1];
    __shared__ float tileG[TILE_DIM][TILE_DIM + 1];
    __shared__ float tileB[TILE_DIM][TILE_DIM + 1];

    int in_col = blockIdx.x * TILE_DIM + threadIdx.x;
    int in_row = blockIdx.y * TILE_DIM + threadIdx.y;

    if (in_row < shape && in_col < shape) {
        const int inputBaseIdx = (in_row * shape + in_col) * 3;
        // convert to float in [0,1]
        float b = static_cast<float>(inImg[inputBaseIdx + 0]) / 255.0f;
        float g = static_cast<float>(inImg[inputBaseIdx + 1]) / 255.0f;
        float r = static_cast<float>(inImg[inputBaseIdx + 2]) / 255.0f;

        // normalize per channel: (x - mean) / std
        tileR[threadIdx.y][threadIdx.x] = (r - mean[0]) / std[0];
        tileG[threadIdx.y][threadIdx.x] = (g - mean[1]) / std[1];
        tileB[threadIdx.y][threadIdx.x] = (b - mean[2]) / std[2];
    }

    __syncthreads();

    int out_row = blockIdx.y * TILE_DIM + threadIdx.y;
    int out_col = blockIdx.x * TILE_DIM + threadIdx.x;

    if (out_row < shape && out_col < shape) {
        int outIdx = out_row * shape + out_col;
        const int shapeSq = shape * shape;
        outImg[outIdx + 0 * shapeSq] = tileR[threadIdx.y][threadIdx.x];
        outImg[outIdx + 1 * shapeSq] = tileG[threadIdx.y][threadIdx.x];
        outImg[outIdx + 2 * shapeSq] = tileB[threadIdx.y][threadIdx.x];
    }
}

#define TILE_DIM 32

extern "C" __global__
void scaleSwapTranspose_opt(
    const unsigned char* __restrict__ inImg,
    float* __restrict__ outImg,
    const float scale,
    const float offset,
    const int shape
) {
    // use a set of shared memory tiles to store intermediate values
    // much more effective use of memory bandwidth, since we stream in memory
    // then we write memory in two separate chunks
    __shared__ float tileR[TILE_DIM][TILE_DIM + 1];
    __shared__ float tileG[TILE_DIM][TILE_DIM + 1];
    __shared__ float tileB[TILE_DIM][TILE_DIM + 1];

    int in_col = blockIdx.x * TILE_DIM + threadIdx.x;
    int in_row = blockIdx.y * TILE_DIM + threadIdx.y;

    // identical to the reading setup in the simple sst.cu kernel
    if (in_row < shape && in_col < shape) {
        const int inputBaseIdx = (in_row * shape + in_col) * 3;
        float b = static_cast<float>(inImg[inputBaseIdx + 0]) * scale + offset;
        float g = static_cast<float>(inImg[inputBaseIdx + 1]) * scale + offset;
        float r = static_cast<float>(inImg[inputBaseIdx + 2]) * scale + offset;

        tileR[threadIdx.y][threadIdx.x] = r;
        tileG[threadIdx.y][threadIdx.x] = g;
        tileB[threadIdx.y][threadIdx.x] = b;
    }

    // wait for all threads so we have read then write
    __syncthreads();

    // compute extra output coords, then use same writing as the simple sst.cu kernel
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

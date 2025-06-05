#define TILE_DIM 32

extern "C" __global__
void scaleSwapTranspose_opt(
    const unsigned char* __restrict__ inImg,
    float* __restrict__ outImg,
    const float scale,
    const float offset,
    const int shape
) {
    // Declare shared memory statically for each channel.
    __shared__ float tileR[TILE_DIM][TILE_DIM + 1];
    __shared__ float tileG[TILE_DIM][TILE_DIM + 1];
    __shared__ float tileB[TILE_DIM][TILE_DIM + 1];

    // Compute the input coordinates
    int in_col = blockIdx.x * TILE_DIM + threadIdx.x;
    int in_row = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load pixel from global memory if it's within the image bounds.
    if (in_row < shape && in_col < shape) {
        // Calculate the index for this pixel for interleaved BGR data.
        const int inputBaseIdx = (in_row * shape + in_col) * 3;
        // Perform scaling and offset conversion.
        float b = static_cast<float>(inImg[inputBaseIdx + 0]) * scale + offset;
        float g = static_cast<float>(inImg[inputBaseIdx + 1]) * scale + offset;
        float r = static_cast<float>(inImg[inputBaseIdx + 2]) * scale + offset;

        // Write the channels into shared memory.
        // The extra column (+1) avoids bank conflicts.
        tileR[threadIdx.y][threadIdx.x] = r;
        tileG[threadIdx.y][threadIdx.x] = g;
        tileB[threadIdx.y][threadIdx.x] = b;
    }

    // Synchronize to ensure the tile is fully loaded.
    __syncthreads();

    // Compute output coordinates - no transpose needed, output in same order as input
    int out_row = blockIdx.y * TILE_DIM + threadIdx.y;
    int out_col = blockIdx.x * TILE_DIM + threadIdx.x;

    if (out_row < shape && out_col < shape) {
        // Calculate the output base index in the planar (NCHW) format.
        int outIdx = out_row * shape + out_col;
        const int shapeSq = shape * shape;
        // Read from shared memory and write to output in RGB order (swap BGR to RGB)
        outImg[outIdx + 0 * shapeSq] = tileR[threadIdx.y][threadIdx.x];
        outImg[outIdx + 1 * shapeSq] = tileG[threadIdx.y][threadIdx.x];
        outImg[outIdx + 2 * shapeSq] = tileB[threadIdx.y][threadIdx.x];
    }
}

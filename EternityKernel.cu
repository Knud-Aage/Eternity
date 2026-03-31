// EternityKernel.cu
// Compilation: nvcc -m64 -arch=sm_75 -ptx -D_ALLOW_COMPILER_AND_HEADER_MISMATCH EternityKernel.cu -o EternityKernel.ptx

extern "C" __global__ void validateMacroTiles(
    const int* candidates,    // Flattened: [numCandidates * 16]
    int* results,             // Output buffer for valid 16-piece sets
    int* resultCounter,       // Atomic counter
    int numCandidates,
    int maxResults)
{
    int tid = blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (tid >= numCandidates) return;

    int base = tid * 16;

    // Internal Horizontal Checks (Checking East of left piece against West of right piece)
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 3; c++) {
            int leftIdx = base + (r * 4) + c;
            int rightIdx = leftIdx + 1;

            unsigned int leftPiece = (unsigned int)candidates[leftIdx];
            unsigned int rightPiece = (unsigned int)candidates[rightIdx];

            // If the colors do not match exactly, kill this thread. It's a bad 4x4 block.
            if (((leftPiece >> 16) & 0xFF) != (rightPiece & 0xFF)) return;
        }
    }

    // Internal Vertical Checks (Checking South of top piece against North of bottom piece)
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 3; r++) {
            int topIdx = base + (r * 4) + c;
            int bottomIdx = topIdx + 4;

            unsigned int topPiece = (unsigned int)candidates[topIdx];
            unsigned int bottomPiece = (unsigned int)candidates[bottomIdx];

            // If the colors do not match exactly, kill this thread.
            if (((topPiece >> 8) & 0xFF) != ((bottomPiece >> 24) & 0xFF)) return;
        }
    }

    // If we survived all the loops, the 4x4 block is internally perfect!
    // Safely add it to the results array so the Java CPU can use it.
    int pos = atomicAdd(resultCounter, 1);
    if (pos < maxResults) {
        int resBase = pos * 16;
        for (int i = 0; i < 16; i++) {
            results[resBase + i] = candidates[base + i];
        }
    }
}
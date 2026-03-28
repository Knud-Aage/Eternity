// EternityKernel.cu
// Compile with: nvcc -ptx EternityKernel.cu -o EternityKernel.ptx

extern "C"
__global__ void validateMacroTiles(
    const int* candidates,    // Flattened: [numCandidates * 16]
    int* results,             // Output buffer for valid 16-piece sets
    int* resultCounter,       // Atomic counter
    int numCandidates,
    int maxResults)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numCandidates) return;

    int base = tid * 16;

    // Internal Horizontal Checks
    // Row 0: 0-1, 1-2, 2-3 | Row 1: 4-5, 5-6, 6-7 ...
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 3; c++) {
            int leftIdx = base + (r * 4) + c;
            int rightIdx = leftIdx + 1;

            // Left East == Right West
            // East is (p >>> 16) & 0xFF, West is p & 0xFF
            if (((candidates[leftIdx] >> 16) & 0xFF) != (candidates[rightIdx] & 0xFF)) return;
        }
    }

    // Internal Vertical Checks
    // Col 0: 0-4, 4-8, 8-12 | Col 1: 1-5, 5-9, 9-13 ...
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 3; r++) {
            int topIdx = base + (r * 4) + c;
            int bottomIdx = topIdx + 4;

            // Top South == Bottom North
            // South is (p >>> 8) & 0xFF, North is (p >>> 24) & 0xFF
            if (((candidates[topIdx] >> 8) & 0xFF) != ((candidates[bottomIdx] >> 24) & 0xFF)) return;
        }
    }

    // If we reached here, the 4x4 is internally consistent
    int pos = atomicAdd(resultCounter, 1);
    if (pos < maxResults) {
        for (int i = 0; i < 16; i++) {
            results[pos * 16 + i] = candidates[base + i];
        }
    }
}

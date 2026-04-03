extern "C"

// --- DEVICE HELPER FUNCTIONS ---
// These run purely on the graphics card. We extract the 8-bit color codes from the 32-bit packed integers.
__device__ inline int getNorth(int p) { return (p >> 24) & 0xFF; }
__device__ inline int getEast(int p)  { return (p >> 16) & 0xFF; }
__device__ inline int getSouth(int p) { return (p >> 8)  & 0xFF; }
__device__ inline int getWest(int p)  { return (p)       & 0xFF; }

__device__ inline bool matches(int p, int n_req, int e_req, int s_req, int w_req) {
    if (n_req != 255 && getNorth(p) != n_req) return false;
    if (e_req != 255 && getEast(p)  != e_req) return false;
    if (s_req != 255 && getSouth(p) != s_req) return false;
    if (w_req != 255 && getWest(p)  != w_req) return false;
    return true;
}

// --- MAIN PIECE-BY-PIECE KERNEL ---
__global__ void solvePBP(
    int* d_partialBoards,      // Input: The 50,000 starting setups the CPU gave us
    int numPartialBoards,      // How many boards are in the array
    int startingPos,           // The index where the GPU should start solving (e.g., pos 150)
    int* d_allOrientations,    // The 1024 array of packed piece colors
    int* d_physicalMapping,    // The 1024 array mapping to physical IDs (0-255)
    int* d_solution,           // Output: The winning 256-piece array
    int* d_solvedFlag          // Global flag so the winner can tell everyone else to stop!
) {
    // Determine which partial board this specific thread is working on
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPartialBoards) return;

    // 1. Thread-Local Memory (Registers/L1 Cache)
    int board[256];
    int pieceStack[256];       // Remembers which orientation (0-1023) we left off at for each position

    // 256 bits representing physical pieces. 1 = Available, 0 = Used.
    unsigned long long inventoryMask[4] = { ~0ULL, ~0ULL, ~0ULL, ~0ULL };

    // 2. Initialize this thread's board from global memory
    int offset = tid * 256;
    for (int i = 0; i < 256; i++) {
        board[i] = d_partialBoards[offset + i];
        pieceStack[i] = 0; // Reset search stack

        // If the CPU pre-placed a piece here, mark it as USED in our bitboard
        if (i < startingPos && board[i] != -1) {
            // We have to reverse-lookup the physical ID.
            // (For speed, the CPU usually passes the initial bitboard, but we do it here for clarity)
            for(int o = 0; o < 1024; o++) {
                if(d_allOrientations[o] == board[i]) {
                    int physId = d_physicalMapping[o];
                    inventoryMask[physId / 64] &= ~(1ULL << (physId % 64)); // Mark Used
                    break;
                }
            }
        }
    }

    int pos = startingPos;

    // 3. THE ITERATIVE BACKTRACKING LOOP (No Recursion!)
    while (pos >= startingPos && pos < 256) {

        // Check if another thread already won. If so, immediately kill this thread.
        if (*d_solvedFlag == 1) return;

        // If we hit the pre-placed centerpiece at index 119, just skip over it
        if (pos == 119) {
            pos++;
            continue;
        }

        int row = pos / 16;
        int col = pos % 16;

        // Dynamic Constraints (255 = Wildcard)
        int n_req = (row == 0) ? 0 : (board[pos - 16] != -1 ? getSouth(board[pos - 16]) : 255);
        int s_req = (row == 15) ? 0 : 255;
        int w_req = (col == 0) ? 0 : (board[pos - 1] != -1 ? getEast(board[pos - 1]) : 255);
        int e_req = (col == 15) ? 0 : 255;

        bool foundPiece = false;
        int startIdx = pieceStack[pos];

        // Search the inventory starting from where we left off
        for (int idx = startIdx; idx < 1024; idx++) {
            int physId = d_physicalMapping[idx];

            // BITWISE CHECK: Is this physical piece available?
            if (inventoryMask[physId / 64] & (1ULL << (physId % 64))) {

                int p = d_allOrientations[idx];

                // MATCHING CHECK
                if (matches(p, n_req, e_req, s_req, w_req)) {

                    // IT FITS! Lock it in.
                    board[pos] = p;
                    pieceStack[pos] = idx + 1; // Remember to check the next piece if we backtrack here

                    // Mark as USED
                    inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));

                    foundPiece = true;
                    pos++; // Move forward!
                    break;
                }
            }
        }

        // BACKTRACKING: No pieces fit in this spot.
        if (!foundPiece) {
            pieceStack[pos] = 0; // Reset search for this spot
            pos--;               // Step backwards

            // Skip back over the centerpiece if we hit it in reverse
            if (pos == 119) pos--;

            // Un-mark the piece we just removed so it becomes AVAILABLE again
            if (pos >= startingPos) {
                int pToUndo = board[pos];
                board[pos] = -1;

                // Find physical ID to unmark
                for(int o = 0; o < 1024; o++) {
                    if(d_allOrientations[o] == pToUndo) {
                        int physId = d_physicalMapping[o];
                        inventoryMask[physId / 64] |= (1ULL << (physId % 64)); // Mark Available
                        break;
                    }
                }
            }
        }
    }

    // 4. WIN CONDITION
    if (pos == 256) {
        // Atomic lock to ensure only the first thread to finish writes the solution
        if (atomicExch(d_solvedFlag, 1) == 0) {
            for (int i = 0; i < 256; i++) {
                d_solution[i] = board[i];
            }
        }
    }
}
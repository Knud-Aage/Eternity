// --- DEVICE HELPER FUNCTIONS ---
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
extern "C" __global__ void solvePBP(
    int* d_partialBoards,
    int numPartialBoards,
    int startingPos,
    int* d_allOrientations,
    int* d_physicalMapping,
    int* d_solution,
    int* d_solvedFlag,
    int* d_gpuHighScore,
    int* d_bestBoardOut
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPartialBoards) return;

    int board[256];
    int pieceStack[256];
    unsigned long long inventoryMask[4] = { ~0ULL, ~0ULL, ~0ULL, ~0ULL };

    // --- NEW: PRE-CALCULATE PIECE COLORS FOR FAST AUDITING ---
    // This allows the thread to instantly know what colors are left in the inventory
    int physColors[256][4];
    for (int i = 0; i < 1024; i++) {
        int phys = d_physicalMapping[i];
        int val = d_allOrientations[i];
        physColors[phys][0] = getNorth(val);
        physColors[phys][1] = getEast(val);
        physColors[phys][2] = getSouth(val);
        physColors[phys][3] = getWest(val);
    }

    int offset = tid * 256;
    for (int i = 0; i < 256; i++) {
        board[i] = d_partialBoards[offset + i];
        pieceStack[i] = 0;

        if (i < startingPos && board[i] != -1) {
            for(int o = 0; o < 1024; o++) {
                if(d_allOrientations[o] == board[i]) {
                    int physId = d_physicalMapping[o];
                    inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));
                    break;
                }
            }
        }
    }

    int pos = startingPos;
    int localMax = startingPos;
    int bestLocalBoard[256];
    unsigned int stepCounter = 0;

    // 3. THE ITERATIVE BACKTRACKING LOOP
    while (pos >= startingPos && pos < 256) {

        stepCounter++;
        if (stepCounter > 5000000) break; // Timebox Kill Switch
        if (*d_solvedFlag == 1) return;

        if (pos == 119) {
            pos++;
            continue;
        }

        int row = pos / 16;
        int col = pos % 16;

        int n_req = (row == 0) ? 0 : (board[pos - 16] != -1 ? getSouth(board[pos - 16]) : 255);
        int s_req = (row == 15) ? 0 : 255;
        int w_req = (col == 0) ? 0 : (board[pos - 1] != -1 ? getEast(board[pos - 1]) : 255);
        int e_req = (col == 15) ? 0 : 255;

        bool foundPiece = false;
        int startIdx = pieceStack[pos];

        for (int idx = startIdx; idx < 1024; idx++) {
            int physId = d_physicalMapping[idx];

            if (inventoryMask[physId / 64] & (1ULL << (physId % 64))) {

                int p = d_allOrientations[idx];

                if (matches(p, n_req, e_req, s_req, w_req)) {

                    // Temporarily place the piece to run the audit
                    board[pos] = p;
                    inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));

                    // --- THE LOOK-AHEAD HEURISTIC (Color Starvation) ---
                    bool doomed = false;

                    // Only run this heavy audit if we just finished a full row!
                    if ((pos + 1) % 16 == 0 && (pos + 1) < 240) {
                        int exposed[32] = {0};
                        int available[32] = {0};

                        // 1. Count exposed South colors of the row we just finished
                        for (int i = (pos + 1) - 16; i <= pos; i++) {
                            exposed[getSouth(board[i])]++;
                        }

                        // 2. Count the colors of all remaining unused pieces
                        for (int pId = 0; pId < 256; pId++) {
                            if (inventoryMask[pId / 64] & (1ULL << (pId % 64))) {
                                available[physColors[pId][0]]++;
                                available[physColors[pId][1]]++;
                                available[physColors[pId][2]]++;
                                available[physColors[pId][3]]++;
                            }
                        }

                        // 3. The Kill Switch! (Ignore Color 0, which is the border)
                        for (int c = 1; c < 32; c++) {
                            if (exposed[c] > available[c]) {
                                doomed = true;
                                break;
                            }
                        }
                    }

                    if (doomed) {
                        // Starvation detected! Undo this piece and try the next one in the inventory.
                        board[pos] = -1;
                        inventoryMask[physId / 64] |= (1ULL << (physId % 64));
                        continue;
                    }
                    // --- END HEURISTIC ---

                    // The piece passed the audit! Lock it in.
                    pieceStack[pos] = idx + 1;
                    foundPiece = true;
                    pos++;

                    // Snapshot the new record!
                    if (pos > localMax) {
                        localMax = pos;
                        for (int i = 0; i < pos; i++) bestLocalBoard[i] = board[i];
                    }

                    break;
                }
            }
        }

        if (!foundPiece) {
            pieceStack[pos] = 0;
            pos--;

            if (pos == 119) pos--;

            if (pos >= startingPos) {
                int pToUndo = board[pos];
                board[pos] = -1;

                for(int o = 0; o < 1024; o++) {
                    if(d_allOrientations[o] == pToUndo) {
                        int physId = d_physicalMapping[o];
                        inventoryMask[physId / 64] |= (1ULL << (physId % 64));
                        break;
                    }
                }
            }
        }
    }

    // 4. WIN CONDITION
    if (pos == 256) {
        if (atomicExch(d_solvedFlag, 1) == 0) {
            for (int i = 0; i < 256; i++) d_solution[i] = board[i];
        }
    }

    // 5. COMPARE-AND-SWAP GATE
    int currentMax = *d_gpuHighScore;
    while (localMax > currentMax) {
        if (atomicCAS(d_gpuHighScore, currentMax, localMax) == currentMax) {
            for (int i = 0; i < localMax; i++) d_bestBoardOut[i] = bestLocalBoard[i];
            for (int i = localMax; i < 256; i++) d_bestBoardOut[i] = -1;
            break;
        }
        currentMax = *d_gpuHighScore;
    }
}
// --- DEVICE HELPER FUNCTIONS ---
__device__ inline int getNorth(int p) { return (p >> 24) & 0xFF; }
__device__ inline int getEast(int p)  { return (p >> 16) & 0xFF; }
__device__ inline int getSouth(int p) { return (p >> 8)  & 0xFF; }
__device__ inline int getWest(int p)  { return (p)       & 0xFF; }

// <--- NEW: Passed row and col into the matches function --->
// __device__ inline bool matches(int p, int n_req, int e_req, int s_req, int w_req, int row, int col) {
//     int n = getNorth(p);
//     int e = getEast(p);
//     int s = getSouth(p);
//     int w = getWest(p);
//
//     // 1. Check if the piece physically fits the adjacent placed pieces
//     if (n_req != 255 && n != n_req) return false;
//     if (e_req != 255 && e != e_req) return false;
//     if (s_req != 255 && s != s_req) return false;
//     if (w_req != 255 && w != w_req) return false;
//
//     // 2. THE ABSOLUTE DOMAIN RULES (Derived from the official palette)
//
//     // NORTH Edge
//     if (row == 0) {
//         if (n != 0) return false; // Top edge MUST be grey
//     } else {
//         if (col == 0 || col == 15) {
//             if (n < 1 || n > 5) return false; // Vertical border track MUST be 1-5
//         } else {
//             if (n < 6) return false; // True interior MUST be 6 or higher
//         }
//     }
//
//     // SOUTH Edge
//     if (row == 15) {
//         if (s != 0) return false; // Bottom edge MUST be grey
//     } else {
//         if (col == 0 || col == 15) {
//             if (s < 1 || s > 5) return false; // Vertical border track
//         } else {
//             if (s < 6) return false; // True interior
//         }
//     }
//
//     // WEST Edge
//     if (col == 0) {
//         if (w != 0) return false; // Left edge MUST be grey
//     } else {
//         if (row == 0 || row == 15) {
//             if (w < 1 || w > 5) return false; // Horizontal border track
//         } else {
//             if (w < 6) return false; // True interior
//         }
//     }
//
//     // EAST Edge
//     if (col == 15) {
//         if (e != 0) return false; // Right edge MUST be grey
//     } else {
//         if (row == 0 || row == 15) {
//             if (e < 1 || e > 5) return false; // Horizontal border track
//         } else {
//             if (e < 6) return false; // True interior
//         }
//     }
//
//     return true;
// }
__device__ inline bool matches(int p, int n_req, int e_req, int s_req, int w_req, int row, int col) {
    int n = getNorth(p);
    int e = getEast(p);
    int s = getSouth(p);
    int w = getWest(p);

    // 1. Check if the piece physically fits the adjacent placed pieces
    if (n_req != 255 && n != n_req) return false;
    if (e_req != 255 && e != e_req) return false;
    if (s_req != 255 && s != s_req) return false;
    if (w_req != 255 && w != w_req) return false;

    // 2. THE BORDER PATROL (Grey edges ONLY)
    // We removed the 1-5 / 6-22 rule so we don't accidentally reject valid CSV pieces!
    if (row != 0 && n == 0) return false;
    if (col != 15 && e == 0) return false;
    if (row != 15 && s == 0) return false;
    if (col != 0 && w == 0) return false;

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

    while (pos >= startingPos && pos < 256) {

        stepCounter++;
        if (stepCounter > 5000000) break;
        if (*d_solvedFlag == 1) return;

        if (pos == 135) {
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

                // <--- NEW: Passed row and col here! --->
                if (matches(p, n_req, e_req, s_req, w_req, row, col)) {

                    board[pos] = p;
                    inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));

                    bool doomed = false;

                    if ((pos + 1) % 16 == 0 && (pos + 1) < 240) {
                        int exposed[32] = {0};
                        int available[32] = {0};

                        for (int i = (pos + 1) - 16; i <= pos; i++) {
                            exposed[getSouth(board[i])]++;
                        }

                        for (int pId = 0; pId < 256; pId++) {
                            if (inventoryMask[pId / 64] & (1ULL << (pId % 64))) {
                                available[physColors[pId][0]]++;
                                available[physColors[pId][1]]++;
                                available[physColors[pId][2]]++;
                                available[physColors[pId][3]]++;
                            }
                        }

                        for (int c = 1; c < 32; c++) {
                            if (exposed[c] > available[c]) {
                                doomed = true;
                                break;
                            }
                        }
                    }

                    if (doomed) {
                        board[pos] = -1;
                        inventoryMask[physId / 64] |= (1ULL << (physId % 64));
                        continue;
                    }

                    pieceStack[pos] = idx + 1;
                    foundPiece = true;
                    pos++;

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

            if (pos == 135) pos--;

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

    if (pos == 256) {
        if (atomicExch(d_solvedFlag, 1) == 0) {
            for (int i = 0; i < 256; i++) d_solution[i] = board[i];
        }
    }

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
__constant__ int c_allOrientations[1024];
__constant__ int c_physicalMapping[1024];

__device__ inline int getNorth(int p) { return (p >> 24) & 0xFF; }
__device__ inline int getEast (int p) { return (p >> 16) & 0xFF; }
__device__ inline int getSouth(int p) { return (p >>  8) & 0xFF; }
__device__ inline int getWest (int p) { return (p)       & 0xFF; }

#define WILDCARD 255
#define NUM_COLORS 23
#define MAX_PER_COLOR 128

__device__ inline bool matches(int p, int n_req, int e_req, int s_req, int w_req, int row, int col) {
    int n = getNorth(p), e = getEast(p), s = getSouth(p), w = getWest(p);
    if (n_req != WILDCARD && n != n_req) return false;
    if (e_req != WILDCARD && e != e_req) return false;
    if (s_req != WILDCARD && s != s_req) return false;
    if (w_req != WILDCARD && w != w_req) return false;
    if (row != 0  && n == 0) return false;
    if (col != 15 && e == 0) return false;
    if (row != 15 && s == 0) return false;
    if (col != 0  && w == 0) return false;
    return true;
}

__device__ bool hasCandidate(
    int n_req, int e_req, int s_req, int w_req,
    int row, int col,
    const unsigned long long* inventoryMask,
    const short* sm_byNorth,
    const short* sm_byNorthCount
) {
    if (n_req != WILDCARD && n_req < NUM_COLORS) {
        int count = sm_byNorthCount[n_req];
        for (int i = 0; i < count; i++) {
            int idx = sm_byNorth[n_req * MAX_PER_COLOR + i];
            int physId = c_physicalMapping[idx];
            if (!(inventoryMask[physId / 64] & (1ULL << (physId % 64)))) continue;
            int p = c_allOrientations[idx];
            if (matches(p, n_req, e_req, s_req, w_req, row, col)) return true;
        }
        return false;
    }

    for (int idx = 0; idx < 1024; idx++) {
        int physId = c_physicalMapping[idx];
        if (!(inventoryMask[physId / 64] & (1ULL << (physId % 64)))) continue;
        int p = c_allOrientations[idx];
        if (matches(p, n_req, e_req, s_req, w_req, row, col)) return true;
    }
    return false;
}

extern "C" __global__ void solvePBP(
    const int* d_partialBoards,
    int numPartialBoards,
    int startingStep,
    const int* d_buildOrder,
    int* d_solution,
    int* d_solvedFlag,
    int* d_gpuHighScore,
    int* d_bestBoardOut,
    unsigned long long* d_totalSteps,
    int lockCenterFlag,
    int* d_threadDepths,
    unsigned long long stepBudget
) {
    __shared__ short sm_byNorth[NUM_COLORS * MAX_PER_COLOR];
    __shared__ short sm_byNorthCount[NUM_COLORS];

    if (threadIdx.x == 0) {
        for (int c = 0; c < NUM_COLORS; c++) sm_byNorthCount[c] = 0;
        for (int i = 0; i < 1024; i++) {
            int p = c_allOrientations[i];
            int nc = getNorth(p);
            if (nc < NUM_COLORS) {
                int cnt = sm_byNorthCount[nc];
                if (cnt < MAX_PER_COLOR) {
                    sm_byNorth[nc * MAX_PER_COLOR + cnt] = (short)i;
                    sm_byNorthCount[nc] = cnt + 1;
                }
            }
        }
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPartialBoards) return;

    int board[256];
    int boardIdxHistory[256];
    int pieceStack[256];
    unsigned long long inventoryMask[4] = { ~0ULL, ~0ULL, ~0ULL, ~0ULL };

    int offset = tid * 256;
    for (int i = 0; i < 256; i++) {
        board[i] = d_partialBoards[offset + i];
        pieceStack[i] = 0;
        boardIdxHistory[i] = -1;

        if (board[i] != -1) {
            for (int o = 0; o < 1024; o++) {
                if (c_allOrientations[o] == board[i]) {
                    int physId = c_physicalMapping[o];
                    boardIdxHistory[i] = o;
                    inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));
                    break;
                }
            }
        }
    }

    int step = startingStep;
    int maxStepReached = startingStep;
    int bestPiecesPlaced = 0;
    int bestLocalBoard[256];
    unsigned long long stepCounter = 0;

    int seedPiecesPlaced = 0;
    for (int i = 0; i < 256; i++) {
        if (board[i] != -1) seedPiecesPlaced++;
    }
    bestPiecesPlaced = seedPiecesPlaced;

    while (step >= startingStep && step < 256) {
        if (stepCounter >= stepBudget) break;
        if (*d_solvedFlag == 1) break;
        stepCounter++;

        int boardIdx = d_buildOrder[step];

        if (lockCenterFlag == 1 && (boardIdx == 135 || boardIdx == 221 || boardIdx == 45 || boardIdx == 210 || boardIdx == 34)) {
            step++;
            continue;
        }

        int row = boardIdx / 16;
        int col = boardIdx % 16;

        int n_req = (row == 0)  ? 0 : (board[boardIdx - 16] != -1 ? getSouth(board[boardIdx - 16]) : WILDCARD);
        int s_req = (row == 15) ? 0 : (board[boardIdx + 16] != -1 ? getNorth(board[boardIdx + 16]) : WILDCARD);
        int w_req = (col == 0)  ? 0 : (board[boardIdx - 1]  != -1 ? getEast (board[boardIdx - 1])  : WILDCARD);
        int e_req = (col == 15) ? 0 : (board[boardIdx + 1]  != -1 ? getWest (board[boardIdx + 1])  : WILDCARD);

        bool foundPiece = false;
        int startIdx = pieceStack[step];

        for (int idx = startIdx; idx < 1024; idx++) {
            int physId = c_physicalMapping[idx];
            if (!(inventoryMask[physId / 64] & (1ULL << (physId % 64)))) continue;

            int p = c_allOrientations[idx];
            if (!matches(p, n_req, e_req, s_req, w_req, row, col)) continue;

            if (row < 15 && board[boardIdx + 16] == -1) {
                int south_n_req = getSouth(p);
                int south_w_req = (col > 0 && board[boardIdx + 15] != -1) ? getEast(board[boardIdx + 15]) : WILDCARD;
                int south_e_req = (col == 15) ? 0 : WILDCARD;
                int south_s_req = (row == 14) ? 0 : WILDCARD;

                inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));
                bool southOk = hasCandidate(south_n_req, south_e_req, south_s_req, south_w_req,
                                            row + 1, col, inventoryMask, sm_byNorth, sm_byNorthCount);
                inventoryMask[physId / 64] |= (1ULL << (physId % 64));

                if (!southOk) continue; 
            }

            if (col < 15 && board[boardIdx + 1] == -1) {
                int east_w_req = getEast(p);
                int east_n_req = (row > 0 && board[boardIdx - 15] != -1) ? getSouth(board[boardIdx - 15]) : WILDCARD;
                int east_s_req = (row == 15) ? 0 : WILDCARD;
                int east_e_req = (col == 14) ? 0 : WILDCARD;

                inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));
                bool eastOk = hasCandidate(east_n_req, east_e_req, east_s_req, east_w_req,
                                           row, col + 1, inventoryMask, sm_byNorth, sm_byNorthCount);
                inventoryMask[physId / 64] |= (1ULL << (physId % 64));

                if (!eastOk) continue; 
            }

            board[boardIdx] = p;
            boardIdxHistory[boardIdx] = idx;
            inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));
            pieceStack[step] = idx + 1;
            foundPiece = true;
            step++;

            if (step > maxStepReached) maxStepReached = step; 

            int piecesNow = 0;
            for (int i = 0; i < 256; i++) { if (board[i] != -1) piecesNow++; }

            if (piecesNow > bestPiecesPlaced) {
                bestPiecesPlaced = piecesNow;
                for (int i = 0; i < 256; i++) bestLocalBoard[i] = board[i];
            }
            break;
        }

        if (!foundPiece) {
            pieceStack[step] = 0;
            step--;

            while (step >= startingStep) {
                int undoIdx = d_buildOrder[step];
                if (lockCenterFlag == 1 && (undoIdx == 135 || undoIdx == 221 || undoIdx == 45 || undoIdx == 210 || undoIdx == 34)) {
                    step--;
                } else {
                    break;
                }
            }

            if (step >= startingStep) {
                int undoBoardIdx = d_buildOrder[step];
                board[undoBoardIdx] = -1;

                int oldIdx = boardIdxHistory[undoBoardIdx];
                int physId = c_physicalMapping[oldIdx];
                inventoryMask[physId / 64] |= (1ULL << (physId % 64));
            }
        }
    }

    if (step == 256) {
        if (atomicExch(d_solvedFlag, 1) == 0) {
            for (int i = 0; i < 256; i++) d_solution[i] = board[i];
        }
    }

    int globalMaxRaw = *d_gpuHighScore;
    int globalMax = globalMaxRaw & 0x0FFFFFFF; 

    while (bestPiecesPlaced > globalMax) {
        int expected = globalMax;
        int lockedVal = bestPiecesPlaced | 0x40000000; 
        int oldVal = atomicCAS(d_gpuHighScore, expected, lockedVal);

        if (oldVal == expected) {
            for (int i = 0; i < 256; i++) {
                d_bestBoardOut[i] = bestLocalBoard[i];
            }
            __threadfence();
            atomicExch(d_gpuHighScore, bestPiecesPlaced);
            break;
        } else {
            globalMaxRaw = oldVal;
            globalMax = globalMaxRaw & 0x0FFFFFFF;
        }
    }
    atomicAdd(d_totalSteps, stepCounter);
    d_threadDepths[tid] = bestPiecesPlaced;
}

extern "C" __global__ void solveRepairMode(
    const int* d_partialBoards,
    int numBoards,
    int* d_solution,
    int* d_solvedFlag,
    int* d_gpuHighScore,
    int* d_bestBoardOut,
    unsigned long long* d_totalSteps,
    unsigned long long stepBudget
) {
    __shared__ short sm_byNorth[NUM_COLORS * MAX_PER_COLOR];
    __shared__ short sm_byNorthCount[NUM_COLORS];

    if (threadIdx.x == 0) {
        for (int c = 0; c < NUM_COLORS; c++) sm_byNorthCount[c] = 0;
        for (int i = 0; i < 1024; i++) {
            int p = c_allOrientations[i];
            int nc = getNorth(p);
            if (nc < NUM_COLORS) {
                int cnt = sm_byNorthCount[nc];
                if (cnt < MAX_PER_COLOR) {
                    sm_byNorth[nc * MAX_PER_COLOR + cnt] = (short)i;
                    sm_byNorthCount[nc] = cnt + 1;
                }
            }
        }
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numBoards) return;

    int board[256];
    int boardIdxHistory[256];
    int holes[256];
    int pieceStack[256];
    unsigned long long inventoryMask[4] = { ~0ULL, ~0ULL, ~0ULL, ~0ULL };

    int offset = tid * 256;
    int numHoles = 0;
    int basePieces = 0;

    for (int i = 0; i < 256; i++) {
        board[i] = d_partialBoards[offset + i];
        boardIdxHistory[i] = -1;
        if (board[i] == -2) {
            holes[numHoles] = i;
            pieceStack[numHoles] = 0;
            numHoles++;
            board[i] = -1;
        } else if (board[i] != -1) {
            basePieces++;
            for (int o = 0; o < 1024; o++) {
                if (c_allOrientations[o] == board[i]) {
                    int physId = c_physicalMapping[o];
                    boardIdxHistory[i] = o;
                    inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));
                    break;
                }
            }
        }
    }

    if (numHoles == 0) return;

    int holeStep = 0;
    unsigned long long stepCounter = 0;
    int bestSoFar = (*d_gpuHighScore) & 0x0FFFFFFF;

    while (holeStep >= 0 && stepCounter < stepBudget) {
        if (*d_solvedFlag == 1) break;

        int currentTotal = basePieces + holeStep;
        if (currentTotal > bestSoFar) {
            int globalMaxRaw = *d_gpuHighScore;
            int globalMax = globalMaxRaw & 0x0FFFFFFF;

            while (currentTotal > globalMax) {
                int expected = globalMax;
                int lockedVal = currentTotal | 0x40000000;
                int oldVal = atomicCAS(d_gpuHighScore, expected, lockedVal);

                if (oldVal == expected) {
                    for (int i = 0; i < 256; i++) {
                        d_bestBoardOut[i] = board[i];
                    }
                    __threadfence();
                    atomicExch(d_gpuHighScore, currentTotal);
                    bestSoFar = currentTotal;
                    break;
                }
                globalMaxRaw = oldVal;
                globalMax = globalMaxRaw & 0x0FFFFFFF;
            }
        }

        if (holeStep == numHoles) {
            if (currentTotal == 256) {
                if (atomicExch(d_solvedFlag, 1) == 0) {
                    for (int i = 0; i < 256; i++) d_solution[i] = board[i];
                }
                break;
            }
            holeStep--;
            if (holeStep >= 0) {
                int undoBoardIdx = holes[holeStep];
                board[undoBoardIdx] = -1;
                
                int oldIdx = boardIdxHistory[undoBoardIdx];
                int physId = c_physicalMapping[oldIdx];
                inventoryMask[physId / 64] |= (1ULL << (physId % 64));
            }
            continue;
        }

        stepCounter++;
        int boardIdx = holes[holeStep];
        int row = boardIdx / 16;
        int col = boardIdx % 16;

        int n_req = (row == 0)  ? 0 : (board[boardIdx - 16] != -1 ? getSouth(board[boardIdx - 16]) : WILDCARD);
        int s_req = (row == 15) ? 0 : (board[boardIdx + 16] != -1 ? getNorth(board[boardIdx + 16]) : WILDCARD);
        int w_req = (col == 0)  ? 0 : (board[boardIdx - 1]  != -1 ? getEast (board[boardIdx - 1])  : WILDCARD);
        int e_req = (col == 15) ? 0 : (board[boardIdx + 1]  != -1 ? getWest (board[boardIdx + 1])  : WILDCARD);

        bool foundPiece = false;
        int startIdx = pieceStack[holeStep];

        for (int idx = startIdx; idx < 1024; idx++) {
            int physId = c_physicalMapping[idx];
            if (!(inventoryMask[physId / 64] & (1ULL << (physId % 64)))) continue;

            int p = c_allOrientations[idx];
            if (!matches(p, n_req, e_req, s_req, w_req, row, col)) continue;

            if (row < 15 && board[boardIdx + 16] == -1) {
                int sn = getSouth(p);
                int sw = (col > 0 && board[boardIdx + 15] != -1) ? getEast(board[boardIdx + 15]) : WILDCARD;
                int se = (col == 15) ? 0 : WILDCARD;
                int ss = (row == 14) ? 0 : WILDCARD;
                inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));
                bool ok = hasCandidate(sn, se, ss, sw, row+1, col, inventoryMask, sm_byNorth, sm_byNorthCount);
                inventoryMask[physId / 64] |= (1ULL << (physId % 64));
                if (!ok) continue;
            }

            if (col < 15 && board[boardIdx + 1] == -1) {
                int ew = getEast(p);
                int en = (row > 0 && board[boardIdx - 15] != -1) ? getSouth(board[boardIdx - 15]) : WILDCARD;
                int es = (row == 15) ? 0 : WILDCARD;
                int ee = (col == 14) ? 0 : WILDCARD;
                inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));
                bool ok = hasCandidate(en, ee, es, ew, row, col+1, inventoryMask, sm_byNorth, sm_byNorthCount);
                inventoryMask[physId / 64] |= (1ULL << (physId % 64));
                if (!ok) continue;
            }

            board[boardIdx] = p;
            boardIdxHistory[boardIdx] = idx;
            inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));
            pieceStack[holeStep] = idx + 1;
            foundPiece = true;
            holeStep++;
            break;
        }

        if (!foundPiece) {
            pieceStack[holeStep] = 0;
            holeStep--;
            if (holeStep >= 0) {
                int undoBoardIdx = holes[holeStep];
                board[undoBoardIdx] = -1;
                
                int oldIdx = boardIdxHistory[undoBoardIdx];
                int physId = c_physicalMapping[oldIdx];
                inventoryMask[physId / 64] |= (1ULL << (physId % 64));
            }
        }
    }
    atomicAdd(d_totalSteps, stepCounter);
}
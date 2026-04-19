/**
 * SolveEternityKernel.cu
 * Upgraded for SPIRAL-BUILD and TYPEWRITER (Radar Leash) methods!
 */

// --- DEVICE HELPER FUNCTIONS ---
__device__ inline int getNorth(int p) { return (p >> 24) & 0xFF; }
__device__ inline int getEast(int p)  { return (p >> 16) & 0xFF; }
__device__ inline int getSouth(int p) { return (p >> 8)  & 0xFF; }
__device__ inline int getWest(int p)  { return (p)       & 0xFF; }

__device__ inline bool matches(int p, int n_req, int e_req, int s_req, int w_req, int row, int col) {
    int n = getNorth(p);
    int e = getEast(p);
    int s = getSouth(p);
    int w = getWest(p);

    if (n_req != 255 && n != n_req) return false;
    if (e_req != 255 && e != e_req) return false;
    if (s_req != 255 && s != s_req) return false;
    if (w_req != 255 && w != w_req) return false;

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
    int startingStep,
    int* d_buildOrder,
    int* d_allOrientations,
    int* d_physicalMapping,
    int* d_solution,
    int* d_solvedFlag,
    int* d_gpuHighScore,
    int* d_bestBoardOut,
    unsigned long long* d_totalSteps,
    int lockCenterFlag,
    int* d_threadDepths,
    int* p_radarLimit
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPartialBoards) return;

    // Read the radar limit into a local variable
    int radarLimit = p_radarLimit[0];

    int board[256];
    int pieceStack[256];
    unsigned long long inventoryMask[4] = { ~0ULL, ~0ULL, ~0ULL, ~0ULL };

    int offset = tid * 256;
    for (int i = 0; i < 256; i++) {
        board[i] = d_partialBoards[offset + i];
        pieceStack[i] = 0;

        if (board[i] != -1) {
            for(int o = 0; o < 1024; o++) {
                if(d_allOrientations[o] == board[i]) {
                    int physId = d_physicalMapping[o];
                    inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));
                    break;
                }
            }
        }
    }

    int step = startingStep;
    int maxStepReached = startingStep;
    int bestLocalBoard[256];
    unsigned long long stepCounter = 0;

    while (step >= startingStep && step < 256) {

        stepCounter++;
        if (stepCounter > 5000000) break;
        if (*d_solvedFlag == 1) break;

        // ==========================================================
        // [NEW] THE RADAR LEASH LOGIC
        // If the leash is active (>0) and we reached the maximum
        // allowed look-ahead distance, we force a backtrack.
        // ==========================================================
        bool forceBacktrack = false;
        if (radarLimit > 0 && step >= startingStep + radarLimit) {
            forceBacktrack = true;
        }

        bool foundPiece = false;

        // Only search for a piece if we haven't hit the radar limit
        if (!forceBacktrack) {
            int boardIdx = d_buildOrder[step];

            // ONLY skip the center piece if the rule is locked!
            if (lockCenterFlag == 1 && boardIdx == 135) {
                step++;
                continue;
            }

            int row = boardIdx / 16;
            int col = boardIdx % 16;

            int n_req = (row == 0) ? 0 : (board[boardIdx - 16] != -1 ? getSouth(board[boardIdx - 16]) : 255);
            int s_req = (row == 15) ? 0 : (board[boardIdx + 16] != -1 ? getNorth(board[boardIdx + 16]) : 255);
            int w_req = (col == 0) ? 0 : (board[boardIdx - 1] != -1 ? getEast(board[boardIdx - 1]) : 255);
            int e_req = (col == 15) ? 0 : (board[boardIdx + 1] != -1 ? getWest(board[boardIdx + 1]) : 255);

            int startIdx = pieceStack[step];

            for (int idx = startIdx; idx < 1024; idx++) {
                int physId = d_physicalMapping[idx];

                if (inventoryMask[physId / 64] & (1ULL << (physId % 64))) {
                    int p = d_allOrientations[idx];

                    if (matches(p, n_req, e_req, s_req, w_req, row, col)) {

                        board[boardIdx] = p;
                        inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));

                        pieceStack[step] = idx + 1;
                        foundPiece = true;
                        step++;

                        if (step > maxStepReached) {
                            maxStepReached = step;
                            // Save the physical board layout for Java retrieval
                            for (int i = 0; i < 256; i++) bestLocalBoard[i] = board[i];
                        }

                        break;
                    }
                }
            }
        } // End of (!forceBacktrack) execution

        // Standard backtrack logic (Triggers on dead ends AND when radar limit is hit)
        if (!foundPiece) {
            pieceStack[step] = 0;
            step--;

            // ONLY skip backwards over the center piece if the rule is locked!
            if (lockCenterFlag == 1 && step >= 0 && d_buildOrder[step] == 135) step--;

            if (step >= startingStep) {
                int pToUndo = board[d_buildOrder[step]];
                board[d_buildOrder[step]] = -1;

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

    if (step == 256) {
        if (atomicExch(d_solvedFlag, 1) == 0) {
            for (int i = 0; i < 256; i++) d_solution[i] = board[i];
        }
    }

    int currentMax = *d_gpuHighScore;
    while (maxStepReached > currentMax) {
        if (atomicCAS(d_gpuHighScore, currentMax, maxStepReached) == currentMax) {
            for (int i = 0; i < 256; i++) d_bestBoardOut[i] = bestLocalBoard[i];
            break;
        }
        currentMax = *d_gpuHighScore;
    }

    atomicAdd(d_totalSteps, stepCounter);
    // Every thread saves how deep its specific seed survived
    d_threadDepths[tid] = maxStepReached;
}
// ============================================================================
// --- REPAIR MODE KERNEL (LARGE NEIGHBORHOOD SEARCH) ---
// This kernel doesn't build linearly. It finds the "holes" (-1) in a mostly
// finished board and attempts to fill them with the remaining inventory.
// ============================================================================
extern "C" __global__ void solveRepairMode(
    const int* d_partialBoards,
    int numBoards,
    const int* d_allOrientations,
    const int* d_physicalMapping,
    int* d_solution,
    int* d_solvedFlag,
    int* d_gpuHighScore,
    int* d_bestBoardOut,
    unsigned long long* d_totalSteps,
    int maxStepsPerThread
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numBoards) return;

    int board[256];
    int holes[64];         // Max 64 holes supported (usually we only punch 15-20)
    int pieceStack[64];    // Stack to remember where we were in the inventory loop
    unsigned long long inventoryMask[4] = { ~0ULL, ~0ULL, ~0ULL, ~0ULL };

    int offset = tid * 256;
    int numHoles = 0;
    int basePiecesPlaced = 0;

    // 1. SCAN THE BOARD & INITIALIZE INVENTORY
    for (int i = 0; i < 256; i++) {
        board[i] = d_partialBoards[offset + i];

        if (board[i] == -1) {
            // Found a hole! Add it to our local build order
            holes[numHoles] = i;
            pieceStack[numHoles] = 0;
            numHoles++;
        } else {
            // Existing piece: Remove its physical ID from the available inventory
            basePiecesPlaced++;
            for(int o = 0; o < 1024; o++) {
                if(d_allOrientations[o] == board[i]) {
                    int physId = d_physicalMapping[o];
                    inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));
                    break;
                }
            }
        }
    }

    // If there are no holes, exit early
    if (numHoles == 0) return;

    int holeStep = 0;
    int maxHolesFilled = 0;
    unsigned long long stepCounter = 0;

    // 2. THE REPAIR LOOP (Backtracking only within the holes)
    while (holeStep >= 0 && holeStep < numHoles) {

        stepCounter++;
        // Limit the search time so the GPU doesn't hang on an impossible repair
        if (stepCounter > maxStepsPerThread) break;
        if (*d_solvedFlag == 1) break;

        int boardIdx = holes[holeStep];
        int row = boardIdx / 16;
        int col = boardIdx % 16;

        // Check constraints from ALL 4 sides (since walls might already exist)
        int n_req = (row == 0)  ? 0 : (board[boardIdx - 16] != -1 ? getSouth(board[boardIdx - 16]) : 255);
        int s_req = (row == 15) ? 0 : (board[boardIdx + 16] != -1 ? getNorth(board[boardIdx + 16]) : 255);
        int w_req = (col == 0)  ? 0 : (board[boardIdx - 1]  != -1 ? getEast(board[boardIdx - 1])  : 255);
        int e_req = (col == 15) ? 0 : (board[boardIdx + 1]  != -1 ? getWest(board[boardIdx + 1])  : 255);

        bool foundPiece = false;
        int startIdx = pieceStack[holeStep];

        for (int idx = startIdx; idx < 1024; idx++) {
            int physId = d_physicalMapping[idx];

            if (inventoryMask[physId / 64] & (1ULL << (physId % 64))) {
                int p = d_allOrientations[idx];

                if (matches(p, n_req, e_req, s_req, w_req, row, col)) {

                    // Place the piece
                    board[boardIdx] = p;
                    inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));

                    pieceStack[holeStep] = idx + 1;
                    foundPiece = true;
                    holeStep++;

                    // Update local max depth
                    if (holeStep > maxHolesFilled) {
                        maxHolesFilled = holeStep;
                    }
                    break;
                }
            }
        }

        // Backtrack if no piece fits this specific hole
        if (!foundPiece) {
            pieceStack[holeStep] = 0;
            holeStep--;

            if (holeStep >= 0) {
                int pToUndo = board[holes[holeStep]];
                board[holes[holeStep]] = -1; // Empty the hole again

                // Restore piece to inventory
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

    // 3. CHECK FOR HIGH SCORES & VICTORIES
    int currentTotalPieces = basePiecesPlaced + maxHolesFilled;

    if (currentTotalPieces == 256) {
        if (atomicExch(d_solvedFlag, 1) == 0) {
            for (int i = 0; i < 256; i++) d_solution[i] = board[i];
        }
    }

    int currentMax = *d_gpuHighScore;
    while (currentTotalPieces > currentMax) {
        if (atomicCAS(d_gpuHighScore, currentMax, currentTotalPieces) == currentMax) {
            // Save the newly repaired board back to Java
            for (int i = 0; i < 256; i++) d_bestBoardOut[i] = board[i];
            break;
        }
        currentMax = *d_gpuHighScore;
    }

    atomicAdd(d_totalSteps, stepCounter);
}
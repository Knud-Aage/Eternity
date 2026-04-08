/**
 * SolveEternityKernel.cu
 * Opgraderet til SPIRAL-BUILD metoden!
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
    int* d_buildOrder, // <--- DET NYE SPIRALKORT
    int* d_allOrientations,
    int* d_physicalMapping,
    int* d_solution,
    int* d_solvedFlag,
    int* d_gpuHighScore,
    int* d_bestBoardOut,
    unsigned long long* d_totalSteps
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPartialBoards) return;

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

        int boardIdx = d_buildOrder[step];

        // Hvis vi lander på centerbrikken i spiralen, gå til næste skridt
        if (boardIdx == 135) {
            step++;
            continue;
        }

        int row = boardIdx / 16;
        int col = boardIdx % 16;

        int n_req = (row == 0) ? 0 : (board[boardIdx - 16] != -1 ? getSouth(board[boardIdx - 16]) : 255);
        int s_req = (row == 15) ? 0 : (board[boardIdx + 16] != -1 ? getNorth(board[boardIdx + 16]) : 255);
        int w_req = (col == 0) ? 0 : (board[boardIdx - 1] != -1 ? getEast(board[boardIdx - 1]) : 255);
        int e_req = (col == 15) ? 0 : (board[boardIdx + 1] != -1 ? getWest(board[boardIdx + 1]) : 255);

        bool foundPiece = false;
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
                        // Vi gemmer det fysiske bræt, så vi nemt kan overlevere det
                        for (int i = 0; i < 256; i++) bestLocalBoard[i] = board[i];
                    }

                    break;
                }
            }
        }

        if (!foundPiece) {
            pieceStack[step] = 0;
            step--;

            if (step >= 0 && d_buildOrder[step] == 135) step--;

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
}
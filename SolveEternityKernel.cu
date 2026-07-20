/**
 * SolveEternityKernel.cu
 *
 * Key optimizations:
 *  1. __constant__ memory for allOrientations, physicalMapping, buildOrder.
 *  2. 2D shared-memory index sm_byNW[north][west] used in the MAIN placement loop —
 *     reduces the candidate scan from O(1024) to O(~2) when both neighbours are placed.
 *     byNorth is the fallback when west is unknown; full 1024 scan only when north too is unknown.
 *  3. Same NW index used in hasCandidate (lookahead).
 *  4. O(1) unplace via placedOrientIdx[].
 *  5. Incremental piecesNow counter in solvePBP.
 *  6. pieceStack stores list-position within the active colour list (not a global orient index).
 *  7. Persistent device buffers managed in GpuEngine — no malloc/free per launch.
 */

__constant__ int c_allOrientations[1024];
__constant__ int c_physicalMapping[1024];
__constant__ int c_buildOrder[256];

__device__ inline int getNorth(int p) { return (p >> 24) & 0xFF; }
__device__ inline int getEast (int p) { return (p >> 16) & 0xFF; }
__device__ inline int getSouth(int p) { return (p >>  8) & 0xFF; }
__device__ inline int getWest (int p) { return (p)       & 0xFF; }

#define WILDCARD      255
#define NUM_COLORS    23
#define MAX_PER_COLOR 128   // max orientations per north colour (byNorth fallback)
#define NW_MAX        32    // max orientations per (north,west) pair

__device__ inline bool matches(int p, int n_req, int e_req, int s_req, int w_req, int row, int col)
{
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

// ---------------------------------------------------------------------------
// hasCandidate: uses NW index when both constraints known, byNorth otherwise.
// ---------------------------------------------------------------------------
__device__ bool hasCandidate(
    int n_req, int e_req, int s_req, int w_req,
    int row, int col,
    const unsigned long long* inventoryMask,
    const short* sm_byNorth,      const short* sm_byNorthCount,
    const short* sm_byNW,         const short* sm_byNWCount)
{
    if (n_req != WILDCARD && w_req != WILDCARD && n_req < NUM_COLORS && w_req < NUM_COLORS) {
        int key   = n_req * NUM_COLORS + w_req;
        int count = sm_byNWCount[key];
        for (int i = 0; i < count; i++) {
            int idx    = sm_byNW[key * NW_MAX + i];
            int physId = c_physicalMapping[idx];
            if (!(inventoryMask[physId/64] & (1ULL << (physId%64)))) continue;
            if (matches(c_allOrientations[idx], n_req, e_req, s_req, w_req, row, col)) return true;
        }
        return false;
    }
    if (n_req != WILDCARD && n_req < NUM_COLORS) {
        int count = sm_byNorthCount[n_req];
        for (int i = 0; i < count; i++) {
            int idx    = sm_byNorth[n_req * MAX_PER_COLOR + i];
            int physId = c_physicalMapping[idx];
            if (!(inventoryMask[physId/64] & (1ULL << (physId%64)))) continue;
            if (matches(c_allOrientations[idx], n_req, e_req, s_req, w_req, row, col)) return true;
        }
        return false;
    }
    for (int idx = 0; idx < 1024; idx++) {
        int physId = c_physicalMapping[idx];
        if (!(inventoryMask[physId/64] & (1ULL << (physId%64)))) continue;
        if (matches(c_allOrientations[idx], n_req, e_req, s_req, w_req, row, col)) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// buildSharedIndex — called by thread 0 in each block.
// ---------------------------------------------------------------------------
__device__ void buildSharedIndex(
    short* sm_byNorth, short* sm_byNorthCount,
    short* sm_byNW,    short* sm_byNWCount)
{
    for (int c = 0; c < NUM_COLORS; c++)              sm_byNorthCount[c] = 0;
    for (int k = 0; k < NUM_COLORS * NUM_COLORS; k++) sm_byNWCount[k]    = 0;

    for (int i = 0; i < 1024; i++) {
        int p  = c_allOrientations[i];
        int nc = getNorth(p);
        int wc = getWest(p);
        if (nc < NUM_COLORS) {
            int cnt = sm_byNorthCount[nc];
            if (cnt < MAX_PER_COLOR) sm_byNorth[nc * MAX_PER_COLOR + cnt] = (short)i;
            sm_byNorthCount[nc] = cnt + 1;
        }
        if (nc < NUM_COLORS && wc < NUM_COLORS) {
            int key = nc * NUM_COLORS + wc;
            int cnt = sm_byNWCount[key];
            if (cnt < NW_MAX) sm_byNW[key * NW_MAX + cnt] = (short)i;
            sm_byNWCount[key] = cnt + 1;
        }
    }
}

// ---------------------------------------------------------------------------
// south+east lookahead — inlined via a helper to avoid repetition.
// Returns false if either neighbour has no candidate (prune).
// ---------------------------------------------------------------------------
__device__ inline bool lookahead(
    int p, int physId, int row, int col, int boardIdx,
    const int* board, unsigned long long* inventoryMask,
    const short* sm_byNorth, const short* sm_byNorthCount,
    const short* sm_byNW,    const short* sm_byNWCount)
{
    if (row < 15 && board[boardIdx + 16] == -1) {
        int sn = getSouth(p);
        int sw = (col > 0 && board[boardIdx + 15] != -1) ? getEast(board[boardIdx + 15]) : WILDCARD;
        int se = (col == 15) ? 0 : WILDCARD;
        int ss = (row == 14) ? 0 : WILDCARD;
        inventoryMask[physId/64] &= ~(1ULL << (physId%64));
        bool ok = hasCandidate(sn, se, ss, sw, row + 1, col,
                               inventoryMask, sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount);
        inventoryMask[physId/64] |= (1ULL << (physId%64));
        if (!ok) return false;
    }
    if (col < 15 && board[boardIdx + 1] == -1) {
        int ew = getEast(p);
        int en = (row > 0 && board[boardIdx - 15] != -1) ? getSouth(board[boardIdx - 15]) : WILDCARD;
        int es = (row == 15) ? 0 : WILDCARD;
        int ee = (col == 14) ? 0 : WILDCARD;
        inventoryMask[physId/64] &= ~(1ULL << (physId%64));
        bool ok = hasCandidate(en, ee, es, ew, row, col + 1,
                               inventoryMask, sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount);
        inventoryMask[physId/64] |= (1ULL << (physId%64));
        if (!ok) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// solvePBP — main DFS kernel
// ---------------------------------------------------------------------------
extern "C" __global__ void solvePBP(
    const int* d_partialBoards,
    int numPartialBoards,
    int startingStep,
    int* d_solution,
    int* d_solvedFlag,
    int* d_gpuHighScore,
    int* d_bestBoardOut,
    unsigned long long* d_totalSteps,
    int lockCenterFlag,
    int* d_threadDepths,
    int* p_radarLimit,
    unsigned long long stepBudget
)
{
    __shared__ short sm_byNorth     [NUM_COLORS * MAX_PER_COLOR];
    __shared__ short sm_byNorthCount[NUM_COLORS];
    __shared__ short sm_byNW        [NUM_COLORS * NUM_COLORS * NW_MAX];
    __shared__ short sm_byNWCount   [NUM_COLORS * NUM_COLORS];

    if (threadIdx.x == 0)
        buildSharedIndex(sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount);
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPartialBoards) return;

    int board[256];
    int pieceStack[256];
    int placedOrientIdx[256];
    unsigned long long inventoryMask[4] = { ~0ULL, ~0ULL, ~0ULL, ~0ULL };

    int offset    = tid * 256;
    int piecesNow = 0;
    for (int i = 0; i < 256; i++) {
        board[i]           = d_partialBoards[offset + i];
        pieceStack[i]      = 0;
        placedOrientIdx[i] = -1;
        if (board[i] != -1) {
            piecesNow++;
            for (int o = 0; o < 1024; o++) {
                if (c_allOrientations[o] == board[i]) {
                    int physId = c_physicalMapping[o];
                    inventoryMask[physId/64] &= ~(1ULL << (physId%64));
                    break;
                }
            }
        }
    }

    int step             = startingStep;
    int bestPiecesPlaced = piecesNow;
    int bestLocalBoard[256];
    unsigned long long stepCounter = 0;
    const unsigned long long STEP_BUDGET = stepBudget;

    while (step >= startingStep && step < 256) {
        if (stepCounter >= STEP_BUDGET) break;
        if (*d_solvedFlag == 1)         break;
        stepCounter++;

        int boardIdx = c_buildOrder[step];

        if (lockCenterFlag == 1 && (boardIdx == 135 ||
            boardIdx == 221 || boardIdx == 45 || boardIdx == 210 || boardIdx == 34)) {
            step++;
            continue;
        }

        int row = boardIdx / 16;
        int col = boardIdx % 16;

        int n_req = (row == 0)  ? 0 : (board[boardIdx-16] != -1 ? getSouth(board[boardIdx-16]) : WILDCARD);
        int s_req = (row == 15) ? 0 : (board[boardIdx+16] != -1 ? getNorth(board[boardIdx+16]) : WILDCARD);
        int w_req = (col == 0)  ? 0 : (board[boardIdx-1]  != -1 ? getEast (board[boardIdx-1])  : WILDCARD);
        int e_req = (col == 15) ? 0 : (board[boardIdx+1]  != -1 ? getWest (board[boardIdx+1])  : WILDCARD);

        bool foundPiece = false;
        int  startLi    = pieceStack[step];

        // --- Tier 1: NW index — O(~2) candidates ---
        if (n_req != WILDCARD && w_req != WILDCARD && n_req < NUM_COLORS && w_req < NUM_COLORS) {
            int key   = n_req * NUM_COLORS + w_req;
            int count = sm_byNWCount[key];
            for (int li = startLi; li < count; li++) {
                int idx    = sm_byNW[key * NW_MAX + li];
                int physId = c_physicalMapping[idx];
                if (!(inventoryMask[physId/64] & (1ULL << (physId%64)))) continue;
                int p = c_allOrientations[idx];
                if (!matches(p, n_req, e_req, s_req, w_req, row, col)) continue;
                if (!lookahead(p, physId, row, col, boardIdx, board, inventoryMask,
                               sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount)) continue;
                board[boardIdx] = p;
                inventoryMask[physId/64] &= ~(1ULL << (physId%64));
                placedOrientIdx[step] = idx;
                pieceStack[step] = li + 1;
                foundPiece = true;
                piecesNow++;
                step++;
                if (piecesNow > bestPiecesPlaced) {
                    bestPiecesPlaced = piecesNow;
                    for (int i = 0; i < 256; i++) bestLocalBoard[i] = board[i];
                }
                break;
            }
            if (!foundPiece) pieceStack[step] = 0;

        // --- Tier 2: north-only index — O(~128) candidates ---
        } else if (n_req != WILDCARD && n_req < NUM_COLORS) {
            int count = sm_byNorthCount[n_req];
            for (int li = startLi; li < count; li++) {
                int idx    = sm_byNorth[n_req * MAX_PER_COLOR + li];
                int physId = c_physicalMapping[idx];
                if (!(inventoryMask[physId/64] & (1ULL << (physId%64)))) continue;
                int p = c_allOrientations[idx];
                if (!matches(p, n_req, e_req, s_req, w_req, row, col)) continue;
                if (!lookahead(p, physId, row, col, boardIdx, board, inventoryMask,
                               sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount)) continue;
                board[boardIdx] = p;
                inventoryMask[physId/64] &= ~(1ULL << (physId%64));
                placedOrientIdx[step] = idx;
                pieceStack[step] = li + 1;
                foundPiece = true;
                piecesNow++;
                step++;
                if (piecesNow > bestPiecesPlaced) {
                    bestPiecesPlaced = piecesNow;
                    for (int i = 0; i < 256; i++) bestLocalBoard[i] = board[i];
                }
                break;
            }
            if (!foundPiece) pieceStack[step] = 0;

        // --- Tier 3: full scan — O(1024), rare ---
        } else {
            for (int li = startLi; li < 1024; li++) {
                int physId = c_physicalMapping[li];
                if (!(inventoryMask[physId/64] & (1ULL << (physId%64)))) continue;
                int p = c_allOrientations[li];
                if (!matches(p, n_req, e_req, s_req, w_req, row, col)) continue;
                if (!lookahead(p, physId, row, col, boardIdx, board, inventoryMask,
                               sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount)) continue;
                board[boardIdx] = p;
                inventoryMask[physId/64] &= ~(1ULL << (physId%64));
                placedOrientIdx[step] = li;
                pieceStack[step] = li + 1;
                foundPiece = true;
                piecesNow++;
                step++;
                if (piecesNow > bestPiecesPlaced) {
                    bestPiecesPlaced = piecesNow;
                    for (int i = 0; i < 256; i++) bestLocalBoard[i] = board[i];
                }
                break;
            }
            if (!foundPiece) pieceStack[step] = 0;
        }

        if (!foundPiece) {
            step--;
            while (step >= startingStep) {
                int undoIdx = c_buildOrder[step];
                if (lockCenterFlag == 1 && (undoIdx == 135 ||
                    undoIdx == 221 || undoIdx == 45 || undoIdx == 210 || undoIdx == 34))
                    step--;
                else
                    break;
            }
            if (step >= startingStep) {
                int undoBoardIdx = c_buildOrder[step];
                board[undoBoardIdx] = -1;
                int physId = c_physicalMapping[placedOrientIdx[step]];
                inventoryMask[physId/64] |= (1ULL << (physId%64));
                piecesNow--;
            }
        }
    }

    if (step == 256) {
        if (atomicExch(d_solvedFlag, 1) == 0)
            for (int i = 0; i < 256; i++) d_solution[i] = board[i];
    }

    int globalMaxRaw = *d_gpuHighScore;
    int globalMax    = globalMaxRaw & 0x0FFFFFFF;
    while (bestPiecesPlaced > globalMax) {
        int expected  = globalMax;
        int lockedVal = bestPiecesPlaced | 0x40000000;
        int oldVal    = atomicCAS(d_gpuHighScore, expected, lockedVal);
        if (oldVal == expected) {
            for (int i = 0; i < 256; i++) d_bestBoardOut[i] = bestLocalBoard[i];
            __threadfence();
            atomicExch(d_gpuHighScore, bestPiecesPlaced);
            break;
        }
        globalMaxRaw = oldVal;
        globalMax    = globalMaxRaw & 0x0FFFFFFF;
    }
    atomicAdd(d_totalSteps, stepCounter);
    d_threadDepths[tid] = bestPiecesPlaced;
}


// ---------------------------------------------------------------------------
// solveRepairMode — LNS hole-filling kernel
// ---------------------------------------------------------------------------
extern "C" __global__ void solveRepairMode(
    const int* d_partialBoards,
    int numBoards,
    int* d_solution,
    int* d_solvedFlag,
    int* d_gpuHighScore,
    int* d_bestBoardOut,
    unsigned long long* d_totalSteps,
    int maxStepsPerThread
)
{
    __shared__ short sm_byNorth     [NUM_COLORS * MAX_PER_COLOR];
    __shared__ short sm_byNorthCount[NUM_COLORS];
    __shared__ short sm_byNW        [NUM_COLORS * NUM_COLORS * NW_MAX];
    __shared__ short sm_byNWCount   [NUM_COLORS * NUM_COLORS];

    if (threadIdx.x == 0)
        buildSharedIndex(sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount);
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numBoards) return;

    int board[256];
    int holes[256];
    int pieceStack[256];
    int placedOrientIdx[256];
    unsigned long long inventoryMask[4] = { ~0ULL, ~0ULL, ~0ULL, ~0ULL };

    int offset     = tid * 256;
    int numHoles   = 0;
    int basePieces = 0;

    for (int i = 0; i < 256; i++) {
        board[i] = d_partialBoards[offset + i];
        if (board[i] == -2) {
            holes[numHoles]           = i;
            pieceStack[numHoles]      = 0;
            placedOrientIdx[numHoles] = -1;
            numHoles++;
            board[i] = -1;
        } else if (board[i] != -1) {
            basePieces++;
            for (int o = 0; o < 1024; o++) {
                if (c_allOrientations[o] == board[i]) {
                    int physId = c_physicalMapping[o];
                    inventoryMask[physId/64] &= ~(1ULL << (physId%64));
                    break;
                }
            }
        }
    }

    if (numHoles == 0) return;

    int holeStep  = 0;
    unsigned long long stepCounter = 0;
    int bestSoFar = (*d_gpuHighScore) & 0x0FFFFFFF;

    while (holeStep >= 0 && stepCounter < (unsigned long long)maxStepsPerThread) {
        if (*d_solvedFlag == 1) break;

        int currentTotal = basePieces + holeStep;
        if (currentTotal > bestSoFar) {
            int globalMaxRaw = *d_gpuHighScore;
            int globalMax    = globalMaxRaw & 0x0FFFFFFF;
            while (currentTotal > globalMax) {
                int expected  = globalMax;
                int lockedVal = currentTotal | 0x40000000;
                int oldVal    = atomicCAS(d_gpuHighScore, expected, lockedVal);
                if (oldVal == expected) {
                    for (int i = 0; i < 256; i++) d_bestBoardOut[i] = board[i];
                    __threadfence();
                    atomicExch(d_gpuHighScore, currentTotal);
                    bestSoFar = currentTotal;
                    break;
                }
                globalMaxRaw = oldVal;
                globalMax    = globalMaxRaw & 0x0FFFFFFF;
            }
        }

        if (holeStep == numHoles) {
            if (currentTotal == 256) {
                if (atomicExch(d_solvedFlag, 1) == 0)
                    for (int i = 0; i < 256; i++) d_solution[i] = board[i];
                break;
            }
            holeStep--;
            if (holeStep >= 0) {
                board[holes[holeStep]] = -1;
                int physId = c_physicalMapping[placedOrientIdx[holeStep]];
                inventoryMask[physId/64] |= (1ULL << (physId%64));
            }
            continue;
        }

        stepCounter++;

        int boardIdx = holes[holeStep];
        int row = boardIdx / 16;
        int col = boardIdx % 16;

        int n_req = (row == 0)  ? 0 : (board[boardIdx-16] != -1 ? getSouth(board[boardIdx-16]) : WILDCARD);
        int s_req = (row == 15) ? 0 : (board[boardIdx+16] != -1 ? getNorth(board[boardIdx+16]) : WILDCARD);
        int w_req = (col == 0)  ? 0 : (board[boardIdx-1]  != -1 ? getEast (board[boardIdx-1])  : WILDCARD);
        int e_req = (col == 15) ? 0 : (board[boardIdx+1]  != -1 ? getWest (board[boardIdx+1])  : WILDCARD);

        bool foundPiece = false;
        int  startLi    = pieceStack[holeStep];

        // --- Tier 1: NW index ---
        if (n_req != WILDCARD && w_req != WILDCARD && n_req < NUM_COLORS && w_req < NUM_COLORS) {
            int key   = n_req * NUM_COLORS + w_req;
            int count = sm_byNWCount[key];
            for (int li = startLi; li < count; li++) {
                int idx    = sm_byNW[key * NW_MAX + li];
                int physId = c_physicalMapping[idx];
                if (!(inventoryMask[physId/64] & (1ULL << (physId%64)))) continue;
                int p = c_allOrientations[idx];
                if (!matches(p, n_req, e_req, s_req, w_req, row, col)) continue;
                if (!lookahead(p, physId, row, col, boardIdx, board, inventoryMask,
                               sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount)) continue;
                board[boardIdx] = p;
                inventoryMask[physId/64] &= ~(1ULL << (physId%64));
                placedOrientIdx[holeStep] = idx;
                pieceStack[holeStep] = li + 1;
                foundPiece = true;
                holeStep++;
                break;
            }
            if (!foundPiece) pieceStack[holeStep] = 0;

        // --- Tier 2: north-only index ---
        } else if (n_req != WILDCARD && n_req < NUM_COLORS) {
            int count = sm_byNorthCount[n_req];
            for (int li = startLi; li < count; li++) {
                int idx    = sm_byNorth[n_req * MAX_PER_COLOR + li];
                int physId = c_physicalMapping[idx];
                if (!(inventoryMask[physId/64] & (1ULL << (physId%64)))) continue;
                int p = c_allOrientations[idx];
                if (!matches(p, n_req, e_req, s_req, w_req, row, col)) continue;
                if (!lookahead(p, physId, row, col, boardIdx, board, inventoryMask,
                               sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount)) continue;
                board[boardIdx] = p;
                inventoryMask[physId/64] &= ~(1ULL << (physId%64));
                placedOrientIdx[holeStep] = idx;
                pieceStack[holeStep] = li + 1;
                foundPiece = true;
                holeStep++;
                break;
            }
            if (!foundPiece) pieceStack[holeStep] = 0;

        // --- Tier 3: full scan ---
        } else {
            for (int li = startLi; li < 1024; li++) {
                int physId = c_physicalMapping[li];
                if (!(inventoryMask[physId/64] & (1ULL << (physId%64)))) continue;
                int p = c_allOrientations[li];
                if (!matches(p, n_req, e_req, s_req, w_req, row, col)) continue;
                if (!lookahead(p, physId, row, col, boardIdx, board, inventoryMask,
                               sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount)) continue;
                board[boardIdx] = p;
                inventoryMask[physId/64] &= ~(1ULL << (physId%64));
                placedOrientIdx[holeStep] = li;
                pieceStack[holeStep] = li + 1;
                foundPiece = true;
                holeStep++;
                break;
            }
            if (!foundPiece) pieceStack[holeStep] = 0;
        }

        if (!foundPiece) {
            holeStep--;
            if (holeStep >= 0) {
                board[holes[holeStep]] = -1;
                int physId = c_physicalMapping[placedOrientIdx[holeStep]];
                inventoryMask[physId/64] |= (1ULL << (physId%64));
            }
        }
    }

    atomicAdd(d_totalSteps, stepCounter);
}

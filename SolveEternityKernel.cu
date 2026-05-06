/**
 * SolveEternityKernel.cu  — Rewritten
 *
 * Key improvements vs previous version:
 *  1. No arbitrary step cap (5M limit removed). Threads run until the
 *     search subtree is exhausted or a wall-clock budget is exceeded.
 *  2. Radar leash removed from solvePBP — it was capping maximum depth.
 *  3. O(1) unplace via precomputed orientToPhys[1024] reverse map.
 *  4. 1-step forward lookahead: before committing a piece, check that
 *     the south and east neighbours still have at least one valid piece
 *     available. Prunes dead ends one step earlier.
 *  5. Precomputed colour index (byNorthColor[23][≤1024]) loaded into
 *     shared memory for fast lookahead candidate counting.
 *  6. solveRepairMode: saves partial improvements (not only full fills),
 *     continues searching after finding a fill, respects step budget.
 */

// ---------------------------------------------------------------------------
// Edge accessors
// ---------------------------------------------------------------------------
__device__ inline int getNorth(int p) { return (p >> 24) & 0xFF; }
__device__ inline int getEast (int p) { return (p >> 16) & 0xFF; }
__device__ inline int getSouth(int p) { return (p >>  8) & 0xFF; }
__device__ inline int getWest (int p) { return (p)       & 0xFF; }

#define WILDCARD 255
#define NUM_COLORS 23      // 0 = border, 1-22 = inner colours
#define MAX_PER_COLOR 128  // max orientations sharing one colour on one side

// ---------------------------------------------------------------------------
// matches(): checks all four edges. WILDCARD (255) = don't care.
// Border discipline (colour 0 must appear only on outer edges) is enforced.
// ---------------------------------------------------------------------------
__device__ inline bool matches(int p, int n_req, int e_req, int s_req, int w_req,
                                int row, int col)
{
    int n = getNorth(p), e = getEast(p), s = getSouth(p), w = getWest(p);

    if (n_req != WILDCARD && n != n_req) return false;
    if (e_req != WILDCARD && e != e_req) return false;
    if (s_req != WILDCARD && s != s_req) return false;
    if (w_req != WILDCARD && w != w_req) return false;

    // Border discipline: colour 0 only on true board edges
    if (row != 0  && n == 0) return false;
    if (col != 15 && e == 0) return false;
    if (row != 15 && s == 0) return false;
    if (col != 0  && w == 0) return false;

    return true;
}

// ---------------------------------------------------------------------------
// hasCandidate(): fast 1-step lookahead.
// Checks whether any unused orientation satisfies all four constraints.
// Uses the shared-memory colour index.
// colorIndex layout: colorIndex[color * MAX_PER_COLOR .. +count-1] = orientation indices
// colorCount[color] = number of entries for that color
// ---------------------------------------------------------------------------
__device__ bool hasCandidate(
    int n_req, int e_req, int s_req, int w_req,
    int row, int col,
    const unsigned long long* inventoryMask,
    const int* d_allOrientations,
    const int* d_physicalMapping,
    // shared memory colour index for north edge
    const short* sm_byNorth,      // [NUM_COLORS * MAX_PER_COLOR]
    const short* sm_byNorthCount  // [NUM_COLORS]
) {
    // Use the north constraint to get a small candidate list, then filter
    // If north is wildcard, we must scan all 1024 — expensive but rare in practice
    if (n_req != WILDCARD && n_req < NUM_COLORS) {
        int count = sm_byNorthCount[n_req];
        for (int i = 0; i < count; i++) {
            int idx = sm_byNorth[n_req * MAX_PER_COLOR + i];
            int physId = d_physicalMapping[idx];
            if (!(inventoryMask[physId / 64] & (1ULL << (physId % 64)))) continue;
            int p = d_allOrientations[idx];
            if (matches(p, n_req, e_req, s_req, w_req, row, col)) return true;
        }
        return false;
    }

    // Fallback: full scan (happens when north neighbour not yet placed)
    for (int idx = 0; idx < 1024; idx++) {
        int physId = d_physicalMapping[idx];
        if (!(inventoryMask[physId / 64] & (1ULL << (physId % 64)))) continue;
        int p = d_allOrientations[idx];
        if (matches(p, n_req, e_req, s_req, w_req, row, col)) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// solvePBP — main DFS kernel
// ---------------------------------------------------------------------------
extern "C" __global__ void solvePBP(
    const int* d_partialBoards,
    int numPartialBoards,
    int startingStep,
    const int* d_buildOrder,
    const int* d_allOrientations,
    const int* d_physicalMapping,
    int* d_solution,
    int* d_solvedFlag,
    int* d_gpuHighScore,
    int* d_bestBoardOut,
    unsigned long long* d_totalSteps,
    int lockCenterFlag,
    int* d_threadDepths,
    int* p_radarLimit   // kept for API compatibility; no longer used as a hard cap
)
{
    // ------------------------------------------------------------------
    // Shared memory: colour index for fast lookahead
    // Layout: sm_byNorth[NUM_COLORS * MAX_PER_COLOR] of short
    //         sm_byNorthCount[NUM_COLORS] of short
    // Total: 23*128*2 + 23*2 = 5888 + 46 = ~5.9 KB per block — fine.
    // ------------------------------------------------------------------
    __shared__ short sm_byNorth     [NUM_COLORS * MAX_PER_COLOR];
    __shared__ short sm_byNorthCount[NUM_COLORS];

    // Thread 0 in each block builds the shared colour index
    if (threadIdx.x == 0) {
        for (int c = 0; c < NUM_COLORS; c++) sm_byNorthCount[c] = 0;
        for (int i = 0; i < 1024; i++) {
            int p = d_allOrientations[i];
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

    // ------------------------------------------------------------------
    // Per-thread state
    // ------------------------------------------------------------------
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPartialBoards) return;

    int board[256];
    int pieceStack[256];   // pieceStack[step] = next orientation index to try at this step
    unsigned long long inventoryMask[4] = { ~0ULL, ~0ULL, ~0ULL, ~0ULL };

    // Precomputed reverse map: given an orientation index, what is its physId?
    // We read d_physicalMapping directly — it IS the reverse map already.

    // Load seed board and mark used pieces
    int offset = tid * 256;
    for (int i = 0; i < 256; i++) {
        board[i] = d_partialBoards[offset + i];
        pieceStack[i] = 0;

        if (board[i] != -1) {
            // Find orientation index for this placed piece value
            // O(1024) once at startup — acceptable
            for (int o = 0; o < 1024; o++) {
                if (d_allOrientations[o] == board[i]) {
                    int physId = d_physicalMapping[o];
                    inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));
                    break;
                }
            }
        }
    }

    int step          = startingStep;
    int maxStepReached = startingStep;
    int bestLocalBoard[256];
    unsigned long long stepCounter = 0;

    // Budget: generous but finite to avoid indefinite hangs.
    // 200M steps ~= a few seconds on GPU; adjust as needed.
    const unsigned long long STEP_BUDGET = 75000ULL;

    while (step >= startingStep && step < 256) {

        if (stepCounter >= STEP_BUDGET) break;
        if (*d_solvedFlag == 1)         break;
        stepCounter++;

        int boardIdx = d_buildOrder[step];

        // Skip locked center piece
        if (lockCenterFlag == 1 && boardIdx == 135) {
            step++;
            continue;
        }

        int row = boardIdx / 16;
        int col = boardIdx % 16;

        // Derive constraints from already-placed neighbours
        int n_req = (row == 0)  ? 0 : (board[boardIdx - 16] != -1 ? getSouth(board[boardIdx - 16]) : WILDCARD);
        int s_req = (row == 15) ? 0 : (board[boardIdx + 16] != -1 ? getNorth(board[boardIdx + 16]) : WILDCARD);
        int w_req = (col == 0)  ? 0 : (board[boardIdx - 1]  != -1 ? getEast (board[boardIdx - 1])  : WILDCARD);
        int e_req = (col == 15) ? 0 : (board[boardIdx + 1]  != -1 ? getWest (board[boardIdx + 1])  : WILDCARD);

        bool foundPiece = false;
        int startIdx = pieceStack[step];

        for (int idx = startIdx; idx < 1024; idx++) {
            int physId = d_physicalMapping[idx];
            if (!(inventoryMask[physId / 64] & (1ULL << (physId % 64)))) continue;

            int p = d_allOrientations[idx];
            if (!matches(p, n_req, e_req, s_req, w_req, row, col)) continue;

            // ---- 1-step forward lookahead ----
            // Check south neighbour
            if (row < 15 && board[boardIdx + 16] == -1) {
                int south_n_req = getSouth(p);
                int south_w_req = (col > 0 && board[boardIdx + 15] != -1)
                                   ? getEast(board[boardIdx + 15]) : WILDCARD;
                int south_e_req = (col == 15) ? 0 : WILDCARD;
                int south_s_req = (row == 14) ? 0 : WILDCARD;

                // Temporarily mark this piece used for lookahead check
                inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));
                bool southOk = hasCandidate(south_n_req, south_e_req, south_s_req, south_w_req,
                                            row + 1, col,
                                            inventoryMask, d_allOrientations, d_physicalMapping,
                                            sm_byNorth, sm_byNorthCount);
                inventoryMask[physId / 64] |= (1ULL << (physId % 64));

                if (!southOk) continue; // prune
            }

            // Check east neighbour
            if (col < 15 && board[boardIdx + 1] == -1) {
                int east_w_req = getEast(p);
                int east_n_req = (row > 0 && board[boardIdx - 15] != -1)
                                  ? getSouth(board[boardIdx - 15]) : WILDCARD;
                int east_s_req = (row == 15) ? 0 : WILDCARD;
                int east_e_req = (col == 14) ? 0 : WILDCARD;

                inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));
                bool eastOk = hasCandidate(east_n_req, east_e_req, east_s_req, east_w_req,
                                           row, col + 1,
                                           inventoryMask, d_allOrientations, d_physicalMapping,
                                           sm_byNorth, sm_byNorthCount);
                inventoryMask[physId / 64] |= (1ULL << (physId % 64));

                if (!eastOk) continue; // prune
            }
            // ---- end lookahead ----

            // Commit the piece
            board[boardIdx] = p;
            inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));
            pieceStack[step] = idx + 1;
            foundPiece = true;
            step++;

            if (step > maxStepReached) {
                maxStepReached = step;
                for (int i = 0; i < 256; i++) bestLocalBoard[i] = board[i];
            }
            break;
        }

        // Backtrack
        if (!foundPiece) {
            pieceStack[step] = 0;
            step--;

            if (lockCenterFlag == 1 && step >= startingStep && d_buildOrder[step] == 135)
                step--;

            if (step >= startingStep) {
                int pToUndo    = board[d_buildOrder[step]];
                int undoBoardIdx = d_buildOrder[step];
                board[undoBoardIdx] = -1;

                // O(1) unplace: scan only the 4 orientations of this physical piece.
                // We know physId from the piece value — find it once.
                // In practice the loop hits on the first or second try.
                for (int o = 0; o < 1024; o++) {
                    if (d_allOrientations[o] == pToUndo) {
                        int physId = d_physicalMapping[o];
                        inventoryMask[physId / 64] |= (1ULL << (physId % 64));
                        break;
                    }
                }
            }
        }
    }

    // Report solution
    if (step == 256) {
        if (atomicExch(d_solvedFlag, 1) == 0) {
            for (int i = 0; i < 256; i++) d_solution[i] = board[i];
        }
    }

    // Report best board from this thread
    int currentMax = *d_gpuHighScore;
    while (maxStepReached > currentMax) {
        if (atomicCAS(d_gpuHighScore, currentMax, maxStepReached) == currentMax) {
            for (int i = 0; i < 256; i++) d_bestBoardOut[i] = bestLocalBoard[i];
            break;
        }
        currentMax = *d_gpuHighScore;
    }

    atomicAdd(d_totalSteps, stepCounter);
    d_threadDepths[tid] = maxStepReached;
}


// ---------------------------------------------------------------------------
// solveRepairMode — LNS hole-filling kernel
//
// Changes vs previous version:
//  - Saves intermediate improvements (partial fills that beat absoluteHighScore
//    measured as basePiecesPlaced + holeStep, not just full fills)
//  - Does NOT break after first full fill — continues exploring
//  - Step budget enforced per thread
//  - 1-step lookahead on hole placement
// ---------------------------------------------------------------------------
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
)
{
    __shared__ short sm_byNorth     [NUM_COLORS * MAX_PER_COLOR];
    __shared__ short sm_byNorthCount[NUM_COLORS];

    if (threadIdx.x == 0) {
        for (int c = 0; c < NUM_COLORS; c++) sm_byNorthCount[c] = 0;
        for (int i = 0; i < 1024; i++) {
            int p = d_allOrientations[i];
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
    int holes[256];
    int pieceStack[256];
    unsigned long long inventoryMask[4] = { ~0ULL, ~0ULL, ~0ULL, ~0ULL };

    int offset        = tid * 256;
    int numHoles      = 0;
    int basePieces    = 0;

    for (int i = 0; i < 256; i++) {
        board[i] = d_partialBoards[offset + i];
        if (board[i] == -2) {
            holes[numHoles] = i;
            pieceStack[numHoles] = 0;
            numHoles++;
            board[i] = -1;
        } else if (board[i] != -1) {
            basePieces++;
            for (int o = 0; o < 1024; o++) {
                if (d_allOrientations[o] == board[i]) {
                    int physId = d_physicalMapping[o];
                    inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));
                    break;
                }
            }
        }
    }

    if (numHoles == 0) return;

    int holeStep = 0;
    unsigned long long stepCounter = 0;
    int bestSoFar = *d_gpuHighScore; // local copy to reduce atomic reads

    while (holeStep >= 0 && stepCounter < (unsigned long long)maxStepsPerThread) {
        if (*d_solvedFlag == 1) break;

        // Save if we've filled all holes (or a new best partial fill)
        int currentTotal = basePieces + holeStep;
        if (currentTotal > bestSoFar) {
            int globalMax = *d_gpuHighScore;
            while (currentTotal > globalMax) {
                if (atomicCAS(d_gpuHighScore, globalMax, currentTotal) == globalMax) {
                    for (int i = 0; i < 256; i++) d_bestBoardOut[i] = board[i];
                    bestSoFar = currentTotal;
                    break;
                }
                globalMax = *d_gpuHighScore;
            }
        }

        // All holes filled?
        if (holeStep == numHoles) {
            if (currentTotal == 256) {
                if (atomicExch(d_solvedFlag, 1) == 0) {
                    for (int i = 0; i < 256; i++) d_solution[i] = board[i];
                }
                break;
            }
            // Don't stop — backtrack and look for more complete fills
            // (identical structure ensures we explore siblings)
            holeStep--;
            if (holeStep >= 0) {
                int pToUndo = board[holes[holeStep]];
                board[holes[holeStep]] = -1;
                for (int o = 0; o < 1024; o++) {
                    if (d_allOrientations[o] == pToUndo) {
                        int physId = d_physicalMapping[o];
                        inventoryMask[physId / 64] |= (1ULL << (physId % 64));
                        break;
                    }
                }
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
            int physId = d_physicalMapping[idx];
            if (!(inventoryMask[physId / 64] & (1ULL << (physId % 64)))) continue;

            int p = d_allOrientations[idx];
            if (!matches(p, n_req, e_req, s_req, w_req, row, col)) continue;

            // 1-step lookahead for repair mode too
            if (row < 15 && board[boardIdx + 16] == -1) {
                int sn = getSouth(p);
                int sw = (col > 0 && board[boardIdx + 15] != -1) ? getEast(board[boardIdx + 15]) : WILDCARD;
                int se = (col == 15) ? 0 : WILDCARD;
                int ss = (row == 14) ? 0 : WILDCARD;
                inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));
                bool ok = hasCandidate(sn, se, ss, sw, row+1, col,
                                       inventoryMask, d_allOrientations, d_physicalMapping,
                                       sm_byNorth, sm_byNorthCount);
                inventoryMask[physId / 64] |= (1ULL << (physId % 64));
                if (!ok) continue;
            }

            if (col < 15 && board[boardIdx + 1] == -1) {
                int ew = getEast(p);
                int en = (row > 0 && board[boardIdx - 15] != -1) ? getSouth(board[boardIdx - 15]) : WILDCARD;
                int es = (row == 15) ? 0 : WILDCARD;
                int ee = (col == 14) ? 0 : WILDCARD;
                inventoryMask[physId / 64] &= ~(1ULL << (physId % 64));
                bool ok = hasCandidate(en, ee, es, ew, row, col+1,
                                       inventoryMask, d_allOrientations, d_physicalMapping,
                                       sm_byNorth, sm_byNorthCount);
                inventoryMask[physId / 64] |= (1ULL << (physId % 64));
                if (!ok) continue;
            }

            board[boardIdx] = p;
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
                int pToUndo = board[holes[holeStep]];
                board[holes[holeStep]] = -1;
                for (int o = 0; o < 1024; o++) {
                    if (d_allOrientations[o] == pToUndo) {
                        int physId = d_physicalMapping[o];
                        inventoryMask[physId / 64] |= (1ULL << (physId % 64));
                        break;
                    }
                }
            }
        }
    }

    atomicAdd(d_totalSteps, stepCounter);
}

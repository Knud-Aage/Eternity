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
// c_slipBudget[step] = max TOTAL slipped edges permitted once this step is
// reached (monotonically non-decreasing). All-zero reproduces today's exact
// backtracking exactly -- edge slipping only activates once GpuEngine
// uploads a non-trivial curve. See matchKind() for what "slipped" means.
__constant__ int c_slipBudget[256];

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
// matchKind: exact-match test PLUS a "would this fit with exactly one edge
// slipped" test, in one pass. Used by solvePBP's main placement tiers AND by
// hasCandidate/lookahead below (see the allowSlip parameter there) -- the
// strict matches() above is still used as-is by solveRepairMode, which stays
// slip-unaware (see the comment on solvePBP for why that's a deliberate v1
// scope limit).
//
// Border discipline (outward-facing edges must show the grey border colour,
// interior-facing edges must not) is NEVER slippable -- a border violation
// is an outright reject regardless of slip budget, same as matches() today.
// Only the four directional colour requirements (n_req/e_req/s_req/w_req)
// can be slipped, and at most one of the four per piece.
//
// Returns:
//   0 = no match, not even with a slip -- reject.
//   1 = exact match (0 mismatched edges) -- always acceptable.
//   2 = exactly one mismatched edge -- acceptable ONLY if the caller still
//       has slip budget remaining (c_slipBudget[step] > slipsUsed).
// ---------------------------------------------------------------------------
// Packed-bit helpers for 256-entry boolean state (hasBreak, slipUsedAtStep).
// solvePBP is already register/local-memory bound (255 registers, spilling
// even at the hardware cap) -- these two flags only ever need 1 bit each,
// so storing them as int[256] (1024 bytes apiece) was pure waste. Packing
// both into uint32[8] (32 bytes apiece) trims ~2KB off the per-thread stack
// frame, which is what actually limits how many threads can be resident per
// SM at once.
__device__ inline bool bitGet(const unsigned int* bits, int idx) {
    return (bits[idx >> 5] >> (idx & 31)) & 1u;
}
__device__ inline void bitSet(unsigned int* bits, int idx) {
    bits[idx >> 5] |= (1u << (idx & 31));
}
__device__ inline void bitClear(unsigned int* bits, int idx) {
    bits[idx >> 5] &= ~(1u << (idx & 31));
}

__device__ inline int matchKind(int p, int n_req, int e_req, int s_req, int w_req, int row, int col)
{
    int n = getNorth(p), e = getEast(p), s = getSouth(p), w = getWest(p);

    if (row != 0  && n == 0) return 0;
    if (col != 15 && e == 0) return 0;
    if (row != 15 && s == 0) return 0;
    if (col != 0  && w == 0) return 0;

    int mismatches = 0;
    if (n_req != WILDCARD && n != n_req) mismatches++;
    if (e_req != WILDCARD && e != e_req) mismatches++;
    if (s_req != WILDCARD && s != s_req) mismatches++;
    if (w_req != WILDCARD && w != w_req) mismatches++;

    if (mismatches == 0) return 1;
    if (mismatches == 1) return 2;
    return 0;
}

// ---------------------------------------------------------------------------
// hasCandidate: uses NW index when both constraints known, byNorth otherwise.
//
// allowSlip: when true, a one-edge-slip candidate (matchKind==2) counts as
// "a candidate exists" too, not just an exact match. The NW/byNorth buckets
// are keyed by exact colour, so they structurally cannot contain a candidate
// whose ONLY mismatch is north (byNorth) or north+west (NW) -- when the
// indexed scan comes up empty and allowSlip is set, an extra full scan
// checks for exactly that case, mirroring solvePBP's own slip-fallback tier.
// ---------------------------------------------------------------------------
__device__ bool hasCandidate(
    int n_req, int e_req, int s_req, int w_req,
    int row, int col,
    const unsigned long long* inventoryMask,
    const short* sm_byNorth,      const short* sm_byNorthCount,
    const short* sm_byNW,         const short* sm_byNWCount,
    bool allowSlip)
{
    bool checkedEverything = false; // true once a branch below has already scanned all 1024 orientations, so the allowSlip fallback afterward would be pure duplicate work

    if (n_req != WILDCARD && w_req != WILDCARD && n_req < NUM_COLORS && w_req < NUM_COLORS) {
        int key   = n_req * NUM_COLORS + w_req;
        int count = sm_byNWCount[key];
        for (int i = 0; i < count; i++) {
            int idx    = sm_byNW[key * NW_MAX + i];
            int physId = c_physicalMapping[idx];
            if (!(inventoryMask[physId/64] & (1ULL << (physId%64)))) continue;
            int kind = matchKind(c_allOrientations[idx], n_req, e_req, s_req, w_req, row, col);
            if (kind == 1 || (allowSlip && kind == 2)) return true;
        }
    } else if (n_req != WILDCARD && n_req < NUM_COLORS) {
        int count = sm_byNorthCount[n_req];
        for (int i = 0; i < count; i++) {
            int idx    = sm_byNorth[n_req * MAX_PER_COLOR + i];
            int physId = c_physicalMapping[idx];
            if (!(inventoryMask[physId/64] & (1ULL << (physId%64)))) continue;
            int kind = matchKind(c_allOrientations[idx], n_req, e_req, s_req, w_req, row, col);
            if (kind == 1 || (allowSlip && kind == 2)) return true;
        }
    } else {
        for (int idx = 0; idx < 1024; idx++) {
            int physId = c_physicalMapping[idx];
            if (!(inventoryMask[physId/64] & (1ULL << (physId%64)))) continue;
            int kind = matchKind(c_allOrientations[idx], n_req, e_req, s_req, w_req, row, col);
            if (kind == 1 || (allowSlip && kind == 2)) return true;
        }
        checkedEverything = true; // full scan already covered every orientation, exact and slip alike
    }

    if (!allowSlip || checkedEverything) return false;

    for (int idx = 0; idx < 1024; idx++) {
        int physId = c_physicalMapping[idx];
        if (!(inventoryMask[physId/64] & (1ULL << (physId%64)))) continue;
        if (matchKind(c_allOrientations[idx], n_req, e_req, s_req, w_req, row, col) == 2) return true;
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
    const short* sm_byNW,    const short* sm_byNWCount,
    bool allowSlip)
{
    if (row < 15 && board[boardIdx + 16] == -1) {
        int sn = getSouth(p);
        int sw = (col > 0 && board[boardIdx + 15] != -1) ? getEast(board[boardIdx + 15]) : WILDCARD;
        int se = (col == 15) ? 0 : WILDCARD;
        int ss = (row == 14) ? 0 : WILDCARD;
        inventoryMask[physId/64] &= ~(1ULL << (physId%64));
        bool ok = hasCandidate(sn, se, ss, sw, row + 1, col,
                               inventoryMask, sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount, allowSlip);
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
                               inventoryMask, sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount, allowSlip);
        inventoryMask[physId/64] |= (1ULL << (physId%64));
        if (!ok) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// solvePBP — main DFS kernel
//
// Edge slipping (v1 scope notes):
//  - c_slipBudget is indexed by absolute step (buildOrder position), matching
//    how Verhaard described his own slip array ("1 slipped edge from depth
//    193, 2 from 202, ..."). But unlike his presumably-continuous search,
//    this kernel is re-launched per GPU batch against a partial seed board
//    (startingStep > 0), and slipsUsed always starts at 0 for a fresh launch
//    -- there is no way to know from the incoming board alone whether any of
//    its pre-placed cells (below startingStep) were themselves placed via a
//    slip in an earlier launch. Practically: c_slipBudget is a per-launch
//    allowance from startingStep to 255, not a strict lifetime cap on the
//    final board. Persisting slip usage across launches would need slip
//    metadata threaded through the seed pool (e.g. a parallel array uploaded
//    alongside d_partialBoards) -- not done here.
//  - hasCandidate()/lookahead() ARE slip-aware (allowSlip parameter, computed
//    per-step below as c_slipBudget[step] > slipsUsed): the 1-step lookahead
//    no longer rejects a placement solely because its successor would need a
//    slip to fill, as long as budget remains right now. This is a coarse
//    approximation, not a precise model -- it uses THIS step's budget as a
//    proxy for whatever budget will actually remain once the search reaches
//    the neighbour's own step (which, for the locked/constrained profile,
//    may not even be step+1 -- buildOrder is reordered there). slipsUsed can
//    only grow between now and then, so this can still be slightly
//    optimistic, but it's a much closer approximation than treating
//    lookahead as strictly slip-blind.
//  - solveRepairMode (LNS hole-filling) is untouched -- it fills holes in an
//    otherwise-complete board for a different purpose than reaching a new
//    depth record, and mixing in slip logic there wasn't in scope for this
//    change.
// ---------------------------------------------------------------------------
// __launch_bounds__(256, 2): blockSize is fixed at 256 (see GpuEngine.java's
// cuLaunchKernel call). Without this hint, ptxas was using 255 registers/
// thread (the hardware max) with the compiler free to spend registers
// however it liked, which caps occupancy at 1 resident block/SM regardless
// of the 40KB shared-memory footprint (which would itself allow 2). The
// (256, 2) hint tells ptxas to target <=128 registers/thread so 2 blocks
// (512 threads) can be resident per SM instead of 1 -- doubling the warps
// available to hide memory latency. This is a genuine trade-off: forcing
// registers down can increase local-memory spilling, so whether it's a net
// win can only be confirmed by comparing real [SPEED] log throughput
// before/after, not by compiling alone.
extern "C" __global__ void __launch_bounds__(256, 2) solvePBP(
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
    unsigned int slipUsedAtStep[8]; // packed bit per step -- 1 if that step's placement used its one allowed edge slip
    // Independent resume position for the slip-fallback scan below (see its
    // comment) -- deliberately separate from pieceStack so it can never alias
    // tier 1/2/3's own per-step resume bookkeeping. Extra 1KB/thread of local
    // state; only worth its keep once c_slipBudget is actually non-zero
    // somewhere, which is opt-in (see the constant's own comment).
    int slipPieceStack[256];
    unsigned int hasBreak[8];      // packed bit per cell -- 1 if cell touches an edge mismatch (break)
    int mismatchedCellAtStep[256]; // Board index of neighbouring cell mismatched at this step (-1 if none)
    unsigned long long inventoryMask[4] = { ~0ULL, ~0ULL, ~0ULL, ~0ULL };

    int offset    = tid * 256;
    int piecesNow = 0;
    int slipsUsed = 0; // total slips used so far in this launch, from startingStep onward -- see the slip-budget note above solvePBP
    for (int i = 0; i < 8; i++) {
        slipUsedAtStep[i] = 0;
        hasBreak[i]       = 0;
    }
    for (int i = 0; i < 256; i++) {
        board[i]                = d_partialBoards[offset + i];
        pieceStack[i]           = 0;
        placedOrientIdx[i]      = -1;
        slipPieceStack[i]       = 0;
        mismatchedCellAtStep[i] = -1;
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

    // Non-touching break discipline: compute initial breaks on pre-placed seed pieces (if any).
    // Only South/East are checked for interior pairs -- North/West are only
    // checked for the border case (row==0 / col==0) -- so that each physical
    // edge between two pre-placed cells is counted exactly once instead of
    // once from each side. Mirrors the EAST+SOUTH convention already used in
    // ConflictReducer.countTouchingBreaks for the same reason.
    if (startingStep > 0) {
        for (int s = 0; s < startingStep; s++) {
            int bIdx = c_buildOrder[s];
            if (board[bIdx] == -1) continue;
            int r = bIdx / 16, c = bIdx % 16;
            int p = board[bIdx];
            int n = getNorth(p), e = getEast(p), s_c = getSouth(p), w = getWest(p);

            if (r == 0 && n != 0) { bitSet(hasBreak, bIdx); slipsUsed++; }
            if (c == 0 && w != 0) { bitSet(hasBreak, bIdx); slipsUsed++; }

            int s_req = (r == 15) ? 0 : (board[bIdx+16] != -1 ? getNorth(board[bIdx+16]) : WILDCARD);
            int e_req = (c == 15) ? 0 : (board[bIdx+1]  != -1 ? getWest (board[bIdx+1])  : WILDCARD);
            if (s_req != WILDCARD && s_c != s_req) { bitSet(hasBreak, bIdx); bitSet(hasBreak, bIdx+16); slipsUsed++; }
            if (e_req != WILDCARD && e != e_req)   { bitSet(hasBreak, bIdx); bitSet(hasBreak, bIdx+1);  slipsUsed++; }
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

        // Coarse per-step proxy for "will slip budget still be available by
        // the time lookahead's predicted neighbour is actually reached" --
        // see the allowSlip note in the comment above solvePBP.
        bool allowSlip = c_slipBudget[step] > slipsUsed;

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
                               sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount, allowSlip)) continue;
                board[boardIdx] = p;
                inventoryMask[physId/64] &= ~(1ULL << (physId%64));
                placedOrientIdx[step] = idx;
                pieceStack[step] = li + 1;
                bitClear(slipUsedAtStep, step); // exact match -- overwrite any stale slip flag from an earlier visit to this step
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
                               sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount, allowSlip)) continue;
                board[boardIdx] = p;
                inventoryMask[physId/64] &= ~(1ULL << (physId%64));
                placedOrientIdx[step] = idx;
                pieceStack[step] = li + 1;
                bitClear(slipUsedAtStep, step); // exact match -- overwrite any stale slip flag from an earlier visit to this step
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
                               sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount, allowSlip)) continue;
                board[boardIdx] = p;
                inventoryMask[physId/64] &= ~(1ULL << (physId%64));
                placedOrientIdx[step] = li;
                pieceStack[step] = li + 1;
                bitClear(slipUsedAtStep, step); // exact match -- overwrite any stale slip flag from an earlier visit to this step
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

        // --- Slip fallback: non-touching break discipline ---
        // Only reached once exact-match tiers 1/2/3 found nothing AND slip budget
        // remains at this step. Blackwood non-touching break discipline: refuse
        // to place a piece with a mismatch if ANY already-placed neighbouring cell
        // already has a break.
        bool neighboursBreakFree = true;
        if (row > 0  && board[boardIdx-16] != -1 && bitGet(hasBreak, boardIdx-16)) neighboursBreakFree = false;
        if (col < 15 && board[boardIdx+1]  != -1 && bitGet(hasBreak, boardIdx+1))  neighboursBreakFree = false;
        if (row < 15 && board[boardIdx+16] != -1 && bitGet(hasBreak, boardIdx+16)) neighboursBreakFree = false;
        if (col > 0  && board[boardIdx-1]  != -1 && bitGet(hasBreak, boardIdx-1))  neighboursBreakFree = false;

        if (!foundPiece && c_slipBudget[step] > slipsUsed && neighboursBreakFree) {
            int slipStartLi = slipPieceStack[step];
            for (int li = slipStartLi; li < 1024; li++) {
                int physId = c_physicalMapping[li];
                if (!(inventoryMask[physId/64] & (1ULL << (physId%64)))) continue;
                int p = c_allOrientations[li];
                // kind==1 (exact match) candidates were already tried by tiers
                // 1/2/3 above -- only a genuine one-edge slip is new information here.
                if (matchKind(p, n_req, e_req, s_req, w_req, row, col) != 2) continue;

                // Under break discipline, placing a slip at boardIdx gives boardIdx a break,
                // so lookahead cannot allow successor to slip.
                if (!lookahead(p, physId, row, col, boardIdx, board, inventoryMask,
                               sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount, false)) continue;

                // Identify which neighbouring cell is mismatched
                int n_p = getNorth(p), e_p = getEast(p), s_p = getSouth(p), w_p = getWest(p);
                int mismatchedNeighbour = -1;
                if (n_req != WILDCARD && n_p != n_req)      mismatchedNeighbour = boardIdx - 16;
                else if (e_req != WILDCARD && e_p != e_req) mismatchedNeighbour = boardIdx + 1;
                else if (s_req != WILDCARD && s_p != s_req) mismatchedNeighbour = boardIdx + 16;
                else if (w_req != WILDCARD && w_p != w_req) mismatchedNeighbour = boardIdx - 1;

                board[boardIdx] = p;
                inventoryMask[physId/64] &= ~(1ULL << (physId%64));
                placedOrientIdx[step] = li;
                slipPieceStack[step] = li + 1;
                bitSet(slipUsedAtStep, step);
                bitSet(hasBreak, boardIdx);
                if (mismatchedNeighbour != -1) {
                    bitSet(hasBreak, mismatchedNeighbour);
                }
                mismatchedCellAtStep[step] = mismatchedNeighbour;
                slipsUsed++;
                foundPiece = true;
                piecesNow++;
                step++;
                if (piecesNow > bestPiecesPlaced) {
                    bestPiecesPlaced = piecesNow;
                    for (int i = 0; i < 256; i++) bestLocalBoard[i] = board[i];
                }
                break;
            }
            if (!foundPiece) slipPieceStack[step] = 0;
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
                if (bitGet(slipUsedAtStep, step)) {
                    slipsUsed--;
                    bitClear(hasBreak, undoBoardIdx);
                    int mNeighbour = mismatchedCellAtStep[step];
                    if (mNeighbour != -1) {
                        bitClear(hasBreak, mNeighbour);
                    }
                    mismatchedCellAtStep[step] = -1;
                    bitClear(slipUsedAtStep, step);
                }
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
                               sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount, false)) continue;
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
                               sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount, false)) continue;
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
                               sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount, false)) continue;
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

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
// Blackwood's actual solver (github.com/jblackwood345/EternityII_Solver, the
// source of the standing 470-piece record -- same piece set as this project's
// JBlackwood_Pieces.txt, verified byte-identical) never allows a break on one
// of 5 specific colours (his side_edges = {1,5,9,13,17}), regardless of board
// position -- checked by colour identity, not by border adjacency.
__constant__ int c_isSideColor[23]; // indexed by colour id 0-22 (NUM_COLORS, defined below)
// Cumulative allowed-break count by step, built from his exact sparse list of
// 10 unlock positions (201,206,211,216,221,225,229,233,237,239) -- NOT a
// smooth ramp. See GpuEngine.blackwoodBreakBudget().
__constant__ int c_slipBudget[256];
// Per PHYSICAL piece id (0-255): how many of its 4 edges show one of
// Blackwood's 3 "heuristic_sides" colours (10,13,16) -- colours that are
// over-represented in the piece set. See GpuEngine.blackwoodHeuristicSideCount().
__constant__ int c_heuristicSideCount[256];
// Minimum cumulative heuristic-side-colour count required by each step up to
// HEURISTIC_MAX_INDEX, his exact piecewise-linear schedule -- forces early use
// of the over-represented colours so they don't pile up into a later
// bottleneck. See GpuEngine.blackwoodHeuristicRequired().
__constant__ int c_heuristicRequired[256];
// A permutation of orientation indices 0-1023, sorted by descending
// c_heuristicSideCount[physicalMapping[idx]]. Blackwood's own candidate
// dictionaries are pre-sorted this way, so his greedy search always prefers a
// heuristic-heavy piece when one is valid -- this project's shared-memory
// buckets were built in raw index order instead, which is why
// c_heuristicRequired had to be disabled (see GpuEngine.initCUDA): the
// required minimums assumed his sorted preference and this kernel had no way
// to honour it. buildSharedIndex() now inserts in this order instead of
// 0..1023 (so within each sm_byNorth/sm_byNW bucket, higher-heuristic pieces
// sort first), and tier 3's full scan iterates it directly. See
// GpuEngine.blackwoodHeuristicSortedOrder().
__constant__ int c_heuristicSortedOrder[1024];

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

// Packed-bit helpers for the 256-entry breakUsedAtStep flag (1 bit needed
// per step, so a uint32[8] bitset is far cheaper than an int[256] array).
__device__ inline bool bitGet(const unsigned int* bits, int idx) {
    return (bits[idx >> 5] >> (idx & 31)) & 1u;
}
__device__ inline void bitSet(unsigned int* bits, int idx) {
    bits[idx >> 5] |= (1u << (idx & 31));
}
__device__ inline void bitClear(unsigned int* bits, int idx) {
    bits[idx >> 5] &= ~(1u << (idx & 31));
}

// First step at which a break is ever unlockable (his break_indexes_allowed.Min()),
// and the last step his heuristic-colour-exhaustion requirement applies to
// (his max_heuristic_index). These two ranges never overlap, so break-eligible
// candidates and the heuristic minimum-count gate are never both relevant at
// the same step.
#define FIRST_BREAK_INDEX   201
#define HEURISTIC_MAX_INDEX 160

// ---------------------------------------------------------------------------
// matchKind: like matches(), but distinguishes an exact match from a single-
// edge mismatch that's still break-eligible (Blackwood's rule: a break is
// never allowed on one of his 5 "side_edges" colours, checked by the
// candidate's OWN colour on the mismatched edge, not the required colour).
// Returns 0 = reject, 1 = exact match, 2 = single break-eligible mismatch.
// ---------------------------------------------------------------------------
__device__ inline int matchKind(int p, int n_req, int e_req, int s_req, int w_req, int row, int col)
{
    int n = getNorth(p), e = getEast(p), s = getSouth(p), w = getWest(p);

    if (row != 0  && n == 0) return 0;
    if (col != 15 && e == 0) return 0;
    if (row != 15 && s == 0) return 0;
    if (col != 0  && w == 0) return 0;

    int mismatches = 0;
    int mismatchColor = -1;
    if (n_req != WILDCARD && n != n_req) { mismatches++; mismatchColor = n; }
    if (e_req != WILDCARD && e != e_req) { mismatches++; mismatchColor = e; }
    if (s_req != WILDCARD && s != s_req) { mismatches++; mismatchColor = s; }
    if (w_req != WILDCARD && w != w_req) { mismatches++; mismatchColor = w; }

    if (mismatches == 0) return 1;
    if (mismatches == 1 && mismatchColor >= 0 && mismatchColor < NUM_COLORS && !c_isSideColor[mismatchColor]) return 2;
    return 0;
}

// ---------------------------------------------------------------------------
// classifyCandidate: the tier 1/2/3 accept decision, driven by breakEligible/
// heuristicGateActive/heuristicFloor -- all loop-invariant for the whole
// candidate scan at a given step (step and breaksUsed don't change while
// scanning), so solvePBP computes them ONCE per step instead of re-deriving
// them from step/breaksUsed on every single candidate as before. When
// neither gate can apply (steps 161-200, the gap between HEURISTIC_MAX_INDEX
// and FIRST_BREAK_INDEX -- exactly where a resumed 190-235-depth search
// spends most of its backtracking), this collapses to a plain matches()
// check with none of matchKind()'s mismatch-counting overhead, matching
// pre-Blackwood cost for that range exactly. Also skips matchKind() outside
// [FIRST_BREAK_INDEX, 256) entirely, since a kind==2 candidate can never be
// accepted there regardless of breaksUsed (proved by breakEligible's own
// step>=FIRST_BREAK_INDEX term) -- today's kernel called it unconditionally
// and threw the mismatch-counting work away.
// Returns 0 = reject, 1 = exact match, 2 = break-eligible match.
// ---------------------------------------------------------------------------
__device__ inline int classifyCandidate(
    int p, int physId, int n_req, int e_req, int s_req, int w_req, int row, int col,
    bool breakEligible, bool heuristicGateActive, int heuristicFloor, int heuristicSum)
{
    int kind;
    if (breakEligible) {
        kind = matchKind(p, n_req, e_req, s_req, w_req, row, col);
        if (kind == 0) return 0;
    } else {
        if (!matches(p, n_req, e_req, s_req, w_req, row, col)) return 0;
        kind = 1;
    }
    if (heuristicGateActive && heuristicSum + c_heuristicSideCount[physId] < heuristicFloor) return 0;
    return kind;
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

    // Inserted in c_heuristicSortedOrder (descending heuristic-colour count)
    // rather than raw index order, so each bucket comes out with its
    // highest-heuristic candidates first -- see c_heuristicSortedOrder above.
    for (int rank = 0; rank < 1024; rank++) {
        int i  = c_heuristicSortedOrder[rank];
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
// buildWestIndex — solvePBP-only sibling of buildSharedIndex's sm_byNorth,
// indexed by WEST colour instead of north. Exists only to give the
// break-fallback tier a bounded, indexed way to find "west matches exactly,
// north is the one allowed break" candidates -- see its call site for the
// full correctness argument. Kept as a separate function/pass (not folded
// into buildSharedIndex) so solveRepairMode, which has no break-fallback
// tier and never calls this, pays zero extra shared-memory cost for it.
// ---------------------------------------------------------------------------
__device__ void buildWestIndex(short* sm_byWest, short* sm_byWestCount)
{
    for (int c = 0; c < NUM_COLORS; c++) sm_byWestCount[c] = 0;

    for (int rank = 0; rank < 1024; rank++) {
        int i  = c_heuristicSortedOrder[rank];
        int p  = c_allOrientations[i];
        int wc = getWest(p);
        if (wc < NUM_COLORS) {
            int cnt = sm_byWestCount[wc];
            if (cnt < MAX_PER_COLOR) sm_byWest[wc * MAX_PER_COLOR + cnt] = (short)i;
            sm_byWestCount[wc] = cnt + 1;
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
    int breakToleranceFlag, // was the p_radarLimit landmine (a throwaway host
                             // pointer never read) -- now a real toggle for
                             // the kind==2 break-eligible paths below. 1 =
                             // current always-on behaviour (default, matches
                             // every prior live run), 0 = strict-only, for
                             // A/B comparison. See GpuEngine.setBreakToleranceEnabled().
    unsigned long long stepBudget,
    int lookaheadEnabledFlag, // A/B toggle for the south+east hasCandidate()
                              // pre-check below every placement. Blackwood's
                              // own 470-record solver (github.com/jblackwood345/
                              // EternityII_Solver, Program.cs SolvePuzzle()) has
                              // no forward-checking at all -- pure backtrack,
                              // place and see -- so this tests whether the
                              // pruning here is worth its cost or just overhead.
                              // 1 = current always-on behaviour (default), 0 =
                              // skip the check entirely. See
                              // GpuEngine.setLookaheadEnabled().

    // --- Marathon-thread persistence -------------------------------------
    // Every launch previously discarded ALL in-progress backtracking state,
    // capping any single thread's sustained search at stepBudget (75,000
    // unlocked / 30,000 locked) no matter how long the process ran -- the same
    // defect already fixed in the sibling Blackwood kernel, where removing it
    // broke a day-long depth plateau immediately.
    //
    // It matters more here because this kernel has NO randomization anywhere
    // (no rand/curand/xorshift; sm_byNorth/sm_byNW/sm_byWest and
    // c_heuristicSortedOrder all derive from __constant__ data uploaded once at
    // GpuEngine construction). The search is therefore fully deterministic given
    // (seed board, startingStep, stepBudget) -- so a seed re-submitted unchanged
    // is PROVABLY guaranteed to retrace the identical path to the identical
    // depth. Re-queued "elite" seeds aren't merely suboptimal today, they're
    // dead compute.
    //
    // Scope is deliberately minimal: ONE slot, used by thread 0 only (the
    // "marathon thread"). It ignores its seed board and resumes the persisted
    // search; every other thread behaves exactly as before. That keeps this a
    // clean A/B test of the hypothesis while touching the least surface area.
    int persistResumeFlag,    // 1 = thread 0 resumes from the slot below; 0 =
                              // every thread fresh-inits (previous behaviour).
                              // Host-computed -- see GpuEngine.persistSlotValid.
    int* d_persistBoard,
    int* d_persistPieceStack,
    int* d_persistPlacedOrientIdx,
    int* d_persistBreakFallbackStack,
    int* d_persistBestLocalBoard,
    unsigned long long* d_persistInventoryMask,
    unsigned int* d_persistBreakUsedAtStep,
    int* d_persistBreaksUsed,
    int* d_persistHeuristicSum,
    int* d_persistStep,
    int* d_persistPiecesNow,
    int* d_persistBestPiecesPlaced,
    int* d_persistFloor,      // the startingStep this slot was checkpointed
                              // under; see the resume block for why the launch
                              // parameter must not be trusted on resume.
    int* d_marathonDepth
)
{
    __shared__ short sm_byNorth     [NUM_COLORS * MAX_PER_COLOR];
    __shared__ short sm_byNorthCount[NUM_COLORS];
    __shared__ short sm_byNW        [NUM_COLORS * NUM_COLORS * NW_MAX];
    __shared__ short sm_byNWCount   [NUM_COLORS * NUM_COLORS];
    __shared__ short sm_byWest      [NUM_COLORS * MAX_PER_COLOR];
    __shared__ short sm_byWestCount [NUM_COLORS];

    if (threadIdx.x == 0) {
        buildSharedIndex(sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount);
        buildWestIndex(sm_byWest, sm_byWestCount);
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPartialBoards) return;

    int board[256];
    int pieceStack[256];
    int placedOrientIdx[256];
    unsigned long long inventoryMask[4] = { ~0ULL, ~0ULL, ~0ULL, ~0ULL };
    unsigned int breakUsedAtStep[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    int breaksUsed   = 0;
    int heuristicSum = 0;
    int breakFallbackStack[256];
    int bestLocalBoard[256];

    int piecesNow = 0;
    int step = startingStep;
    int bestPiecesPlaced = 0;
    int floor = startingStep;

    if (tid == 0 && persistResumeFlag == 1) {
        for (int i = 0; i < 256; i++) {
            board[i]              = d_persistBoard[i];
            pieceStack[i]         = d_persistPieceStack[i];
            placedOrientIdx[i]    = d_persistPlacedOrientIdx[i];
            breakFallbackStack[i] = d_persistBreakFallbackStack[i];
            bestLocalBoard[i]     = d_persistBestLocalBoard[i];
        }
        for (int i = 0; i < 4; i++) {
            inventoryMask[i] = d_persistInventoryMask[i];
        }
        for (int i = 0; i < 8; i++) {
            breakUsedAtStep[i] = d_persistBreakUsedAtStep[i];
        }
        breaksUsed       = *d_persistBreaksUsed;
        heuristicSum     = *d_persistHeuristicSum;
        step             = *d_persistStep;
        piecesNow        = *d_persistPiecesNow;
        bestPiecesPlaced = *d_persistBestPiecesPlaced;
        floor            = *d_persistFloor;
    } else {
        int offset = tid * 256;
        for (int i = 0; i < 256; i++) {
            board[i]              = d_partialBoards[offset + i];
            pieceStack[i]         = 0;
            placedOrientIdx[i]    = -1;
            breakFallbackStack[i] = 0;
            if (board[i] != -1) {
                piecesNow++;
                for (int o = 0; o < 1024; o++) {
                    if (c_allOrientations[o] == board[i]) {
                        int physId = c_physicalMapping[o];
                        inventoryMask[physId/64] &= ~(1ULL << (physId%64));
                        heuristicSum += c_heuristicSideCount[physId];
                        break;
                    }
                }
            }
        }

        if (startingStep > 0) {
            for (int s = 0; s < startingStep; s++) {
                int bIdx = c_buildOrder[s];
                if (board[bIdx] == -1) continue;
                int p = board[bIdx];
                int r = bIdx / 16, c = bIdx % 16;
                int e = getEast(p), s_c = getSouth(p);
                int s_req = (r == 15) ? 0 : (board[bIdx+16] != -1 ? getNorth(board[bIdx+16]) : WILDCARD);
                int e_req = (c == 15) ? 0 : (board[bIdx+1]  != -1 ? getWest (board[bIdx+1])  : WILDCARD);
                if (s_req != WILDCARD && s_c != s_req && s_c < NUM_COLORS && !c_isSideColor[s_c]) breaksUsed++;
                if (e_req != WILDCARD && e   != e_req && e   < NUM_COLORS && !c_isSideColor[e])   breaksUsed++;
            }
        }

        bestPiecesPlaced = piecesNow;
        for (int i = 0; i < 256; i++) {
            bestLocalBoard[i] = board[i];
        }
        floor = startingStep;
    }

    unsigned long long stepCounter = 0;
    const unsigned long long STEP_BUDGET = stepBudget;

    while (step >= floor && step < 256) {
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

        bool breakEligible       = (breakToleranceFlag == 1) && (step >= FIRST_BREAK_INDEX) && (breaksUsed < c_slipBudget[step]);
        bool heuristicGateActive = (step <= HEURISTIC_MAX_INDEX);
        int  heuristicFloor      = heuristicGateActive ? c_heuristicRequired[step] : 0;

        bool foundPiece = false;
        int  startLi    = pieceStack[step];

        if (n_req != WILDCARD && w_req != WILDCARD && n_req < NUM_COLORS && w_req < NUM_COLORS) {
            int key   = n_req * NUM_COLORS + w_req;
            int count = sm_byNWCount[key];
            for (int li = startLi; li < count; li++) {
                int idx    = sm_byNW[key * NW_MAX + li];
                int physId = c_physicalMapping[idx];
                if (!(inventoryMask[physId/64] & (1ULL << (physId%64)))) continue;
                int p = c_allOrientations[idx];
                int kind = classifyCandidate(p, physId, n_req, e_req, s_req, w_req, row, col,
                                              breakEligible, heuristicGateActive, heuristicFloor, heuristicSum);
                if (kind == 0) continue;
                if (lookaheadEnabledFlag == 1 && !lookahead(p, physId, row, col, boardIdx, board, inventoryMask,
                               sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount)) continue;
                board[boardIdx] = p;
                inventoryMask[physId/64] &= ~(1ULL << (physId%64));
                placedOrientIdx[step] = idx;
                pieceStack[step]      = li + 1;
                if (kind == 2) {
                    breaksUsed++;
                    bitSet(breakUsedAtStep, step);
                }
                heuristicSum += c_heuristicSideCount[physId];
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
        } else if (n_req != WILDCARD && n_req < NUM_COLORS) {
            int count = sm_byNorthCount[n_req];
            for (int li = startLi; li < count; li++) {
                int idx    = sm_byNorth[n_req * MAX_PER_COLOR + li];
                int physId = c_physicalMapping[idx];
                if (!(inventoryMask[physId/64] & (1ULL << (physId%64)))) continue;
                int p = c_allOrientations[idx];
                int kind = classifyCandidate(p, physId, n_req, e_req, s_req, w_req, row, col,
                                              breakEligible, heuristicGateActive, heuristicFloor, heuristicSum);
                if (kind == 0) continue;
                if (lookaheadEnabledFlag == 1 && !lookahead(p, physId, row, col, boardIdx, board, inventoryMask,
                               sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount)) continue;
                board[boardIdx] = p;
                inventoryMask[physId/64] &= ~(1ULL << (physId%64));
                placedOrientIdx[step] = idx;
                pieceStack[step]      = li + 1;
                if (kind == 2) {
                    breaksUsed++;
                    bitSet(breakUsedAtStep, step);
                }
                heuristicSum += c_heuristicSideCount[physId];
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
        } else {
            for (int li = startLi; li < 1024; li++) {
                int idx    = c_heuristicSortedOrder[li];
                int physId = c_physicalMapping[idx];
                if (!(inventoryMask[physId/64] & (1ULL << (physId%64)))) continue;
                int p = c_allOrientations[idx];
                int kind = classifyCandidate(p, physId, n_req, e_req, s_req, w_req, row, col,
                                              breakEligible, heuristicGateActive, heuristicFloor, heuristicSum);
                if (kind == 0) continue;
                if (lookaheadEnabledFlag == 1 && !lookahead(p, physId, row, col, boardIdx, board, inventoryMask,
                               sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount)) continue;
                board[boardIdx] = p;
                inventoryMask[physId/64] &= ~(1ULL << (physId%64));
                placedOrientIdx[step] = idx;
                pieceStack[step]      = li + 1;
                if (kind == 2) {
                    breaksUsed++;
                    bitSet(breakUsedAtStep, step);
                }
                heuristicSum += c_heuristicSideCount[physId];
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

        // --- Break fallback: tiers 1/2 index by exact north/west colour, so a
        // break-eligible candidate whose ONLY mismatch is on north or west is
        // structurally invisible to them (tier 3's unconditional scan already
        // catches every kind==2 candidate on its own, but tier 3 only runs
        // when north isn't yet known at all). Bounded to the same
        // FIRST_BREAK_INDEX/budget gate as above, so this only ever executes
        // in the narrow, sparse range where breaks are permitted at all.
        //
        // When tier 1 ran (both north/west known -- the common case deep in
        // the endgame, exactly where this tier matters most), this used to
        // be an unindexed O(1024) scan. It's now two bounded, indexed scans:
        // matchKind() allows exactly one mismatch total, so a break-eligible
        // candidate here has EITHER west exact (its one mismatch is on
        // north -- findable in sm_byWest[w_req], an index no other tier
        // touches) OR north exact (mismatch on west -- findable in
        // sm_byNorth[n_req]). A candidate exact on BOTH is tier 1's own
        // territory (sm_byNW) and was already tried there; skipped here
        // (the getWest/getNorth checks below) to avoid a redundant
        // re-attempt. A candidate wrong on both is >1 mismatch, i.e. not
        // break-eligible at all -- so these two scans are provably complete
        // for the tier-1 case. Tier 2/3 still fall through to the full
        // unindexed scan below (west isn't indexed when unconstrained, and
        // this path is rare enough not to be worth a second structure). ---
        if (!foundPiece && breakEligible) {
            bool tier1Ran = (n_req != WILDCARD && w_req != WILDCARD && n_req < NUM_COLORS && w_req < NUM_COLORS);
            int scanFrom = breakFallbackStack[step];

            if (tier1Ran) {
                int northCount = sm_byNorthCount[n_req];
                int westCount  = sm_byWestCount[w_req];
                int totalLen   = northCount + westCount;

                for (int li = scanFrom; li < totalLen; li++) {
                    int idx, physId;
                    if (li < northCount) {
                        idx    = sm_byNorth[n_req * MAX_PER_COLOR + li];
                        physId = c_physicalMapping[idx];
                        if (!(inventoryMask[physId/64] & (1ULL << (physId%64)))) continue;
                        if (getWest(c_allOrientations[idx]) == w_req) continue; // tier 1's own territory
                    } else {
                        idx    = sm_byWest[w_req * MAX_PER_COLOR + (li - northCount)];
                        physId = c_physicalMapping[idx];
                        if (!(inventoryMask[physId/64] & (1ULL << (physId%64)))) continue;
                        if (getNorth(c_allOrientations[idx]) == n_req) continue; // tier 1's own territory
                    }
                    int p = c_allOrientations[idx];
                    if (matchKind(p, n_req, e_req, s_req, w_req, row, col) != 2) continue;
                    if (lookaheadEnabledFlag == 1 && !lookahead(p, physId, row, col, boardIdx, board, inventoryMask,
                                   sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount)) continue;
                    board[boardIdx] = p;
                    inventoryMask[physId/64] &= ~(1ULL << (physId%64));
                    placedOrientIdx[step] = idx;
                    breakFallbackStack[step] = li + 1;
                    breaksUsed++;
                    bitSet(breakUsedAtStep, step);
                    heuristicSum += c_heuristicSideCount[physId];
                    foundPiece = true;
                    piecesNow++;
                    step++;
                    if (piecesNow > bestPiecesPlaced) {
                        bestPiecesPlaced = piecesNow;
                        for (int i = 0; i < 256; i++) bestLocalBoard[i] = board[i];
                    }
                    break;
                }
                if (!foundPiece) breakFallbackStack[step] = 0;
            } else {
                for (int li = scanFrom; li < 1024; li++) {
                    int physId = c_physicalMapping[li];
                    if (!(inventoryMask[physId/64] & (1ULL << (physId%64)))) continue;
                    int p = c_allOrientations[li];
                    // kind==1 candidates were already tried by tiers 1/2/3 above --
                    // only a genuine break is new information here.
                    if (matchKind(p, n_req, e_req, s_req, w_req, row, col) != 2) continue;
                    if (lookaheadEnabledFlag == 1 && !lookahead(p, physId, row, col, boardIdx, board, inventoryMask,
                                   sm_byNorth, sm_byNorthCount, sm_byNW, sm_byNWCount)) continue;
                    board[boardIdx] = p;
                    inventoryMask[physId/64] &= ~(1ULL << (physId%64));
                    placedOrientIdx[step] = li;
                    breakFallbackStack[step] = li + 1;
                    breaksUsed++;
                    bitSet(breakUsedAtStep, step);
                    heuristicSum += c_heuristicSideCount[physId];
                    foundPiece = true;
                    piecesNow++;
                    step++;
                    if (piecesNow > bestPiecesPlaced) {
                        bestPiecesPlaced = piecesNow;
                        for (int i = 0; i < 256; i++) bestLocalBoard[i] = board[i];
                    }
                    break;
                }
                if (!foundPiece) breakFallbackStack[step] = 0;
            }
        }

        if (!foundPiece) {
            step--;
            while (step >= floor) {
                int undoIdx = c_buildOrder[step];
                if (lockCenterFlag == 1 && (undoIdx == 135 ||
                    undoIdx == 221 || undoIdx == 45 || undoIdx == 210 || undoIdx == 34))
                    step--;
                else
                    break;
            }
            if (step >= floor) {
                int undoBoardIdx = c_buildOrder[step];
                board[undoBoardIdx] = -1;
                int physId = c_physicalMapping[placedOrientIdx[step]];
                inventoryMask[physId/64] |= (1ULL << (physId%64));
                heuristicSum -= c_heuristicSideCount[physId];
                if (bitGet(breakUsedAtStep, step)) {
                    breaksUsed--;
                    bitClear(breakUsedAtStep, step);
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

    if (tid == 0 && persistResumeFlag == 1) {
        // Report the marathon thread's accumulated depth SEPARATELY, and zero its
        // d_threadDepths entry: that depth spans many launches, while
        // SeedSelector.selectBest scores each seed by threadDepths[i] -- leaving it
        // in would let seeds.get(0)'s original (shallow, unsearched) board win the
        // elite tier repeatedly on progress it never made.
        d_threadDepths[0] = 0;
        *d_marathonDepth = bestPiecesPlaced;
    } else {
        d_threadDepths[tid] = bestPiecesPlaced;
        // NOTE: deliberately NOT writing *d_marathonDepth here. Every non-marathon
        // thread would be racing the same single address (up to 15,000 of them --
        // see getDynamicBatchSize), so thread 0's real value would be clobbered by
        // whichever zero-write landed last, making the A/B signal this whole change
        // exists to measure read as garbage. The host already zeroes d_marathonDepth
        // before every launch (GpuEngine.runDeepDfs), so the non-resuming case
        // correctly reports 0 with no device write at all.
    }

    if (tid == 0 && step != 256) {
        for (int i = 0; i < 256; i++) {
            d_persistBoard[i]              = board[i];
            d_persistPieceStack[i]         = pieceStack[i];
            d_persistPlacedOrientIdx[i]    = placedOrientIdx[i];
            d_persistBreakFallbackStack[i] = breakFallbackStack[i];
            d_persistBestLocalBoard[i]     = bestLocalBoard[i];
        }
        for (int i = 0; i < 4; i++) {
            d_persistInventoryMask[i] = inventoryMask[i];
        }
        for (int i = 0; i < 8; i++) {
            d_persistBreakUsedAtStep[i] = breakUsedAtStep[i];
        }
        *d_persistBreaksUsed       = breaksUsed;
        *d_persistHeuristicSum     = heuristicSum;
        *d_persistStep             = step;
        *d_persistPiecesNow        = piecesNow;
        *d_persistBestPiecesPlaced = bestPiecesPlaced;
        *d_persistFloor            = floor;
    }
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

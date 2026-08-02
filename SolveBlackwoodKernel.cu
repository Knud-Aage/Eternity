/**
 * SolveBlackwoodKernel.cu
 *
 * A genuinely faithful GPU-native port of Joshua Blackwood's own algorithm
 * (github.com/jblackwood345/EternityII_Solver) -- NOT a modification of
 * SolveEternityKernel.cu's solvePBP. That kernel stays untouched; this is a
 * new, separate, additive kernel, built so a real three-way comparison is
 * possible: CPU-only (dk.puzzle.blackwood.BlackwoodSolver), the existing
 * generic-index hybrid (solvePBP), and this.
 *
 * Faithfully mirrors dk.puzzle.blackwood.BlackwoodSolver.solvePuzzle() (the
 * already-verified CPU port) -- see that class and BwUtil.java for the
 * reference algorithm being translated here. Candidate tables are flattened
 * host-side by dk.puzzle.blackwood.BwGpuTables from prepare()'s own
 * already-verified table objects (not re-derived a second time) into a CSR
 * (offset+count index, in __constant__ memory) + flat payload (in global
 * device memory) layout.
 *
 * IMPORTANT: this kernel deliberately has NO runtime 4-direction border/
 * mismatch check (unlike solvePBP's matches()/matchKind()) -- Blackwood's
 * real algorithm doesn't have one either. Border-colour correctness falls
 * out of hard rotation filters already baked into table construction (see
 * BlackwoodSolver.prepare(): bottomSides->rotation 0 only, leftSides->
 * rotation 1 only, topSides->rotation 2 only, rightSides*->rotation 3 only,
 * start->rotation 2 only -- confirmed directly against that code before this
 * kernel was written). Porting matches()/matchKind() here would be
 * UNFAITHFUL, not just redundant -- do not "fix" this back in later.
 *
 * All colours are Blackwood's own raw 0-22 numbering throughout, exactly
 * like the CPU port -- never TheSil numbering, no translation needed here.
 *
 * Candidate packing (single int32, mirrors this project's existing
 * getNorth/getEast-style bit packing): byte0 = pieceNumber-1 (0-255),
 * byte1 = topSide (0-22), byte2 = rightSide (0-22), byte3 = rotation
 * (bits0-1) | breakCount (bit2, 0 or 1 -- never 2, addCandidateIfValid
 * already filters those out) | heuristicSideCount (bits3-5, 0-4). Empty-cell
 * sentinel is -1, matching SolveEternityKernel.cu's own convention.
 */

#define NUM_TABLES 10
#define KEY_SPACE  529
#define TABLE_CORNERS         0
#define TABLE_LEFT_SIDES      1
#define TABLE_TOP_SIDES       2
#define TABLE_RIGHT_NOBREAK   3
#define TABLE_RIGHT_BREAK     4
#define TABLE_MIDDLES_NOBREAK 5
#define TABLE_MIDDLES_BREAK   6
#define TABLE_SOUTH_START     7
#define TABLE_WEST_START      8
#define TABLE_START           9

#define FIRST_BREAK_INDEX   201
#define HEURISTIC_MAX_INDEX 160
#define MAX_BOTTOM_PAYLOAD  96

__constant__ int c_csrOffset[NUM_TABLES * KEY_SPACE];        // 21,160 B
__constant__ int c_csrCount [NUM_TABLES * KEY_SPACE];        // 21,160 B
__constant__ int c_bottomRawOffset[23];                      // 92 B
__constant__ int c_bottomRawCount [23];                      // 92 B
__constant__ int c_bottomRawPayload[MAX_BOTTOM_PAYLOAD];     // 384 B, real count ~56
__constant__ int c_stepToTableId[256];                       // 1,024 B; row-0 steps hold an unused sentinel
__constant__ int c_stepBoardIdx[256];                        // 1,024 B; row*16+col per step
__constant__ int c_breakArray[256];                          // 1,024 B
__constant__ int c_heuristicArray[256];                      // 1,024 B
// Total ~47 KB of this module's own fresh 64 KB __constant__ budget.

__device__ inline int bwPieceNum(int r)       { return (r & 0xFF) + 1; }
__device__ inline int bwTopSide(int r)        { return (r >> 8)  & 0xFF; }
__device__ inline int bwRightSide(int r)      { return (r >> 16) & 0xFF; }
__device__ inline int bwRotation(int r)       { return (r >> 24) & 0x3; }
__device__ inline int bwBreakCount(int r)     { return (r >> 26) & 0x1; }
__device__ inline int bwHeuristicCount(int r) { return (r >> 27) & 0x7; }

// Packed-bit helpers for the 256-entry pieceUsed flag -- copied verbatim from
// SolveEternityKernel.cu (__device__ functions aren't shared across
// separately-compiled modules).
__device__ inline bool bitGet(const unsigned int* bits, int idx) {
    return (bits[idx >> 5] >> (idx & 31)) & 1u;
}
__device__ inline void bitSet(unsigned int* bits, int idx) {
    bits[idx >> 5] |= (1u << (idx & 31));
}
__device__ inline void bitClear(unsigned int* bits, int idx) {
    bits[idx >> 5] &= ~(1u << (idx & 31));
}

// xorshift64* -- lightweight in-kernel PRNG. No curand infrastructure exists
// anywhere in this codebase (dependency present in pom.xml/cp.txt, zero
// actual device-side usage); adding it for one kernel's candidate-order
// jitter isn't worth the new dependency surface. This mirrors the
// DISTRIBUTION SHAPE of BwUtil's java.util.Random-based jitter (uniform
// picks, same score formulas, same re-randomization cadence) -- NOT
// java.util.Random's exact algorithm or draw order. A GPU thread and a CPU
// attempt given "the same seed" will legitimately produce different boards;
// that's expected, not a bug.
__device__ inline unsigned long long xorshift64star(unsigned long long *state) {
    unsigned long long x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

// Uniform int in [0, bound) -- mirrors java.util.Random.nextInt(bound)'s
// CONTRACT (uniform over the range), not its rejection-sampling algorithm.
__device__ inline int randInt(unsigned long long *state, unsigned int bound) {
    if (bound == 0) return 0;
    return (int)(xorshift64star(state) % (unsigned long long)bound);
}

extern "C" __global__ void solveBlackwoodDfs(
    const int* d_payload,             // global memory: flat candidate payload, all 10 tables concatenated
    unsigned long long seedBase,      // host-varied every launch (nanoTime ^ launchCounter)
    unsigned long long stepBudget,    // per-thread node cap THIS launch -- a TDR safety valve, not a checkpoint
    int  numThreads,
    int* d_gpuHighScore,              // atomic high-water maxSolveIndex across all threads, all launches this run
    int* d_bestBoardOut,              // [256] packed records of the current best board
    int* d_solution,                  // [256] set once if any thread reaches step 256 (a genuine full solve)
    int* d_solvedFlag,
    unsigned long long* d_totalNodes, // atomicAdd, for throughput reporting
    int* d_threadDepths               // [numThreads] this thread's maxSolveIndex this attempt
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numThreads) return;

    int board[256];
    int pieceIndexToTryNext[256];
    int cumulativeBreaks[256];
    int cumulativeHeuristicSideCount[256];
    unsigned int pieceUsedBits[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    int bsOffset[23];
    int bsCount[23];
    int bsPayload[MAX_BOTTOM_PAYLOAD];
    int bestLocalBoard[256];

    for (int i = 0; i < 256; i++) {
        board[i] = -1;
        pieceIndexToTryNext[i] = 0;
    }

    unsigned long long rngState = seedBase ^ ((unsigned long long)tid * 0x9E3779B97F4A7C15ULL);
    if (rngState == 0) rngState = 0x9E3779B97F4A7C15ULL; // xorshift64* requires a non-zero state

    // Per-thread, per-attempt bottomSides rebuild: mirrors
    // BwUtil.sortAndFreezeBottomSides's formula exactly, applied to the same
    // raw pool BwGpuTables.build() already extracted -- (heuristicCount>0
    // ? 100 : 0) + jitter, descending, insertion sort. This table is tiny
    // (~56 entries total across 23 buckets, real count measured in
    // BwGpuTablesTest), so an O(n^2) insertion sort per attempt is cheap --
    // faithful re-randomization every attempt, matching Blackwood's own
    // per-attempt (not per-batch) rebuild of this one table.
    {
        int keys[MAX_BOTTOM_PAYLOAD]; // scratch, reused per bucket
        int fillPos = 0;
        for (int left = 0; left < 23; left++) {
            int rawOff = c_bottomRawOffset[left];
            int rawCnt = c_bottomRawCount[left];
            bsOffset[left] = fillPos;
            bsCount[left] = rawCnt;
            for (int i = 0; i < rawCnt; i++) {
                int rec = c_bottomRawPayload[rawOff + i];
                int hc = bwHeuristicCount(rec);
                keys[i] = (hc > 0 ? 100 : 0) + randInt(&rngState, 99);
                bsPayload[fillPos + i] = rec;
            }
            for (int i = 1; i < rawCnt; i++) {
                int keyVal = keys[i];
                int recVal = bsPayload[fillPos + i];
                int j = i - 1;
                while (j >= 0 && keys[j] < keyVal) {
                    keys[j + 1] = keys[j];
                    bsPayload[fillPos + j + 1] = bsPayload[fillPos + j];
                    j--;
                }
                keys[j + 1] = keyVal;
                bsPayload[fillPos + j + 1] = recVal;
            }
            fillPos += rawCnt;
        }
    }

    // Step 0: uniform-random pick from corners at key 0 (left=0,bottom=0) --
    // mirrors BlackwoodSolver.solvePuzzle()'s uniform pick from corners[0]
    // exactly. corners[0] non-empty is a verified invariant (BlackwoodSolver.
    // prepare() throws if not; BwGpuTablesTest asserts the same survives CSR
    // flattening), so cnt>0 here is not runtime-checked.
    {
        int off = c_csrOffset[TABLE_CORNERS * KEY_SPACE + 0];
        int cnt = c_csrCount [TABLE_CORNERS * KEY_SPACE + 0];
        int pick = off + randInt(&rngState, (unsigned int)cnt);
        board[0] = d_payload[pick];
        bitSet(pieceUsedBits, bwPieceNum(board[0]) - 1);
        cumulativeBreaks[0] = 0;
        cumulativeHeuristicSideCount[0] = bwHeuristicCount(board[0]);
    }

    int solveIndex = 1;
    int maxSolveIndex = 1;
    int bestPiecesPlaced = 1;
    unsigned long long nodeCount = 0;
    bool completed = false;

    while (true) {
        nodeCount++;

        if (solveIndex > maxSolveIndex) {
            maxSolveIndex = solveIndex;
            if (maxSolveIndex > bestPiecesPlaced) {
                bestPiecesPlaced = maxSolveIndex;
                for (int i = 0; i < 256; i++) bestLocalBoard[i] = board[i];
            }
            if (maxSolveIndex >= 256) { completed = true; break; }
        }

        if (nodeCount > stepBudget) break;          // this launch's budget hit, NOT genuine exhaustion
        if (solveIndex < 1) break;                  // genuine exhaustion -- BlackwoodSolver.attemptExhausted() guard
        if (*d_solvedFlag == 1) break;               // another thread already found a full solution this launch

        int boardIdx = c_stepBoardIdx[solveIndex];
        int row = boardIdx >> 4;
        int col = boardIdx & 15;

        if (board[boardIdx] != -1) {
            bitClear(pieceUsedBits, bwPieceNum(board[boardIdx]) - 1);
            board[boardIdx] = -1;
        }

        int off, cnt;
        bool useBottom = false;
        if (row == 0) {
            // (row=0,col=0) is exclusively step 0, seeded above -- the solveIndex<1
            // guard prevents the general loop from ever revisiting it, so col>=1
            // here is guaranteed and board[boardIdx-1] is always a valid, already-
            // placed west neighbour.
            int westRight = bwRightSide(board[boardIdx - 1]);
            if (col < 15) {
                useBottom = true;
                off = bsOffset[westRight];
                cnt = bsCount[westRight];
            } else {
                int key = westRight * 23;
                off = c_csrOffset[TABLE_CORNERS * KEY_SPACE + key];
                cnt = c_csrCount [TABLE_CORNERS * KEY_SPACE + key];
            }
        } else {
            int leftSide = (col == 0) ? 0 : bwRightSide(board[boardIdx - 1]);
            int southTop = bwTopSide(board[boardIdx - 16]);
            int key = leftSide * 23 + southTop;
            int tableId = c_stepToTableId[solveIndex];
            off = c_csrOffset[tableId * KEY_SPACE + key];
            cnt = c_csrCount [tableId * KEY_SPACE + key];
        }

        bool foundPiece = false;
        if (cnt > 0) {
            int breaksThisTurn = c_breakArray[solveIndex] - cumulativeBreaks[solveIndex - 1];
            bool heuristicGateActive = (solveIndex <= HEURISTIC_MAX_INDEX);
            int heuristicFloor = heuristicGateActive ? c_heuristicArray[solveIndex] : 0;

            for (int i = pieceIndexToTryNext[solveIndex]; i < cnt; i++) {
                int rec = useBottom ? bsPayload[off + i] : d_payload[off + i];
                if (bwBreakCount(rec) > breaksThisTurn) break; // sort-order invariant: table is break-count-monotonic

                int pieceNum = bwPieceNum(rec);
                if (bitGet(pieceUsedBits, pieceNum - 1)) continue;

                int hc = bwHeuristicCount(rec);
                if (heuristicGateActive && (cumulativeHeuristicSideCount[solveIndex - 1] + hc) < heuristicFloor) {
                    break; // abandons the WHOLE scan for this step, not just this candidate -- matches the CPU port exactly
                }

                board[boardIdx] = rec;
                bitSet(pieceUsedBits, pieceNum - 1);
                cumulativeBreaks[solveIndex] = cumulativeBreaks[solveIndex - 1] + bwBreakCount(rec);
                cumulativeHeuristicSideCount[solveIndex] = cumulativeHeuristicSideCount[solveIndex - 1] + hc;
                pieceIndexToTryNext[solveIndex] = i + 1;
                foundPiece = true;
                solveIndex++;
                break;
            }
        }

        if (!foundPiece) {
            pieceIndexToTryNext[solveIndex] = 0;
            solveIndex--;
        }
    }

    if (completed) {
        if (atomicExch(d_solvedFlag, 1) == 0) {
            for (int i = 0; i < 256; i++) d_solution[i] = board[i];
        }
    }

    // Same lock-bit atomic best-board update pattern as solvePBP (SolveEternityKernel.cu).
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
    atomicAdd(d_totalNodes, nodeCount);
    d_threadDepths[tid] = bestPiecesPlaced;
}

package dk.puzzle.gpu;

import dk.puzzle.model.PieceInventory;
import dk.puzzle.util.PieceUtils;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.JCuda;

import java.util.List;
import static jcuda.driver.JCudaDriver.*;

/**
 * GPU-accelerated engine for the Eternity II solver.
 *
 * <p>Persistent device buffers are allocated once in the constructor and reused
 * across every kernel launch, eliminating per-call cuMemAlloc/cuMemFree overhead.
 * Read-only lookup tables (allOrientations, physicalMapping, buildOrder) are
 * uploaded once to CUDA __constant__ memory for hardware-broadcast O(1) reads.</p>
 */
public class GpuEngine {

    private static final int MAX_BOARDS = 110_000;

    // Per-thread iteration cap for solvePBP. Locked runs pre-commit 5 core
    // cells (see lockCenterFlag in the kernel), which removes some of the
    // contradictions that would otherwise trigger a fast dead-end/backtrack,
    // so locked threads tend to ride the budget out instead of exiting early
    // — that's what was making locked batches take ~13s vs ~3-5s unlocked
    // at the old fixed 75000 cap. Give locked a smaller budget to compensate.
    private static final long STEP_BUDGET_UNLOCKED = 75_000L;
    private static final long STEP_BUDGET_LOCKED   = 30_000L;

    private CUfunction dfsFunction;
    private CUfunction repairFunction;
    private CUmodule   cuModule;

    private final PieceInventory inventory;
    private final boolean lockCenter;
    private final long stepBudget;
    // Gates the kernel's kind==2 break-eligible paths (see solvePBP's
    // breakToleranceFlag param). Default true reproduces every prior live
    // run's behaviour exactly (break-tolerance has been unconditionally on
    // since the Blackwood integration) -- false is for A/B comparison only,
    // e.g. "does allowing breaks at all help or hurt" holding everything
    // else constant. Ported from an unmerged exploration that toggled a
    // cruder, non-colour-excluded break mechanism; this gates the real one.
    private volatile boolean breakToleranceEnabled = true;
    // Gates the kernel's south+east hasCandidate() forward-check before every
    // placement (see solvePBP's lookaheadEnabledFlag param). DEFAULT FALSE
    // (2026-08-01): live A/B evidence beat the theory -- two isolated runs
    // seeded from the identical 235-piece board, equal GPU time, only this
    // flag differing. Lookahead-off found two new all-time records (238 then
    // 240) in under 50 minutes; lookahead-on found nothing. Matches
    // Blackwood's own 470-record solver (github.com/jblackwood345/
    // EternityII_Solver, Program.cs SolvePuzzle()), which has no
    // forward-checking at all -- pure backtrack, place and see. Kept
    // toggleable (not removed) so this can be flipped back on for further
    // comparison without a recompile.
    private volatile boolean lookaheadEnabled = false;
    // Gates marathon-thread persistence (see solvePBP's persistResumeFlag). When
    // on, thread 0 resumes its checkpointed search across launches instead of
    // restarting from its seed; every other thread is unaffected. DEFAULT FALSE
    // so the live pipeline behaves exactly as it did until this is deliberately
    // switched on for an A/B run -- same convention as the two flags above.
    private volatile boolean seedPersistenceEnabled = false;
    // Whether the persistence slot currently holds a resumable checkpoint. Host-
    // side only: the kernel is told the answer via persistResumeFlag rather than
    // reading a device flag, which keeps invalidation a pure Java-side operation
    // (see invalidatePersistedState) with no device write to forget.
    private boolean persistSlotValid = false;

    // Persistent device buffers — allocated once, reused every launch
    private CUdeviceptr d_partialBoards;
    private CUdeviceptr d_solution;
    private CUdeviceptr d_solvedFlag;
    private CUdeviceptr d_gpuHighScore;
    private CUdeviceptr d_bestBoardOut;
    private CUdeviceptr d_totalSteps;
    private CUdeviceptr d_threadDepths;

    // Persistent search state for marathon thread (slot 0)
    private CUdeviceptr d_persistBoard;
    private CUdeviceptr d_persistPieceStack;
    private CUdeviceptr d_persistPlacedOrientIdx;
    private CUdeviceptr d_persistBreakFallbackStack;
    private CUdeviceptr d_persistBestLocalBoard;
    private CUdeviceptr d_persistInventoryMask;
    private CUdeviceptr d_persistBreakUsedAtStep;
    private CUdeviceptr d_persistBreaksUsed;
    private CUdeviceptr d_persistHeuristicSum;
    private CUdeviceptr d_persistStep;
    private CUdeviceptr d_persistPiecesNow;
    private CUdeviceptr d_persistBestPiecesPlaced;
    private CUdeviceptr d_persistFloor;
    private CUdeviceptr d_marathonDepth;

    public GpuEngine(PieceInventory inventory, boolean lockCenter, int[] buildOrder) {
        JCuda.cudaSetDeviceFlags(JCuda.cudaDeviceScheduleBlockingSync);
        this.inventory  = inventory;
        this.lockCenter = lockCenter;
        this.stepBudget = lockCenter ? STEP_BUDGET_LOCKED : STEP_BUDGET_UNLOCKED;
        initCUDA(buildOrder);
    }

    private void initCUDA(int[] buildOrder) {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext cuContext = new CUcontext();
        cuCtxCreate(cuContext, 0, device);

        cuModule = new CUmodule();
        cuModuleLoad(cuModule, "SolveEternityKernel.ptx");

        dfsFunction = new CUfunction();
        cuModuleGetFunction(dfsFunction, cuModule, "solvePBP");

        repairFunction = new CUfunction();
        cuModuleGetFunction(repairFunction, cuModule, "solveRepairMode");

        // Upload read-only data to __constant__ memory — done once, no per-call cost
        uploadConstant("c_allOrientations", inventory.allOrientations, 1024L * Sizeof.INT);
        uploadConstant("c_physicalMapping",  inventory.physicalMapping,  1024L * Sizeof.INT);
        uploadConstant("c_buildOrder",       buildOrder,                  256L * Sizeof.INT);
        uploadConstant("c_isSideColor",       blackwoodSideColorMask(),     23L * Sizeof.INT);
        uploadConstant("c_slipBudget",        blackwoodBreakBudget(),      256L * Sizeof.INT);
        uploadConstant("c_heuristicSideCount", blackwoodHeuristicSideCount(inventory), 256L * Sizeof.INT);
        // ENABLED (2026-07-31): candidates are now walked via
        // c_heuristicSortedOrder (see buildSharedIndex/tier 3 in the kernel),
        // so a heuristic-heavy piece is preferred whenever one is valid --
        // matching how Blackwood's own pre-sorted dictionaries behave. Live-
        // tested in isolation: Peak P2 Depth held at 214-221 (baseline range),
        // so the sort itself is safe on its own.
        uploadConstant("c_heuristicSortedOrder", blackwoodHeuristicSortedOrder(inventory), 1024L * Sizeof.INT);
        // STILL DISABLED (2026-08-02, re-confirmed after fixing two real bugs):
        // briefly re-enabled today after fixing (1) BLACKWOOD_SIDE_COLORS/
        // BLACKWOOD_HEURISTIC_COLORS being Blackwood's raw colour IDs applied
        // unmapped against this project's differently-numbered PieceInventory,
        // and (2) blackwoodHeuristicRequired()'s last branch using uniform
        // float precision instead of his actual float/double split. Live-
        // tested with both fixes in place: Peak depth still didn't clear
        // ~100 -- much better than the original collapse (~25-26) with the
        // color bug still present, but still far below the 211-235 baseline.
        // So the color bug wasn't the whole story: even with the right
        // colours, Blackwood's exact cumulative-minimum schedule is calibrated
        // against HIS OWN candidate pool (9 pre-sorted, pre-filtered,
        // per-cell-role tables), not this kernel's generic uniform tiered
        // index (see buildSharedIndex/sm_byNorth/sm_byNW) -- the two don't
        // necessarily have the same heuristic-colour candidate availability at
        // a given step, so his numbers may simply not transfer here without
        // being re-derived against what this kernel can actually reach.
        // c_heuristicSortedOrder (the PREFERENCE, not a hard requirement)
        // stays enabled -- only the hard cumulative-minimum gate is off again.
        uploadConstant("c_heuristicRequired", new int[256], 256L * Sizeof.INT);

        allocatePersistentBuffers();
    }

    // -----------------------------------------------------------------------
    // Joshua Blackwood's actual solver constants (github.com/jblackwood345/
    // EternityII_Solver, the source of the standing 470-piece record) -- this
    // project's piece set (src/main/resources/JBlackwood_Pieces.txt) matches
    // his Get_Pieces() verbatim (the same 256 physical pieces, same order),
    // so the SEQUENCE-POSITION constants (BLACKWOOD_BREAK_INDEXES) apply with
    // no re-derivation. The COLOUR constants (BLACKWOOD_SIDE_COLORS,
    // BLACKWOOD_HEURISTIC_COLORS) are a different story: his colour IDs are
    // NOT the same numbering as this project's (TheSil) -- see
    // translateToTheSil() and dk.puzzle.blackwood.BwUtil.BLACKWOOD_TO_THESIL.
    // -----------------------------------------------------------------------

    /**
     * His side_edges: a break is never allowed on an edge showing one of these 5 colours, by colour
     * identity -- but by HIS colour identity. These are Blackwood's raw colour IDs, and this
     * project's PieceInventory (and everything the GPU kernel actually operates on) uses TheSil's
     * numbering, which is a DIFFERENT, non-identity bijection (see dk.puzzle.blackwood.BwUtil.
     * BLACKWOOD_TO_THESIL, verified 2026-08-02). Applying these raw IDs directly against
     * TheSil-numbered piece data -- which is what every consumer of blackwoodSideColorMask() and
     * blackwoodHeuristicSideCount() did before this fix -- silently excludes/weights the WRONG
     * colours. Must always be translated through BLACKWOOD_TO_THESIL before use.
     */
    private static final int[] BLACKWOOD_SIDE_COLORS = translateToTheSil(new int[]{1, 5, 9, 13, 17});

    /** His break_indexes_allowed: exactly 10 sequence positions where +1 total break is unlocked. Not a smooth ramp.
     *  Sequence positions, not colours -- no translation needed. */
    private static final int[] BLACKWOOD_BREAK_INDEXES = {201, 206, 211, 216, 221, 225, 229, 233, 237, 239};

    /** His heuristic_sides: 3 colours over-represented in the piece set, requiring early (not late) use.
     *  Same colour-numbering caveat as BLACKWOOD_SIDE_COLORS above -- translated through BLACKWOOD_TO_THESIL. */
    private static final int[] BLACKWOOD_HEURISTIC_COLORS = translateToTheSil(new int[]{13, 16, 10});

    private static int[] translateToTheSil(int[] blackwoodRawColors) {
        int[] translated = new int[blackwoodRawColors.length];
        for (int i = 0; i < blackwoodRawColors.length; i++) {
            translated[i] = dk.puzzle.blackwood.BwUtil.BLACKWOOD_TO_THESIL[blackwoodRawColors[i]];
        }
        return translated;
    }

    private static int[] blackwoodSideColorMask() {
        int[] mask = new int[23];
        for (int c : BLACKWOOD_SIDE_COLORS) mask[c] = 1;
        return mask;
    }

    /** Cumulative allowed-break count by step, matching his Get_Break_Array() exactly. */
    private static int[] blackwoodBreakBudget() {
        int[] budget = new int[256];
        int cumulative = 0;
        for (int i = 0; i < 256; i++) {
            for (int idx : BLACKWOOD_BREAK_INDEXES) {
                if (idx == i) { cumulative++; break; }
            }
            budget[i] = cumulative;
        }
        return budget;
    }

    /** Per physical piece id (0-255): how many of its 4 edges show one of the 3 heuristic colours. */
    private static int[] blackwoodHeuristicSideCount(PieceInventory inventory) {
        int[] counts = new int[256];
        for (int physId = 0; physId < 256; physId++) {
            int p = inventory.allOrientations[physId * 4]; // rotation doesn't affect which colours are present
            int n = PieceUtils.getNorth(p), e = PieceUtils.getEast(p), s = PieceUtils.getSouth(p), w = PieceUtils.getWest(p);
            int count = 0;
            for (int hc : BLACKWOOD_HEURISTIC_COLORS) {
                if (n == hc) count++;
                if (e == hc) count++;
                if (s == hc) count++;
                if (w == hc) count++;
            }
            counts[physId] = count;
        }
        return counts;
    }

    /**
     * A permutation of orientation indices 0-1023, sorted by descending
     * heuristic-side-count of the underlying physical piece -- mirrors
     * Blackwood's own candidate dictionaries, which are pre-sorted the same
     * way so his search always prefers a heuristic-heavy piece when one is
     * valid. The kernel's buildSharedIndex() inserts into sm_byNorth/sm_byNW
     * in this order (so each bucket comes out highest-heuristic-first) and
     * tier 3's full scan walks it directly. Ties keep their original
     * ascending-index order (Integer boxing makes Arrays.sort stable) --
     * arbitrary but deterministic, and Blackwood's own tie-breaking isn't
     * specified in his source, so there's no "correct" order to match.
     */
    private static int[] blackwoodHeuristicSortedOrder(PieceInventory inventory) {
        int[] sideCount = blackwoodHeuristicSideCount(inventory);
        Integer[] order = new Integer[1024];
        for (int i = 0; i < 1024; i++) order[i] = i;
        java.util.Arrays.sort(order, (a, b) ->
                sideCount[inventory.physicalMapping[b]] - sideCount[inventory.physicalMapping[a]]);
        int[] result = new int[1024];
        for (int i = 0; i < 1024; i++) result[i] = order[i];
        return result;
    }

    /**
     * His heuristic_array: minimum cumulative heuristic-side-colour count
     * required by each step up to index 160, a piecewise-linear schedule.
     * Uses float arithmetic to match his C# (float) casts exactly, including
     * truncation at every segment boundary -- EXCEPT the last branch's
     * divisor, which his C# source leaves as an unsuffixed double literal
     * (confirmed by direct read of Program.cs), promoting just that division
     * to double precision while the other four branches stay float. Fixed
     * 2026-08-02 (was `4.4615f` here, a uniform-float approximation that can
     * shift a boundary index by ±1 versus his actual behaviour) to match
     * the independently-verified dk.puzzle.blackwood.BwUtil.getHeuristicArray(),
     * which has the same asymmetry and the same test proving it matters.
     * Steps beyond 160 are left at 0 (his own array leaves them at C#'s int
     * default, and the kernel only ever consults this for step &lt;= HEURISTIC_MAX_INDEX).
     */
    private static int[] blackwoodHeuristicRequired() {
        int[] arr = new int[256];
        for (int i = 0; i <= 160; i++) {
            if (i <= 16) arr[i] = 0;
            else if (i <= 26) arr[i] = (int) (((float) i - 16) * 2.8f);
            else if (i <= 56) arr[i] = (int) ((((float) i - 26) * 1.43333f) + 28);
            else if (i <= 76) arr[i] = (int) ((((float) i - 56) * 0.9f) + 71);
            else if (i <= 102) arr[i] = (int) ((((float) i - 76) * 0.6538f) + 89);
            else arr[i] = (int) ((((float) i - 102) / 4.4615) + 106); // 4.4615 deliberately double, not float
        }
        return arr;
    }

    private void uploadConstant(String symbol, int[] data, long bytes) {
        CUdeviceptr ptr  = new CUdeviceptr();
        long[]      size = new long[1];
        cuModuleGetGlobal(ptr, size, cuModule, symbol);
        cuMemcpyHtoD(ptr, Pointer.to(data), bytes);
    }

    private void allocatePersistentBuffers() {
        d_partialBoards = alloc((long) MAX_BOARDS * 256 * Sizeof.INT);
        d_solution      = alloc(256L * Sizeof.INT);
        d_solvedFlag    = alloc(Sizeof.INT);
        d_gpuHighScore  = alloc(Sizeof.INT);
        d_bestBoardOut  = alloc(256L * Sizeof.INT);
        d_totalSteps    = alloc(Sizeof.LONG);
        d_threadDepths  = alloc((long) MAX_BOARDS * Sizeof.INT);

        // Persistent search state for marathon thread 0
        d_persistBoard              = alloc(256L * Sizeof.INT);
        d_persistPieceStack         = alloc(256L * Sizeof.INT);
        d_persistPlacedOrientIdx    = alloc(256L * Sizeof.INT);
        d_persistBreakFallbackStack = alloc(256L * Sizeof.INT);
        d_persistBestLocalBoard     = alloc(256L * Sizeof.INT);
        d_persistInventoryMask      = alloc(4L * Sizeof.LONG);
        d_persistBreakUsedAtStep    = alloc(8L * Sizeof.INT);
        d_persistBreaksUsed         = alloc(Sizeof.INT);
        d_persistHeuristicSum       = alloc(Sizeof.INT);
        d_persistStep               = alloc(Sizeof.INT);
        d_persistPiecesNow          = alloc(Sizeof.INT);
        d_persistBestPiecesPlaced   = alloc(Sizeof.INT);
        d_persistFloor              = alloc(Sizeof.INT);
        d_marathonDepth             = alloc(Sizeof.INT);
    }

    private static CUdeviceptr alloc(long bytes) {
        CUdeviceptr p = new CUdeviceptr();
        cuMemAlloc(p, bytes);
        return p;
    }

    /** GUI/config hook: enable or disable break-tolerant placement (the
     *  kernel's kind==2 paths) for A/B comparison. Takes effect on the next
     *  runDeepDfs call. Default true -- matches every prior live run. */
    public void setBreakToleranceEnabled(boolean enabled) {
        this.breakToleranceEnabled = enabled;
    }

    /** GUI/config hook: enable or disable the south+east forward-check before
     *  every placement, for A/B comparison. Takes effect on the next
     *  runDeepDfs call. Default true -- matches every prior live run. */
    public void setLookaheadEnabled(boolean enabled) {
        this.lookaheadEnabled = enabled;
    }

    /** GUI/config hook: enable or disable marathon-thread persistence, for A/B
     *  comparison. Takes effect on the next runDeepDfs call. Default FALSE --
     *  unlike the two toggles above, this one starts off, so enabling it is an
     *  explicit opt-in to changed search behaviour. Turning it off also drops
     *  any existing checkpoint: re-enabling later must not resume a search
     *  captured under a startingStep floor that may since have moved. */
    public void setSeedPersistenceEnabled(boolean enabled) {
        this.seedPersistenceEnabled = enabled;
        if (!enabled) {
            this.persistSlotValid = false;
        }
    }

    public boolean isSeedPersistenceEnabled() {
        return seedPersistenceEnabled;
    }

    /** Forces the next launch to fresh-init instead of resuming. Called from
     *  EternitySolver.triggerBranchScrap: a teardown changes the locked prefix
     *  (and therefore startingStep), so any checkpoint taken under the old floor
     *  is no longer valid to continue from. */
    public void invalidatePersistedState() {
        this.persistSlotValid = false;
    }

    // ==========================================================
    // PHASE 2: DEEP DFS
    // ==========================================================
    public GpuResult runDeepDfs(List<int[]> seeds, int startingStep, int currentHighScore,
                                 int[] bestBoardOut) {
        return runDeepDfs(seeds, startingStep, currentHighScore, bestBoardOut, this.stepBudget);
    }

    /**
     * Same as {@link #runDeepDfs(List, int, int, int[])} but with an explicit per-thread
     * step budget instead of the constructor-derived {@link #stepBudget}. Exists for
     * {@code MarathonPersistenceCheck}, whose central assertion is that N chained resumed
     * launches equal one uninterrupted launch of N times the budget -- which is only
     * expressible if the budget can be varied per call. Production callers use the
     * 4-arg form and are unaffected.
     */
    GpuResult runDeepDfs(List<int[]> seeds, int startingStep, int currentHighScore,
                         int[] bestBoardOut, long stepBudgetOverride) {
        int numBoards = seeds.size();
        if (numBoards == 0) return new GpuResult(currentHighScore, false, 0, new int[0], 0);

        int[] flatBoards = new int[numBoards * 256];
        for (int i = 0; i < numBoards; i++)
            System.arraycopy(seeds.get(i), 0, flatBoards, i * 256, 256);

        cuMemcpyHtoD(d_partialBoards, Pointer.to(flatBoards),           (long) numBoards * 256 * Sizeof.INT);
        cuMemcpyHtoD(d_solvedFlag,    Pointer.to(new int[]{0}),         Sizeof.INT);
        cuMemcpyHtoD(d_gpuHighScore,  Pointer.to(new int[]{currentHighScore}), Sizeof.INT);
        cuMemcpyHtoD(d_totalSteps,    Pointer.to(new long[]{0L}),       Sizeof.LONG);
        cuMemcpyHtoD(d_threadDepths,  Pointer.to(new int[numBoards]),   (long) numBoards * Sizeof.INT);
        cuMemcpyHtoD(d_marathonDepth, Pointer.to(new int[]{0}),         Sizeof.INT);

        int persistResumeFlag = (seedPersistenceEnabled && persistSlotValid) ? 1 : 0;

        Pointer kernelParameters = Pointer.to(
                Pointer.to(d_partialBoards),
                Pointer.to(new int[]{numBoards}),
                Pointer.to(new int[]{startingStep}),
                Pointer.to(d_solution),
                Pointer.to(d_solvedFlag),
                Pointer.to(d_gpuHighScore),
                Pointer.to(d_bestBoardOut),
                Pointer.to(d_totalSteps),
                Pointer.to(new int[]{lockCenter ? 1 : 0}),
                Pointer.to(d_threadDepths),
                Pointer.to(new int[]{breakToleranceEnabled ? 1 : 0}),
                Pointer.to(new long[]{stepBudgetOverride}),
                Pointer.to(new int[]{lookaheadEnabled ? 1 : 0}),
                Pointer.to(new int[]{persistResumeFlag}),
                Pointer.to(d_persistBoard),
                Pointer.to(d_persistPieceStack),
                Pointer.to(d_persistPlacedOrientIdx),
                Pointer.to(d_persistBreakFallbackStack),
                Pointer.to(d_persistBestLocalBoard),
                Pointer.to(d_persistInventoryMask),
                Pointer.to(d_persistBreakUsedAtStep),
                Pointer.to(d_persistBreaksUsed),
                Pointer.to(d_persistHeuristicSum),
                Pointer.to(d_persistStep),
                Pointer.to(d_persistPiecesNow),
                Pointer.to(d_persistBestPiecesPlaced),
                Pointer.to(d_persistFloor),
                Pointer.to(d_marathonDepth)
        );

        int blockSize = 256;
        int gridSize  = (int) Math.ceil((double) numBoards / blockSize);
        cuLaunchKernel(dfsFunction, gridSize, 1, 1, blockSize, 1, 1, 0, null, kernelParameters, null);
        cuCtxSynchronize();

        int[]  resultHighScore = new int[1];
        long[] totalSteps      = new long[1];
        int[]  solved          = new int[1];
        int[]  threadDepths    = new int[numBoards];
        int[]  marathonDepth   = new int[1];

        cuMemcpyDtoH(Pointer.to(resultHighScore), d_gpuHighScore,  Sizeof.INT);
        cuMemcpyDtoH(Pointer.to(totalSteps),      d_totalSteps,    Sizeof.LONG);
        cuMemcpyDtoH(Pointer.to(solved),          d_solvedFlag,    Sizeof.INT);
        cuMemcpyDtoH(Pointer.to(threadDepths),    d_threadDepths,  (long) numBoards * Sizeof.INT);
        cuMemcpyDtoH(Pointer.to(marathonDepth),   d_marathonDepth, Sizeof.INT);

        if (seedPersistenceEnabled && solved[0] == 0) {
            persistSlotValid = true;
        } else if (solved[0] == 1) {
            persistSlotValid = false;
        }

        if (resultHighScore[0] > currentHighScore)
            cuMemcpyDtoH(Pointer.to(bestBoardOut), d_bestBoardOut, 256L * Sizeof.INT);
        if (solved[0] == 1)
            cuMemcpyDtoH(Pointer.to(bestBoardOut), d_solution,     256L * Sizeof.INT);

        return new GpuResult(resultHighScore[0], solved[0] == 1, totalSteps[0], threadDepths, marathonDepth[0]);
    }

    // ==========================================================
    // PHASE 3: REPAIR MODE (LNS)
    // ==========================================================
    public GpuResult runRepairMode(List<int[]> swissCheeseBoards, int currentHighScore,
                                    int[] bestBoardOut) {
        int numBoards = swissCheeseBoards.size();
        if (numBoards == 0) return new GpuResult(currentHighScore, false, 0, new int[0], 0);

        int[] flatBoards = new int[numBoards * 256];
        for (int i = 0; i < numBoards; i++)
            System.arraycopy(swissCheeseBoards.get(i), 0, flatBoards, i * 256, 256);

        cuMemcpyHtoD(d_partialBoards, Pointer.to(flatBoards),           (long) numBoards * 256 * Sizeof.INT);
        cuMemcpyHtoD(d_solvedFlag,    Pointer.to(new int[]{0}),         Sizeof.INT);
        cuMemcpyHtoD(d_gpuHighScore,  Pointer.to(new int[]{currentHighScore}), Sizeof.INT);
        cuMemcpyHtoD(d_totalSteps,    Pointer.to(new long[]{0L}),       Sizeof.LONG);

        int maxStepsPerThread = 100_000;
        Pointer kernelParameters = Pointer.to(
                Pointer.to(d_partialBoards),
                Pointer.to(new int[]{numBoards}),
                Pointer.to(d_solution),
                Pointer.to(d_solvedFlag),
                Pointer.to(d_gpuHighScore),
                Pointer.to(d_bestBoardOut),
                Pointer.to(d_totalSteps),
                Pointer.to(new int[]{maxStepsPerThread})
        );

        int blockSize = 256;
        int gridSize  = (int) Math.ceil((double) numBoards / blockSize);
        cuLaunchKernel(repairFunction, gridSize, 1, 1, blockSize, 1, 1, 0, null, kernelParameters, null);
        cuCtxSynchronize();

        int[]  resultHighScore = new int[1];
        long[] totalSteps      = new long[1];
        int[]  solved          = new int[1];

        cuMemcpyDtoH(Pointer.to(resultHighScore), d_gpuHighScore, Sizeof.INT);
        cuMemcpyDtoH(Pointer.to(totalSteps),      d_totalSteps,   Sizeof.LONG);
        cuMemcpyDtoH(Pointer.to(solved),          d_solvedFlag,   Sizeof.INT);

        long steps = totalSteps[0];
        if (steps == 0) steps = (long) numBoards * 150_000;

        if (resultHighScore[0] > currentHighScore)
            cuMemcpyDtoH(Pointer.to(bestBoardOut), d_bestBoardOut, 256L * Sizeof.INT);
        if (solved[0] == 1)
            cuMemcpyDtoH(Pointer.to(bestBoardOut), d_solution,     256L * Sizeof.INT);

        return new GpuResult(resultHighScore[0], solved[0] == 1, steps, new int[0], 0);
    }

    public record GpuResult(int newHighScore, boolean solved, long stepsTaken, int[] threadDepths, int marathonDepth) {
        public GpuResult(int newHighScore, boolean solved, long stepsTaken, int[] threadDepths) {
            this(newHighScore, solved, stepsTaken, threadDepths, 0);
        }
    }
}

package dk.puzzle.gpu;

import dk.puzzle.model.PieceInventory;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.JCuda;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
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

    // Reverse lookup (packed piece value -> physical id), built once so seed
    // boards can be flattened with their physical ids already resolved —
    // avoids the kernel having to linear-scan c_allOrientations (up to 1024
    // entries) for every already-placed piece on every launch. Populated in
    // orientation-index order so it agrees with the kernel's old behaviour
    // of taking the first (lowest-index) match on any accidental collision.
    private final Map<Integer, Integer> pieceValueToPhysicalId;

    // Persistent device buffers — allocated once, reused every launch
    private CUdeviceptr d_partialBoards;
    private CUdeviceptr d_partialBoardPhysIds;
    private CUdeviceptr d_solution;
    private CUdeviceptr d_solvedFlag;
    private CUdeviceptr d_gpuHighScore;
    private CUdeviceptr d_bestBoardOut;
    private CUdeviceptr d_totalSteps;
    private CUdeviceptr d_threadDepths;

    public GpuEngine(PieceInventory inventory, boolean lockCenter, int[] buildOrder) {
        JCuda.cudaSetDeviceFlags(JCuda.cudaDeviceScheduleBlockingSync);
        this.inventory  = inventory;
        this.lockCenter = lockCenter;
        this.stepBudget = lockCenter ? STEP_BUDGET_LOCKED : STEP_BUDGET_UNLOCKED;
        this.pieceValueToPhysicalId = new HashMap<>();
        for (int o = 0; o < 1024; o++) {
            pieceValueToPhysicalId.putIfAbsent(inventory.allOrientations[o], inventory.physicalMapping[o]);
        }
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

        allocatePersistentBuffers();
    }

    private void uploadConstant(String symbol, int[] data, long bytes) {
        CUdeviceptr ptr  = new CUdeviceptr();
        long[]      size = new long[1];
        cuModuleGetGlobal(ptr, size, cuModule, symbol);
        cuMemcpyHtoD(ptr, Pointer.to(data), bytes);
    }

    private void allocatePersistentBuffers() {
        d_partialBoards        = alloc((long) MAX_BOARDS * 256 * Sizeof.INT);
        d_partialBoardPhysIds  = alloc((long) MAX_BOARDS * 256 * Sizeof.INT);
        d_solution      = alloc(256L * Sizeof.INT);
        d_solvedFlag    = alloc(Sizeof.INT);
        d_gpuHighScore  = alloc(Sizeof.INT);
        d_bestBoardOut  = alloc(256L * Sizeof.INT);
        d_totalSteps    = alloc(Sizeof.LONG);
        d_threadDepths  = alloc((long) MAX_BOARDS * Sizeof.INT);
    }

    private static CUdeviceptr alloc(long bytes) {
        CUdeviceptr p = new CUdeviceptr();
        cuMemAlloc(p, bytes);
        return p;
    }

    // ==========================================================
    // PHASE 2: DEEP DFS
    // ==========================================================
    public GpuResult runDeepDfs(List<int[]> seeds, int startingStep, int currentHighScore,
                                 int[] bestBoardOut) {
        int numBoards = seeds.size();
        if (numBoards == 0) return new GpuResult(currentHighScore, false, 0, new int[0]);

        int[] flatBoards  = new int[numBoards * 256];
        int[] flatPhysIds = new int[numBoards * 256];
        for (int i = 0; i < numBoards; i++) {
            int[] seed = seeds.get(i);
            System.arraycopy(seed, 0, flatBoards, i * 256, 256);
            for (int c = 0; c < 256; c++) {
                int piece = seed[c];
                flatPhysIds[i * 256 + c] = piece == -1
                        ? -1
                        : pieceValueToPhysicalId.getOrDefault(piece, -1);
            }
        }

        cuMemcpyHtoD(d_partialBoards,       Pointer.to(flatBoards),    (long) numBoards * 256 * Sizeof.INT);
        cuMemcpyHtoD(d_partialBoardPhysIds, Pointer.to(flatPhysIds),   (long) numBoards * 256 * Sizeof.INT);
        cuMemcpyHtoD(d_solvedFlag,    Pointer.to(new int[]{0}),         Sizeof.INT);
        cuMemcpyHtoD(d_gpuHighScore,  Pointer.to(new int[]{currentHighScore}), Sizeof.INT);
        cuMemcpyHtoD(d_totalSteps,    Pointer.to(new long[]{0L}),       Sizeof.LONG);
        cuMemcpyHtoD(d_threadDepths,  Pointer.to(new int[numBoards]),   (long) numBoards * Sizeof.INT);

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
                Pointer.to(new int[]{0}),
                Pointer.to(new long[]{stepBudget}),
                Pointer.to(d_partialBoardPhysIds)
        );

        int blockSize = 256;
        int gridSize  = (int) Math.ceil((double) numBoards / blockSize);
        cuLaunchKernel(dfsFunction, gridSize, 1, 1, blockSize, 1, 1, 0, null, kernelParameters, null);
        cuCtxSynchronize();

        int[]  resultHighScore = new int[1];
        long[] totalSteps      = new long[1];
        int[]  solved          = new int[1];
        int[]  threadDepths    = new int[numBoards];

        cuMemcpyDtoH(Pointer.to(resultHighScore), d_gpuHighScore, Sizeof.INT);
        cuMemcpyDtoH(Pointer.to(totalSteps),      d_totalSteps,   Sizeof.LONG);
        cuMemcpyDtoH(Pointer.to(solved),          d_solvedFlag,   Sizeof.INT);
        cuMemcpyDtoH(Pointer.to(threadDepths),    d_threadDepths, (long) numBoards * Sizeof.INT);

        if (resultHighScore[0] > currentHighScore)
            cuMemcpyDtoH(Pointer.to(bestBoardOut), d_bestBoardOut, 256L * Sizeof.INT);
        if (solved[0] == 1)
            cuMemcpyDtoH(Pointer.to(bestBoardOut), d_solution,     256L * Sizeof.INT);

        return new GpuResult(resultHighScore[0], solved[0] == 1, totalSteps[0], threadDepths);
    }

    // ==========================================================
    // PHASE 3: REPAIR MODE (LNS)
    // ==========================================================
    public GpuResult runRepairMode(List<int[]> swissCheeseBoards, int currentHighScore,
                                    int[] bestBoardOut) {
        int numBoards = swissCheeseBoards.size();
        if (numBoards == 0) return new GpuResult(currentHighScore, false, 0, new int[0]);

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

        return new GpuResult(resultHighScore[0], solved[0] == 1, steps, new int[0]);
    }

    public void hardReset() {
        freePersistentBuffers();
        allocatePersistentBuffers();
    }

    private void freePersistentBuffers() {
        free(d_partialBoards); d_partialBoards = null;
        free(d_solution);      d_solution      = null;
        free(d_solvedFlag);    d_solvedFlag    = null;
        free(d_gpuHighScore);  d_gpuHighScore  = null;
        free(d_bestBoardOut);  d_bestBoardOut  = null;
        free(d_totalSteps);    d_totalSteps    = null;
        free(d_threadDepths);  d_threadDepths  = null;
    }

    private static void free(CUdeviceptr p) {
        if (p != null) cuMemFree(p);
    }

    public record GpuResult(int newHighScore, boolean solved, long stepsTaken, int[] threadDepths) {}
}

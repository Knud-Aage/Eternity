package dk.puzzle.gpu;

import dk.puzzle.model.PieceInventory;
import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.JCuda;

import java.util.List;
import static jcuda.driver.JCudaDriver.*;

/**
 * The GPU-accelerated engine for the Eternity II solver, interface via JCuda.
 * * <p>This class manages the lifecycle of CUDA resources, including context
 * initialization, PTX module loading, and the synchronization of board data
 * between Host (CPU) and Device (GPU) memory. It provides entry points for
 * both the Phase 2 exploratory DFS and the Phase 3 repair-based LNS.</p>
 */
public class GpuEngine {
    private CUfunction dfsFunction;
    private CUfunction repairFunction;
    private final PieceInventory inventory;
    private final boolean lockCenter;

    /**
     * Constructs a new {@code GpuEngine} and initializes the CUDA driver.
     * * @param inventory The piece inventory used to populate orientation and
     * physical mapping tables on the GPU.
     * @param lockCenter Whether to enforce the fixed center piece constraint
     * within the GPU kernels.
     */
    public GpuEngine(PieceInventory inventory, boolean lockCenter) {
        JCuda.cudaSetDeviceFlags(JCuda.cudaDeviceScheduleBlockingSync);
        this.inventory = inventory;
        this.lockCenter = lockCenter;
        initCUDA();
    }

    private void initCUDA() {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext cuContext = new CUcontext();
        cuCtxCreate(cuContext, 0, device);
        CUmodule cuModule = new CUmodule();
        cuModuleLoad(cuModule, "SolveEternityKernel.ptx");

        CUdeviceptr c_all = new CUdeviceptr();
        cuModuleGetGlobal(c_all, null, cuModule, "c_allOrientations");
        cuMemcpyHtoD(c_all, Pointer.to(inventory.allOrientations), 1024L * Sizeof.INT);

        CUdeviceptr c_phys = new CUdeviceptr();
        cuModuleGetGlobal(c_phys, null, cuModule, "c_physicalMapping");
        cuMemcpyHtoD(c_phys, Pointer.to(inventory.physicalMapping), 1024L * Sizeof.INT);

        dfsFunction = new CUfunction();
        cuModuleGetFunction(dfsFunction, cuModule, "solvePBP");

        repairFunction = new CUfunction();
        cuModuleGetFunction(repairFunction, cuModule, "solveRepairMode");
    }

    // ==========================================================
    // PHASE 2: DEEP DFS (GPU EXPLORER)
    // ==========================================================
    public GpuResult runDeepDfs(List<int[]> seeds, int startingStep, int currentHighScore,
                                 int[] bestBoardOut, int[] buildOrder, long stepBudget) {
        int numBoards = seeds.size();
        if (numBoards == 0) return new GpuResult(currentHighScore, false, 0, new int[0]);

        int[] flatBoards = new int[numBoards * 256];
        for (int i = 0; i < numBoards; i++)
            System.arraycopy(seeds.get(i), 0, flatBoards, i * 256, 256);

        CUdeviceptr d_partialBoards = new CUdeviceptr();
        cuMemAlloc(d_partialBoards, (long) numBoards * 256 * Sizeof.INT);
        cuMemcpyHtoD(d_partialBoards, Pointer.to(flatBoards), (long) numBoards * 256 * Sizeof.INT);

        CUdeviceptr d_buildOrder = new CUdeviceptr();
        cuMemAlloc(d_buildOrder, 256L * Sizeof.INT);
        cuMemcpyHtoD(d_buildOrder, Pointer.to(buildOrder), 256L * Sizeof.INT);

        CUdeviceptr d_solution = new CUdeviceptr();
        cuMemAlloc(d_solution, 256L * Sizeof.INT);

        CUdeviceptr d_solvedFlag = new CUdeviceptr();
        cuMemAlloc(d_solvedFlag, Sizeof.INT);
        cuMemcpyHtoD(d_solvedFlag, Pointer.to(new int[]{0}), Sizeof.INT);

        CUdeviceptr d_gpuHighScore = new CUdeviceptr();
        cuMemAlloc(d_gpuHighScore, Sizeof.INT);
        cuMemcpyHtoD(d_gpuHighScore, Pointer.to(new int[]{currentHighScore}), Sizeof.INT);

        CUdeviceptr d_bestBoardOut = new CUdeviceptr();
        cuMemAlloc(d_bestBoardOut, 256L * Sizeof.INT);

        CUdeviceptr d_totalSteps = new CUdeviceptr();
        cuMemAlloc(d_totalSteps, Sizeof.LONG);
        cuMemcpyHtoD(d_totalSteps, Pointer.to(new long[]{0L}), Sizeof.LONG);

        CUdeviceptr d_threadDepths = new CUdeviceptr();
        cuMemAlloc(d_threadDepths, (long) numBoards * Sizeof.INT);
        cuMemcpyHtoD(d_threadDepths, Pointer.to(new int[numBoards]), (long) numBoards * Sizeof.INT);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(d_partialBoards),
                Pointer.to(new int[]{numBoards}),
                Pointer.to(new int[]{startingStep}),
                Pointer.to(d_buildOrder),
                Pointer.to(d_solution),
                Pointer.to(d_solvedFlag),
                Pointer.to(d_gpuHighScore),
                Pointer.to(d_bestBoardOut),
                Pointer.to(d_totalSteps),
                Pointer.to(new int[]{lockCenter ? 1 : 0}),
                Pointer.to(d_threadDepths),
                Pointer.to(new long[]{stepBudget})
        );

        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) numBoards / blockSizeX);

        cuLaunchKernel(dfsFunction,
                gridSizeX, 1, 1,
                blockSizeX, 1, 1,
                0, null,
                kernelParameters, null
        );

        cuCtxSynchronize();

        int[] resultHighScore = new int[1];
        cuMemcpyDtoH(Pointer.to(resultHighScore), d_gpuHighScore, Sizeof.INT);

        long[] totalSteps = new long[1];
        cuMemcpyDtoH(Pointer.to(totalSteps), d_totalSteps, Sizeof.LONG);

        int[] solved = new int[1];
        cuMemcpyDtoH(Pointer.to(solved), d_solvedFlag, Sizeof.INT);

        int[] threadDepths = new int[numBoards];
        cuMemcpyDtoH(Pointer.to(threadDepths), d_threadDepths, (long) numBoards * Sizeof.INT);

        int cleanScore = resultHighScore[0] & 0x0FFFFFFF;
        if (cleanScore > 0) {
            cuMemcpyDtoH(Pointer.to(bestBoardOut), d_bestBoardOut, 256L * Sizeof.INT);
        }

        if (solved[0] == 1) {
            cuMemcpyDtoH(Pointer.to(bestBoardOut), d_solution, 256L * Sizeof.INT);
        }

        cuMemFree(d_partialBoards); cuMemFree(d_buildOrder);
        cuMemFree(d_solution); cuMemFree(d_solvedFlag); cuMemFree(d_gpuHighScore);
        cuMemFree(d_bestBoardOut); cuMemFree(d_totalSteps); cuMemFree(d_threadDepths);

        return new GpuResult(resultHighScore[0], solved[0] == 1, totalSteps[0], threadDepths);
    }

    // ==========================================================
    // HYPER DIVE (Fixed execution)
    // ==========================================================
    public GpuResult runHyperDive(int[] goldmineSeed, int startingStep, int currentHighScore,
                                   int[] bestBoardOut, int[] buildOrder, long stepBudget) {

        // We must feed the GPU at least 1 full block (256 threads) to prevent
        // memory misalignment and CUDA_ERROR_INVALID_VALUE.
        List<int[]> seeds = new java.util.ArrayList<>(256);
        for (int i = 0; i < 256; i++) {
            seeds.add(goldmineSeed.clone());
        }

        return runDeepDfs(seeds, startingStep, currentHighScore, bestBoardOut, buildOrder, stepBudget);
    }

    // ==========================================================
    // PHASE 3: REPAIR MODE (LNS hole-filling)
    // ==========================================================
    public GpuResult runRepairMode(List<int[]> swissCheeseBoards, int currentHighScore, int[] bestBoardOut, long stepBudget) {
        int numBoards = swissCheeseBoards.size();
        if (numBoards == 0) return new GpuResult(currentHighScore, false, 0, new int[0]);

        int[] flatBoards = new int[numBoards * 256];
        for (int i = 0; i < numBoards; i++)
            System.arraycopy(swissCheeseBoards.get(i), 0, flatBoards, i * 256, 256);

        CUdeviceptr d_partialBoards = new CUdeviceptr();
        cuMemAlloc(d_partialBoards, (long) numBoards * 256 * Sizeof.INT);
        cuMemcpyHtoD(d_partialBoards, Pointer.to(flatBoards), (long) numBoards * 256 * Sizeof.INT);

        CUdeviceptr d_solution = new CUdeviceptr();
        cuMemAlloc(d_solution, 256L * Sizeof.INT);

        CUdeviceptr d_solvedFlag = new CUdeviceptr();
        cuMemAlloc(d_solvedFlag, Sizeof.INT);
        cuMemcpyHtoD(d_solvedFlag, Pointer.to(new int[]{0}), Sizeof.INT);

        CUdeviceptr d_gpuHighScore = new CUdeviceptr();
        cuMemAlloc(d_gpuHighScore, Sizeof.INT);
        cuMemcpyHtoD(d_gpuHighScore, Pointer.to(new int[]{currentHighScore}), Sizeof.INT);

        CUdeviceptr d_bestBoardOut = new CUdeviceptr();
        cuMemAlloc(d_bestBoardOut, 256L * Sizeof.INT);

        CUdeviceptr d_totalSteps = new CUdeviceptr();
        cuMemAlloc(d_totalSteps, Sizeof.LONG);
        cuMemcpyHtoD(d_totalSteps, Pointer.to(new long[]{0L}), Sizeof.LONG);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(d_partialBoards),
                Pointer.to(new int[]{numBoards}),
                Pointer.to(d_solution),
                Pointer.to(d_solvedFlag),
                Pointer.to(d_gpuHighScore),
                Pointer.to(d_bestBoardOut),
                Pointer.to(d_totalSteps),
                Pointer.to(new long[]{stepBudget})
        );

        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) numBoards / blockSizeX);

        long steps = 0;
        int maxRetries = 10;
        int retryCount = 0;
        boolean success = false;

        while (retryCount < maxRetries && !success) {
            try {
                cuLaunchKernel(repairFunction,
                        gridSizeX, 1, 1,
                        blockSizeX, 1, 1,
                        0, null,
                        kernelParameters, null
                );
                cuCtxSynchronize();

                long[] totalStepsArray = new long[1];
                cuMemcpyDtoH(Pointer.to(totalStepsArray), d_totalSteps, Sizeof.LONG);
                steps = totalStepsArray[0];

                success = true;
            } catch (CudaException e) {
                if (e.getMessage().contains("CUDA_ERROR_LAUNCH_TIMEOUT")) {
                    System.err.println(">>> [HARDWARE TDR] Phase 3 Repair timed out (TDR limit exceeded). Retrying...");
                    retryCount++;
                    if (retryCount >= maxRetries) {
                        System.err.println(">>> [HARDWARE TDR FATAL] Phase 3 Repair failed after " + maxRetries + " retries.");
                        break;
                    }
                } else {
                    throw e;
                }
            }
        }

        if (steps == 0) steps = (long) numBoards * 150000;

        int[] resultHighScore = new int[1];
        cuMemcpyDtoH(Pointer.to(resultHighScore), d_gpuHighScore, Sizeof.INT);

        int[] solved = new int[1];
        cuMemcpyDtoH(Pointer.to(solved), d_solvedFlag, Sizeof.INT);

        if (resultHighScore[0] > currentHighScore) {
            cuMemcpyDtoH(Pointer.to(bestBoardOut), d_bestBoardOut, 256L * Sizeof.INT);
        }
        if (solved[0] == 1) {
            cuMemcpyDtoH(Pointer.to(bestBoardOut), d_solution, 256L * Sizeof.INT);
        }

        cuMemFree(d_partialBoards);
        cuMemFree(d_solution); cuMemFree(d_solvedFlag); cuMemFree(d_gpuHighScore);
        cuMemFree(d_bestBoardOut); cuMemFree(d_totalSteps);

        return new GpuResult(resultHighScore[0], solved[0] == 1, steps, new int[0]);
    }

    public static class GpuResult {
        public final int newHighScore;
        public final boolean solved;
        public final long stepsTaken;
        public final int[] threadDepths;

        public GpuResult(int h, boolean s, long st) {
            newHighScore = h; solved = s; stepsTaken = st; threadDepths = new int[0];
        }
        public GpuResult(int h, boolean s, long st, int[] td) {
            newHighScore = h; solved = s; stepsTaken = st; threadDepths = td;
        }
        public long stepsTaken() { return stepsTaken; }
        public int newHighScore() { return newHighScore; }
        public boolean solved() { return solved; }
        public int[] threadDepths() { return threadDepths; }
    }
}
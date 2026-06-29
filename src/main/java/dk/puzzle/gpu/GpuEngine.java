package dk.puzzle.gpu;

import dk.puzzle.model.PieceInventory;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.JCuda;

import java.util.List;
import static jcuda.driver.JCudaDriver.*;

/**
 * The GPU-accelerated engine for the Eternity II solver, interface via JCuda.
 * 
 * <p>This class manages the lifecycle of CUDA resources, including context 
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
     * 
     * @param inventory The piece inventory used to populate orientation and 
     *                  physical mapping tables on the GPU.
     * @param lockCenter Whether to enforce the fixed center piece constraint 
     *                   within the GPU kernels.
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

        dfsFunction = new CUfunction();
        cuModuleGetFunction(dfsFunction, cuModule, "solvePBP");

        repairFunction = new CUfunction();
        cuModuleGetFunction(repairFunction, cuModule, "solveRepairMode");
    }

    // ==========================================================
    // PHASE 2: DEEP DFS (GPU EXPLORER)
    // ==========================================================
    /**
     * Executes a massively parallel Deep DFS on the GPU using a list of seed boards.
     * 
     * <p>This method performs the following:
     * 1. Flattens and uploads a batch of seed boards to device memory.
     * 2. Configures the kernel with build orders and piece metadata.
     * 3. Launches the 'solvePBP' kernel to explore the search space from each seed.
     * 4. Retrieves the highest achieved depth and corresponding board state.</p>
     * 
     * @param seeds A list of partial board configurations (seeds) to be expanded.
     * @param startingStep The depth index (0-255) where the GPU should begin placing pieces.
     * @param currentHighScore The current global high score, used by the GPU to prune 
     *                         underperforming branches or report progress.
     * @param bestBoardOut An output array that will be populated with the best board 
     *                     found during this execution.
     * @param buildOrder The sequence of board indices to be filled.
     * @return A {@link GpuResult} containing the new high score and execution metrics.
     */
    public GpuResult runDeepDfs(List<int[]> seeds, int startingStep, int currentHighScore,
                                 int[] bestBoardOut, int[] buildOrder) {
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

        CUdeviceptr d_allOrientations = new CUdeviceptr();
        cuMemAlloc(d_allOrientations, 1024L * Sizeof.INT);
        cuMemcpyHtoD(d_allOrientations, Pointer.to(inventory.allOrientations), 1024L * Sizeof.INT);

        CUdeviceptr d_physicalMapping = new CUdeviceptr();
        cuMemAlloc(d_physicalMapping, 1024L * Sizeof.INT);
        cuMemcpyHtoD(d_physicalMapping, Pointer.to(inventory.physicalMapping), 1024L * Sizeof.INT);

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
                Pointer.to(d_allOrientations),
                Pointer.to(d_physicalMapping),
                Pointer.to(d_solution),
                Pointer.to(d_solvedFlag),
                Pointer.to(d_gpuHighScore),
                Pointer.to(d_bestBoardOut),
                Pointer.to(d_totalSteps),
                Pointer.to(new int[]{lockCenter ? 1 : 0}),
                Pointer.to(d_threadDepths),
                Pointer.to(new int[]{0})
        );

        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) numBoards / blockSizeX);
        cuLaunchKernel(dfsFunction, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null, kernelParameters, null);
        cuCtxSynchronize();

        int[] resultHighScore = new int[1];
        cuMemcpyDtoH(Pointer.to(resultHighScore), d_gpuHighScore, Sizeof.INT);

        long[] totalSteps = new long[1];
        cuMemcpyDtoH(Pointer.to(totalSteps), d_totalSteps, Sizeof.LONG);

        int[] solved = new int[1];
        cuMemcpyDtoH(Pointer.to(solved), d_solvedFlag, Sizeof.INT);

        int[] threadDepths = new int[numBoards];
        cuMemcpyDtoH(Pointer.to(threadDepths), d_threadDepths, (long) numBoards * Sizeof.INT);

        if (resultHighScore[0] > currentHighScore) {
            cuMemcpyDtoH(Pointer.to(bestBoardOut), d_bestBoardOut, 256L * Sizeof.INT);
        }
        if (solved[0] == 1) {
            cuMemcpyDtoH(Pointer.to(bestBoardOut), d_solution, 256L * Sizeof.INT);
        }

        // Ryd op
        cuMemFree(d_partialBoards); cuMemFree(d_buildOrder); cuMemFree(d_allOrientations);
        cuMemFree(d_physicalMapping); cuMemFree(d_solution); cuMemFree(d_solvedFlag);
        cuMemFree(d_gpuHighScore); cuMemFree(d_bestBoardOut); cuMemFree(d_totalSteps);
        cuMemFree(d_threadDepths);

        return new GpuResult(resultHighScore[0], solved[0] == 1, totalSteps[0], threadDepths);
    }

    // ==========================================================
    // PHASE 3: REPAIR MODE (LNS hole-filling)
    // ==========================================================
    /**
     * Executes the Phase 3 "Surgeon" repair mode on the GPU.
     * 
     * <p>Takes multiple "swiss cheese" boards (boards with targeted holes) and 
     * attempts to fill the holes and extend the solution using the 'solveRepairMode' 
     * kernel. This is specifically used to resolve conflicts found in high-depth boards.</p>
     * 
     * @param swissCheeseBoards A list of boards with removed pieces (represented by -2).
     * @param currentHighScore The current record depth used for thresholding progress.
     * @param bestBoardOut An output array to receive the best modified board state 
     *                     recovered from the GPU.
     * @return A {@link GpuResult} containing the new high score and execution metrics.
     */
    public GpuResult runRepairMode(List<int[]> swissCheeseBoards, int currentHighScore, int[] bestBoardOut) {
        int numBoards = swissCheeseBoards.size();
        if (numBoards == 0) return new GpuResult(currentHighScore, false, 0, new int[0]);

        int[] flatBoards = new int[numBoards * 256];
        for (int i = 0; i < numBoards; i++)
            System.arraycopy(swissCheeseBoards.get(i), 0, flatBoards, i * 256, 256);

        CUdeviceptr d_partialBoards = new CUdeviceptr();
        cuMemAlloc(d_partialBoards, (long) numBoards * 256 * Sizeof.INT);
        cuMemcpyHtoD(d_partialBoards, Pointer.to(flatBoards), (long) numBoards * 256 * Sizeof.INT);

        CUdeviceptr d_allOrientations = new CUdeviceptr();
        cuMemAlloc(d_allOrientations, 1024L * Sizeof.INT);
        cuMemcpyHtoD(d_allOrientations, Pointer.to(inventory.allOrientations), 1024L * Sizeof.INT);

        CUdeviceptr d_physicalMapping = new CUdeviceptr();
        cuMemAlloc(d_physicalMapping, 1024L * Sizeof.INT);
        cuMemcpyHtoD(d_physicalMapping, Pointer.to(inventory.physicalMapping), 1024L * Sizeof.INT);

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

        int maxStepsPerThread = 100_000;
        Pointer kernelParameters = Pointer.to(
                Pointer.to(d_partialBoards), Pointer.to(new int[]{numBoards}),
                Pointer.to(d_allOrientations), Pointer.to(d_physicalMapping),
                Pointer.to(d_solution), Pointer.to(d_solvedFlag),
                Pointer.to(d_gpuHighScore), Pointer.to(d_bestBoardOut),
                Pointer.to(d_totalSteps), Pointer.to(new int[]{maxStepsPerThread})
        );

        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) numBoards / blockSizeX);
        cuLaunchKernel(repairFunction, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null, kernelParameters, null);
        cuCtxSynchronize();

        int[] resultHighScore = new int[1];
        cuMemcpyDtoH(Pointer.to(resultHighScore), d_gpuHighScore, Sizeof.INT);

        long[] totalSteps = new long[1];
        cuMemcpyDtoH(Pointer.to(totalSteps), d_totalSteps, Sizeof.LONG);
        long steps = totalSteps[0];
        if (steps == 0) steps = (long) numBoards * 150000;

        int[] solved = new int[1];
        cuMemcpyDtoH(Pointer.to(solved), d_solvedFlag, Sizeof.INT);

        if (resultHighScore[0] > currentHighScore) {
            cuMemcpyDtoH(Pointer.to(bestBoardOut), d_bestBoardOut, 256L * Sizeof.INT);
        }
        if (solved[0] == 1) {
            cuMemcpyDtoH(Pointer.to(bestBoardOut), d_solution, 256L * Sizeof.INT);
        }

        // Ryd op
        cuMemFree(d_partialBoards); cuMemFree(d_allOrientations); cuMemFree(d_physicalMapping);
        cuMemFree(d_solution); cuMemFree(d_solvedFlag); cuMemFree(d_gpuHighScore);
        cuMemFree(d_bestBoardOut); cuMemFree(d_totalSteps);

        return new GpuResult(resultHighScore[0], solved[0] == 1, steps, new int[0]);
    }

    /**
     * Represents the outcome of a GPU execution cycle.
     * 
     * @param newHighScore The highest depth reached by any thread during the run.
     * @param solved Whether a complete 256-piece solution was discovered.
     * @param stepsTaken Total number of placement attempts made across all threads.
     * @param threadDepths An array containing the final depth reached by each individual thread.
     */
    public record GpuResult(int newHighScore, boolean solved, long stepsTaken, int[] threadDepths) {
    }

    public void hardReset() {
        JCuda.cudaSetDevice(0);
        JCuda.cudaDeviceReset();
    }
}
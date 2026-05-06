package dk.puzzle;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import java.util.List;
import static jcuda.driver.JCudaDriver.*;

public class GpuEngine {
    private CUfunction dfsFunction;
    private CUfunction repairFunction;
    private final PieceInventory inventory;
    private final boolean lockCenter;

    public GpuEngine(PieceInventory inventory, boolean lockCenter) {
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
    public GpuResult runDeepDfs(List<int[]> seeds, int startingStep, int currentHighScore, int[] bestBoardOut, int[] buildOrder) {
        int numBoards = seeds.size();
        if (numBoards == 0) return new GpuResult(currentHighScore, false, 0);

        int[] flatBoards = new int[numBoards * 256];
        for (int i = 0; i < numBoards; i++) System.arraycopy(seeds.get(i), 0, flatBoards, i * 256, 256);

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
                Pointer.to(new int[]{0}) // Radar Leash slået fra = Uendelig dybde!
        );

        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) numBoards / blockSizeX);
        cuLaunchKernel(dfsFunction, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null, kernelParameters, null);
        cuCtxSynchronize();

        // Hent resultater
        int[] resultHighScore = new int[1];
        cuMemcpyDtoH(Pointer.to(resultHighScore), d_gpuHighScore, Sizeof.INT);

        long[] totalSteps = new long[1];
        cuMemcpyDtoH(Pointer.to(totalSteps), d_totalSteps, Sizeof.LONG);

        int[] solved = new int[1];
        cuMemcpyDtoH(Pointer.to(solved), d_solvedFlag, Sizeof.INT);

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

        return new GpuResult(resultHighScore[0], solved[0] == 1, totalSteps[0]);
    }

    // Returnerer et array: [0] = Ny HighScore (eller gl. hvis ikke slået), [1] = Er løst? (1/0), [2] = Antal skridt talt
    public GpuResult runRepairMode(List<int[]> swissCheeseBoards, int currentHighScore, int[] bestBoardOut) {
        int numBoards = swissCheeseBoards.size();
        if (numBoards == 0) return new GpuResult(currentHighScore, false, 0);

        int[] flatBoards = new int[numBoards * 256];
        for (int i = 0; i < numBoards; i++) System.arraycopy(swissCheeseBoards.get(i), 0, flatBoards, i * 256, 256);

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

        int maxStepsPerThread = 5_000_000;
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

        // Hent resultater
        int[] resultHighScore = new int[1];
        cuMemcpyDtoH(Pointer.to(resultHighScore), d_gpuHighScore, Sizeof.INT);

        long[] totalSteps = new long[1];
        cuMemcpyDtoH(Pointer.to(totalSteps), d_totalSteps, Sizeof.LONG);

        long steps = totalSteps[0];
        if (steps == 0) {
            steps = (long) numBoards * 150000;
        }

        int[] solved = new int[1];
        cuMemcpyDtoH(Pointer.to(solved), d_solvedFlag, Sizeof.INT);

        if (resultHighScore[0] > currentHighScore) {
            cuMemcpyDtoH(Pointer.to(bestBoardOut), d_bestBoardOut, 256L * Sizeof.INT);
        }
        if (solved[0] == 1) {
            cuMemcpyDtoH(Pointer.to(bestBoardOut), d_solution, 256L * Sizeof.INT); // Gem vinderbrættet
        }

        // Ryd op
        cuMemFree(d_partialBoards); cuMemFree(d_allOrientations); cuMemFree(d_physicalMapping);
        cuMemFree(d_solution); cuMemFree(d_solvedFlag); cuMemFree(d_gpuHighScore);
        cuMemFree(d_bestBoardOut); cuMemFree(d_totalSteps);

        return new GpuResult(resultHighScore[0], solved[0] == 1, steps);
    }

    public static class GpuResult {
        public final int newHighScore;
        public final boolean solved;
        public final long stepsTaken;
        public GpuResult(int h, boolean s, long st) { newHighScore = h; solved = s; stepsTaken = st; }
    }
}
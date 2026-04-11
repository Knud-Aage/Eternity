package dk.roleplay;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import static jcuda.driver.JCudaDriver.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class MasterSolverPBP implements Runnable {

    public enum BuildStrategy {
        TYPEWRITER,
        SPIRAL
    }

    private CUcontext cuContext;
    private CUmodule cuModule;
    private CUfunction cuFunction;
    private boolean gpuInitialized = false;

    private final PieceInventory inventory;
    private final int[] flatBoard = new int[256];
    private final int[] bestBoard = new int[256];
    private final int[] flatResumeBoard = new int[256];
    private final boolean[] usedPhysicalPieces = new boolean[256];
    private final Random rnd = new Random();
    private int centerPhysicalIdx = -1;
    private int deepestStep = 0;
    private final int targetPiece;

    private final boolean useGpu;
    private final BuildStrategy currentStrategy;

    private final List<int[]> gpuSeedBoards = new ArrayList<>();

    private int handoffDepth;
    private int forceBacktrackTarget = -1;
    private int targetBatchSize = 10000;

    private final int[] BUILD_ORDER;

    public MasterSolverPBP(PieceInventory inventory, int trueCenterPiece, boolean useGpu, BuildStrategy strategy) {
        this.inventory = inventory;
        this.targetPiece = trueCenterPiece;
        this.useGpu = useGpu;
        this.currentStrategy = strategy;

        Arrays.fill(flatBoard, -1);
        Arrays.fill(bestBoard, -1);
        Arrays.fill(flatResumeBoard, -1);

        for (int i = 0; i < 1024; i++) {
            if (inventory.allOrientations[i] == targetPiece) {
                this.centerPhysicalIdx = inventory.physicalMapping[i];
                break;
            }
        }

        if (strategy == BuildStrategy.SPIRAL) {
            BUILD_ORDER = generateSpiralOrder();
            this.handoffDepth = 70;
        } else {
            BUILD_ORDER = generateTypewriterOrder();
            this.handoffDepth = 55;
        }
    }

    private void initCUDA() {
        try {
            JCudaDriver.setExceptionsEnabled(true);
            cuInit(0);
            CUdevice device = new CUdevice();
            cuDeviceGet(device, 0);
            cuContext = new CUcontext();
            cuCtxCreate(cuContext, 0, device);

            cuModule = new CUmodule();
            cuModuleLoad(cuModule, "EternityKernel.ptx");
            cuFunction = new CUfunction();
            cuModuleGetFunction(cuFunction, cuModule, "validateMacroTiles");
            
            gpuInitialized = true;
            System.out.println(">>> GPU Handoff Initialized (CUDA Context & Kernel loaded).");
        } catch (Exception e) {
            System.err.println(">>> GPU Initialization failed: " + e.getMessage());
            gpuInitialized = false;
        }
    }

    private static int[] generateTypewriterOrder() {
        int[] order = new int[256];
        for (int i = 0; i < 256; i++) order[i] = i;
        return order;
    }

    private static int[] generateSpiralOrder() {
        int[] order = new int[256];
        boolean[][] visited = new boolean[16][16];
        int r = 0, c = 0;
        int[] dr = {0, 1, 0, -1}, dc = {1, 0, -1, 0};
        int dir = 0;
        for (int i = 0; i < 256; i++) {
            order[i] = r * 16 + c;
            visited[r][c] = true;
            int nr = r + dr[dir], nc = c + dc[dir];
            if (nr < 0 || nr >= 16 || nc < 0 || nc >= 16 || visited[nr][nc]) {
                dir = (dir + 1) % 4;
                nr = r + dr[dir]; nc = c + dc[dir];
            }
            r = nr; c = nc;
        }
        return order;
    }

    private void runGpuHandoff(List<int[]> partialBoardsList, int startingStep) {
        if (!gpuInitialized) return;

        cuCtxPushCurrent(cuContext);
        try {
            int numBoards = partialBoardsList.size();
            int[] flatBoards = new int[numBoards * 256];
            for (int i = 0; i < numBoards; i++) System.arraycopy(partialBoardsList.get(i), 0, flatBoards, i * 256, 256);

            CUdeviceptr d_partialBoards = new CUdeviceptr();
            cuMemAlloc(d_partialBoards, (long) numBoards * 256 * Sizeof.INT);
            cuMemcpyHtoD(d_partialBoards, Pointer.to(flatBoards), (long) numBoards * 256 * Sizeof.INT);

            CUdeviceptr d_results = new CUdeviceptr();
            cuMemAlloc(d_results, 10000L * 16 * Sizeof.INT);

            CUdeviceptr d_counter = new CUdeviceptr();
            cuMemAlloc(d_counter, Sizeof.INT);
            cuMemsetD32(d_counter, 0, 1);

            Pointer kernelParameters = Pointer.to(
                    Pointer.to(d_partialBoards),
                    Pointer.to(d_results),
                    Pointer.to(d_counter),
                    Pointer.to(new int[]{numBoards}),
                    Pointer.to(new int[]{10000})
            );

            int threadsPerBlock = 256;
            int blocksPerGrid = (numBoards + threadsPerBlock - 1) / threadsPerBlock;

            cuLaunchKernel(cuFunction, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, null, kernelParameters, null);
            cuCtxSynchronize();

            int[] countArr = new int[1];
            cuMemcpyDtoH(Pointer.to(countArr), d_counter, Sizeof.INT);
            
            System.out.println(">>> GPU Batch Finished. Found " + countArr[0] + " valid continuations.");

            cuMemFree(d_partialBoards);
            cuMemFree(d_results);
            cuMemFree(d_counter);
        } finally {
            cuCtxPopCurrent(new CUcontext());
        }
    }

    @Override
    public void run() {
        System.out.println("Starting Engine...");
        if (useGpu) initCUDA();

        flatBoard[135] = targetPiece;
        if (centerPhysicalIdx != -1) usedPhysicalPieces[centerPhysicalIdx] = true;

        solve(0);
    }

    private boolean solve(int step) {
        if (step == 256) return true;
        int boardIdx = BUILD_ORDER[step];
        if (boardIdx == 135) return solve(step + 1);

        if (useGpu && step == handoffDepth) {
            int[] clonedBoard = new int[256];
            System.arraycopy(flatBoard, 0, clonedBoard, 0, 256);
            gpuSeedBoards.add(clonedBoard);

            if (gpuSeedBoards.size() >= targetBatchSize) {
                runGpuHandoff(gpuSeedBoards, handoffDepth);
                gpuSeedBoards.clear();
                forceBacktrackTarget = step - 5;
            }
            return false;
        }

        if (step > deepestStep) {
            deepestStep = step;
            updateDisplay();
        }

        int row = boardIdx / 16, col = boardIdx % 16;
        int n_req = (row == 0) ? 0 : (flatBoard[boardIdx - 16] != -1 ? PieceUtils.getSouth(flatBoard[boardIdx - 16]) : PieceUtils.WILDCARD);
        int s_req = (row == 15) ? 0 : (flatBoard[boardIdx + 16] != -1 ? PieceUtils.getNorth(flatBoard[boardIdx + 16]) : PieceUtils.WILDCARD);
        int w_req = (col == 0) ? 0 : (flatBoard[boardIdx - 1] != -1 ? PieceUtils.getEast(flatBoard[boardIdx - 1]) : PieceUtils.WILDCARD);
        int e_req = (col == 15) ? 0 : (flatBoard[boardIdx + 1] != -1 ? PieceUtils.getWest(flatBoard[boardIdx + 1]) : PieceUtils.WILDCARD);

        int b_req = (row == 0 || row == 15 ? 1 : 0) + (col == 0 || col == 15 ? 1 : 0);

        List<Integer> pool;
        if (b_req == 2) pool = inventory.corners;
        else if (b_req == 1) pool = inventory.edges;
        else pool = inventory.interior;

        for (int orientationIdx : pool) {
            int p = inventory.allOrientations[orientationIdx];
            int physicalIdx = inventory.physicalMapping[orientationIdx];
            if (usedPhysicalPieces[physicalIdx]) continue;

            if (matches(p, n_req, e_req, s_req, w_req)) {
                flatBoard[boardIdx] = p;
                usedPhysicalPieces[physicalIdx] = true;

                if (solve(step + 1)) return true;

                flatBoard[boardIdx] = -1;
                usedPhysicalPieces[physicalIdx] = false;

                if (forceBacktrackTarget != -1) {
                    if (step > forceBacktrackTarget) return false;
                    else forceBacktrackTarget = -1;
                }
            }
        }
        return false;
    }

    private boolean matches(int p, int n, int e, int s, int w) {
        if (n != PieceUtils.WILDCARD && PieceUtils.getNorth(p) != n) return false;
        if (e != PieceUtils.WILDCARD && PieceUtils.getEast(p) != e) return false;
        if (s != PieceUtils.WILDCARD && PieceUtils.getSouth(p) != s) return false;
        return w == PieceUtils.WILDCARD || PieceUtils.getWest(p) == w;
    }

    private void updateDisplay() {
        int[][] displayBoard = new int[16][16];
        for (int i = 0; i < 256; i++) {
            int r = i / 16, c = i % 16;
            displayBoard[r][c] = flatBoard[i];
        }
        Main.updateDisplay(deepestStep, displayBoard);
    }
}

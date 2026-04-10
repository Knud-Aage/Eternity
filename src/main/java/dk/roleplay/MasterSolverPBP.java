package dk.roleplay;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class MasterSolverPBP implements Runnable {

    private final PieceInventory inventory;
    private final int[] flatBoard = new int[256];
    private final int[] bestBoard = new int[256];
    private final int[] flatResumeBoard = new int[256];
    private final boolean[] usedPhysicalPieces = new boolean[256];
    private final Random rnd = new Random();
    private final int targetPiece;
    private final boolean lockCenter;
    private final boolean useGpu;
    private final BuildStrategy currentStrategy; // Stores the chosen strategy
    private final List<int[]> gpuSeedBoards = new ArrayList<>();
    // Dynamic settings based on strategy
    private final int HANDOFF_DEPTH;
    // The active build map
    private final int[] BUILD_ORDER;
    private int centerPhysicalIdx = -1;
    private int deepestStep = 0;
    private int forceBacktrackTarget = -1;

    // Constructor updated to accept the strategy
    public MasterSolverPBP(PieceInventory inventory, int trueCenterPiece, boolean useGpu, BuildStrategy strategy,
                           boolean lockCenter) {
        this.inventory = inventory;
        this.targetPiece = trueCenterPiece;
        this.useGpu = useGpu;
        this.currentStrategy = strategy;
        this.lockCenter = lockCenter;

        Arrays.fill(flatBoard, -1);
        Arrays.fill(bestBoard, -1);
        Arrays.fill(flatResumeBoard, -1);

        for (int i = 0; i < 1024; i++) {
            if (inventory.allOrientations[i] == targetPiece) {
                this.centerPhysicalIdx = inventory.physicalMapping[i];
                break;
            }
        }

        // Apply strategy-specific settings
        if (strategy == BuildStrategy.SPIRAL) {
            System.out.println(">>> Mode: SPIRAL BUILD initialized.");
            BUILD_ORDER = generateSpiralOrder();
            HANDOFF_DEPTH = 70; // Requires full frame + inner ring start
        } else {
            System.out.println(">>> Mode: TYPEWRITER BUILD initialized.");
            BUILD_ORDER = generateTypewriterOrder();
            HANDOFF_DEPTH = 50; // Standard 3-row block
        }

        int[][] loaded = CheckpointManager.load(currentStrategy.name());
        if (loaded != null) {
            int validPieceCount = 0;
            int highestStepLoaded = -1;

            for (int step = 0; step < 256; step++) {
                int boardIdx = BUILD_ORDER[step];
                // Skip the center piece whenever the map lands on it, IF locked
                if (lockCenter && boardIdx == 135) {
                    return solve(step + 1);
                }
                int r = boardIdx / 16;
                int c = boardIdx % 16;

                if (loaded[r] != null) {
                    int p = loaded[r][c];
                    if (isValidPiece(p) && p != targetPiece) {
                        flatResumeBoard[boardIdx] = p;
                        bestBoard[boardIdx] = p;
                        validPieceCount++;
                        highestStepLoaded = Math.max(highestStepLoaded, step);
                    }
                }
            }
            if (validPieceCount > 0) {
                this.deepestStep = highestStepLoaded;
                bestBoard[135] = targetPiece;
                System.out.println(">>> PBP Checkpoint Loaded! Fast-forwarding up to step " + highestStepLoaded + ".." +
                        ".");
            }
        }
    }

    // --- MAP GENERATORS ---
    private static int[] generateTypewriterOrder() {
        int[] order = new int[256];
        for (int i = 0; i < 256; i++) {
            order[i] = i; // Just left to right, top to bottom
        }
        return order;
    }

    private static int[] generateSpiralOrder() {
        int[] order = new int[256];
        boolean[][] visited = new boolean[16][16];
        int r = 0, c = 0;
        int[] dr = {0, 1, 0, -1};
        int[] dc = {1, 0, -1, 0};
        int dir = 0;
        for (int i = 0; i < 256; i++) {
            order[i] = r * 16 + c;
            visited[r][c] = true;
            int nr = r + dr[dir];
            int nc = c + dc[dir];
            if (nr < 0 || nr >= 16 || nc < 0 || nc >= 16 || visited[nr][nc]) {
                dir = (dir + 1) % 4;
                nr = r + dr[dir];
                nc = c + dc[dir];
            }
            r = nr;
            c = nc;
        }
        return order;
    }

    private boolean isValidPiece(int p) {
        for (int i = 0; i < 1024; i++) {
            if (inventory.allOrientations[i] == p) {
                return true;
            }
        }
        return false;
    }

    private void runGpuHandoff(List<int[]> partialBoardsList, int startingStep) {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        CUmodule module = new CUmodule();
        cuModuleLoad(module, "SolveEternityKernel.ptx");
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "solvePBP");

        int numBoards = partialBoardsList.size();
        int[] flatBoards = new int[numBoards * 256];
        for (int i = 0; i < numBoards; i++) System.arraycopy(partialBoardsList.get(i), 0, flatBoards, i * 256, 256);

        CUdeviceptr d_partialBoards = new CUdeviceptr();
        cuMemAlloc(d_partialBoards, (long) numBoards * 256 * Sizeof.INT);
        cuMemcpyHtoD(d_partialBoards, Pointer.to(flatBoards), (long) numBoards * 256 * Sizeof.INT);

        // Send the active map to the GPU
        CUdeviceptr d_buildOrder = new CUdeviceptr();
        cuMemAlloc(d_buildOrder, 256L * Sizeof.INT);
        cuMemcpyHtoD(d_buildOrder, Pointer.to(BUILD_ORDER), 256L * Sizeof.INT);

        CUdeviceptr d_allOrientations = new CUdeviceptr();
        cuMemAlloc(d_allOrientations, 1024L * Sizeof.INT);
        cuMemcpyHtoD(d_allOrientations, Pointer.to(inventory.allOrientations), 1024L * Sizeof.INT);

        CUdeviceptr d_physicalMapping = new CUdeviceptr();
        cuMemAlloc(d_physicalMapping, 1024L * Sizeof.INT);
        cuMemcpyHtoD(d_physicalMapping, Pointer.to(inventory.physicalMapping), 1024L * Sizeof.INT);

        CUdeviceptr d_solution = new CUdeviceptr();
        cuMemAlloc(d_solution, 256L * Sizeof.INT);

        int[] solvedFlag = {0};
        CUdeviceptr d_solvedFlag = new CUdeviceptr();
        cuMemAlloc(d_solvedFlag, Sizeof.INT);
        cuMemcpyHtoD(d_solvedFlag, Pointer.to(solvedFlag), Sizeof.INT);

        int[] gpuScore = {0};
        CUdeviceptr d_gpuHighScore = new CUdeviceptr();
        cuMemAlloc(d_gpuHighScore, Sizeof.INT);
        cuMemcpyHtoD(d_gpuHighScore, Pointer.to(gpuScore), Sizeof.INT);

        CUdeviceptr d_bestBoardOut = new CUdeviceptr();
        cuMemAlloc(d_bestBoardOut, 256L * Sizeof.INT);

        long[] totalSteps = {0L};
        CUdeviceptr d_totalSteps = new CUdeviceptr();
        cuMemAlloc(d_totalSteps, 8L);
        cuMemcpyHtoD(d_totalSteps, Pointer.to(totalSteps), 8L);

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
                Pointer.to(new int[]{lockCenter ? 1 : 0})
        );

        int threadsPerBlock = 256;
        int blocksPerGrid = (int) Math.ceil((double) numBoards / threadsPerBlock);

        long startTime = System.currentTimeMillis();
        cuLaunchKernel(function, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, null, kernelParameters, null);
        cuCtxSynchronize();
        long timeTaken = System.currentTimeMillis() - startTime;

        cuMemcpyDtoH(Pointer.to(totalSteps), d_totalSteps, 8L);
        double timeSeconds = Math.max(timeTaken / 1000.0, 0.001);
        long speed = (long) (totalSteps[0] / timeSeconds);

        System.out.printf("%s: GPU Batch Finished in %d ms! Speed: %,d pieces/sec (Total: %,d)%n",
                LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")), timeTaken, speed,
                totalSteps[0]);

        cuMemcpyDtoH(Pointer.to(solvedFlag), d_solvedFlag, Sizeof.INT);
        if (solvedFlag[0] == 1) {
            System.out.println(">>> GPU FOUND THE SOLUTION! <<<");
            int[] winningBoard = new int[256];
            cuMemcpyDtoH(Pointer.to(winningBoard), d_solution, 256L * Sizeof.INT);
            updateDisplay(buildDisplayBoard(winningBoard));
            RecordManager.saveRecord(buildDisplayBoard(winningBoard), 256, currentStrategy.name());
            System.exit(0);
        }

        cuMemcpyDtoH(Pointer.to(gpuScore), d_gpuHighScore, Sizeof.INT);

        if (gpuScore[0] > deepestStep) {
            deepestStep = gpuScore[0];
            int[] gpuWinningBoard = new int[256];
            cuMemcpyDtoH(Pointer.to(gpuWinningBoard), d_bestBoardOut, 256L * Sizeof.INT);
            System.arraycopy(gpuWinningBoard, 0, bestBoard, 0, 256);

            int[][] displayBoard = buildDisplayBoard(bestBoard);
            updateDisplay(displayBoard);
            RecordManager.saveRecord(displayBoard, deepestStep, currentStrategy.name());
            CheckpointManager.save(displayBoard, currentStrategy.name());
        }

        cuMemFree(d_partialBoards);
        cuMemFree(d_buildOrder);
        cuMemFree(d_allOrientations);
        cuMemFree(d_physicalMapping);
        cuMemFree(d_solution);
        cuMemFree(d_solvedFlag);
        cuMemFree(d_gpuHighScore);
        cuMemFree(d_bestBoardOut);
        cuMemFree(d_totalSteps);
    }

    @Override
    public void run() {
        System.out.println("Starting Engine...");

        if (lockCenter) {
            flatBoard[135] = targetPiece;
            if (centerPhysicalIdx != -1) {
                usedPhysicalPieces[centerPhysicalIdx] = true;
            }
        }

        flatBoard[135] = targetPiece;
        if (centerPhysicalIdx != -1) {
            usedPhysicalPieces[centerPhysicalIdx] = true;
        }

        if (deepestStep > 0) {
            updateDisplay(buildDisplayBoard(bestBoard));
        }

        if (solve(0)) {
            System.out.println("SOLVED!");
        } else {
            System.out.println("Exhausted search space. No solution found.");
        }
    }

    private boolean solve(int step) {
        if (step == 256) {
            return true;
        }

        int boardIdx = BUILD_ORDER[step];

        // Skip the center piece whenever the map lands on it
        if (boardIdx == 135) {
            return solve(step + 1);
        }

        if (useGpu && step == HANDOFF_DEPTH) {
            int[] clonedBoard = new int[256];
            System.arraycopy(flatBoard, 0, clonedBoard, 0, 256);
            gpuSeedBoards.add(clonedBoard);

            if (gpuSeedBoards.size() >= 10000) {
                runGpuHandoff(gpuSeedBoards, HANDOFF_DEPTH);
                gpuSeedBoards.clear();

                // Strategy-specific backtracking logic
                if (currentStrategy == BuildStrategy.SPIRAL) {
                    forceBacktrackTarget = 60 + rnd.nextInt(5); // Keep the frame
                } else {
                    forceBacktrackTarget = 10 + rnd.nextInt(11); // Standard backtrack
                }
            }
            return false;
        }

        if (step > deepestStep) {
            deepestStep = step;
            System.arraycopy(flatBoard, 0, bestBoard, 0, 256);
            int[][] displayBoard = buildDisplayBoard(bestBoard);
            updateDisplay(displayBoard);
            RecordManager.saveRecord(displayBoard, deepestStep, currentStrategy.name());
            CheckpointManager.save(displayBoard, currentStrategy.name());
        }

        int row = boardIdx / 16;
        int col = boardIdx % 16;

        int n_req = (row == 0) ? 0 : (flatBoard[boardIdx - 16] != -1 ? PieceUtils.getSouth(flatBoard[boardIdx - 16])
                                      : PieceUtils.WILDCARD);
        int s_req = (row == 15) ? 0 : (flatBoard[boardIdx + 16] != -1 ?
                PieceUtils.getNorth(flatBoard[boardIdx + 16]) : PieceUtils.WILDCARD);
        int w_req = (col == 0) ? 0 : (flatBoard[boardIdx - 1] != -1 ? PieceUtils.getEast(flatBoard[boardIdx - 1]) :
                                      PieceUtils.WILDCARD);
        int e_req = (col == 15) ? 0 : (flatBoard[boardIdx + 1] != -1 ? PieceUtils.getWest(flatBoard[boardIdx + 1]) :
                                       PieceUtils.WILDCARD);

        int b_req = 0;
        if (row == 0 || row == 15) {
            b_req++;
        }
        if (col == 0 || col == 15) {
            b_req++;
        }

        List<Integer> pool;
        if (b_req == 2) {
            pool = inventory.corners;
        } else if (b_req == 1) {
            pool = inventory.edges;
        } else {
            pool = inventory.interior;
        }

        if (pool.isEmpty()) {
            return false;
        }

        int size = pool.size();
        int offset = rnd.nextInt(size);

        for (int i = 0; i < size; i++) {
            int orientationIdx = pool.get((i + offset) % size);
            int p = inventory.allOrientations[orientationIdx];
            int physicalIdx = inventory.physicalMapping[orientationIdx];

            if (flatResumeBoard[boardIdx] != -1) {
                if (p != flatResumeBoard[boardIdx]) {
                    continue;
                }
            }

            if (usedPhysicalPieces[physicalIdx]) {
                continue;
            }

            if (matches(p, n_req, e_req, s_req, w_req)) {

                flatBoard[boardIdx] = p;
                usedPhysicalPieces[physicalIdx] = true;

                int ghostPiece = flatResumeBoard[boardIdx];
                flatResumeBoard[boardIdx] = -1;

                if (solve(step + 1)) {
                    return true;
                }

                flatBoard[boardIdx] = -1;
                usedPhysicalPieces[physicalIdx] = false;

                if (forceBacktrackTarget != -1) {
                    if (step > forceBacktrackTarget) {
                        return false;
                    } else if (step == forceBacktrackTarget) {
                        forceBacktrackTarget = -1;
                    }
                }
            }
        }
        return false;
    }

    private boolean matches(int p, int n, int e, int s, int w) {
        if (n != PieceUtils.WILDCARD && PieceUtils.getNorth(p) != n) {
            return false;
        }
        if (e != PieceUtils.WILDCARD && PieceUtils.getEast(p) != e) {
            return false;
        }
        if (s != PieceUtils.WILDCARD && PieceUtils.getSouth(p) != s) {
            return false;
        }
        return w == PieceUtils.WILDCARD || PieceUtils.getWest(p) == w;
    }

    private int[][] buildDisplayBoard(int[] sourceArray) {
        int[][] displayBoard = new int[16][16];
        for (int i = 0; i < 16; i++) Arrays.fill(displayBoard[i], -1);
        for (int i = 0; i < 256; i++) {
            if (sourceArray[i] == -1) {
                continue;
            }
            int r = i / 16;
            int c = i % 16;
            displayBoard[r][c] = sourceArray[i];
        }
        return displayBoard;
    }

    private void updateDisplay(int[][] displayBoard) {
        Main.updateDisplay(deepestStep, displayBoard);
    }

    // --- NEW: Strategy Enum ---
    public enum BuildStrategy {
        TYPEWRITER,
        SPIRAL
    }
}
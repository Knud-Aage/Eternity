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

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

public class MasterSolverPBP implements Runnable {

    private final PieceInventory inventory;
    private final int[] flatBoard = new int[256];
    private final int[] bestBoard = new int[256];
    private final int[] flatResumeBoard = new int[256];
    private final boolean[] usedPhysicalPieces = new boolean[256];
    private final Object displayLock = new Object();
    private final int targetPiece;
    private final boolean useGpu;
    private final BuildStrategy currentStrategy;
    private final boolean lockCenter;
    private final String saveProfile;
    private final ExecutorService executor;
    private final int numCores;
    private final ConcurrentLinkedQueue<int[]> gpuSeedBoards = new ConcurrentLinkedQueue<>();
    private final AtomicInteger currentBatchSize = new AtomicInteger(0);
    private final int[] buildOrder;
    private CUfunction cuFunction;
    private int centerPhysicalIdx = -1;
    private volatile int deepestStep = 0;
    private volatile int absoluteHighScore = 0;
    private int handoffDepth;
    private int targetBatchSize = 10000;
    private volatile double extinctionThreshold = 0.98;
    private volatile boolean manualOverrideRequested = false;
    private volatile int manualBaseCampTarget = 0;
    private volatile int userBatchSizeOverride = -1;

    public MasterSolverPBP(PieceInventory inventory, int trueCenterPiece, boolean useGpu, BuildStrategy strategy,
                           boolean lockCenter) {
        Arrays.fill(flatBoard, -1);
        Arrays.fill(bestBoard, -1);
        Arrays.fill(flatResumeBoard, -1);

        this.inventory = inventory;
        this.targetPiece = trueCenterPiece;
        this.useGpu = useGpu;
        this.currentStrategy = strategy;
        this.lockCenter = lockCenter;

        this.saveProfile = strategy.name() + (lockCenter ? "_LOCKED" : "_UNLOCKED");

        // Start CPU Thread Pool
        this.numCores = Runtime.getRuntime().availableProcessors();
        this.executor = Executors.newFixedThreadPool(numCores);
        System.out.println(timestamp() + ">>> Multithreading activity with " + numCores + " dedicated CPU-kernels.");

        for (int i = 0; i < 1024; i++) {
            if (inventory.allOrientations[i] == targetPiece) {
                this.centerPhysicalIdx = inventory.physicalMapping[i];
                break;
            }
        }

        if (strategy == BuildStrategy.SPIRAL) {
            buildOrder = generateSpiralOrder();
            handoffDepth = 70;
        } else {
            buildOrder = generateTypewriterOrder();
            handoffDepth = 50;
        }

        int[][] loaded = CheckpointManager.load(saveProfile);
        if (loaded != null) {
            int highestStepLoaded = -1;
            for (int step = 0; step < 256; step++) {
                int boardIdx = buildOrder[step];
                int r = boardIdx / 16;
                int c = boardIdx % 16;

                if (loaded[r] != null) {
                    int p = loaded[r][c];
                    if (isValidPiece(p) && p != targetPiece) {
                        bestBoard[boardIdx] = p;
                        highestStepLoaded = Math.max(highestStepLoaded, step);
                    }
                }
            }
            if (highestStepLoaded > 0) {
                this.deepestStep = highestStepLoaded;
                this.absoluteHighScore = highestStepLoaded;
                if (lockCenter) {
                    bestBoard[135] = targetPiece;
                }
            }
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

    private String timestamp() {
        return "[" + LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss")) + "] ";
    }

    public void setExtinctionThreshold(double threshold) {
        this.extinctionThreshold = threshold;
    }

    public void triggerManualOverride(int targetBaseCamp) {
        this.manualBaseCampTarget = targetBaseCamp;
        this.manualOverrideRequested = true;
    }

    public void setBatchSizeOverride(int size) {
        this.userBatchSizeOverride = size;
    }

    private void initCUDA() {
        CUcontext cuContext;
        CUmodule cuModule;
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);

        cuContext = new CUcontext();
        cuCtxCreate(cuContext, 0, device);

        cuModule = new CUmodule();
        cuModuleLoad(cuModule, "SolveEternityKernel.ptx");

        cuFunction = new CUfunction();
        cuModuleGetFunction(cuFunction, cuModule, "solvePBP");
        System.out.println(timestamp() + ">>> CUDA Context & Kernel read successfully!");
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
        int numBoards = partialBoardsList.size();
        if (numBoards == 0) {
            return;
        }

        int[] flatBoards = new int[numBoards * 256];
        for (int i = 0; i < numBoards; i++) System.arraycopy(partialBoardsList.get(i), 0, flatBoards, i * 256, 256);

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

        int[] threadDepths = new int[numBoards];
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
                Pointer.to(d_threadDepths)
        );

        int threadsPerBlock = 256;
        int blocksPerGrid = (int) Math.ceil((double) numBoards / threadsPerBlock);

        long startTime = System.currentTimeMillis();
        cuLaunchKernel(cuFunction, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, null, kernelParameters, null);
        cuCtxSynchronize();
        long timeTaken = System.currentTimeMillis() - startTime;

        cuMemcpyDtoH(Pointer.to(totalSteps), d_totalSteps, 8L);
        cuMemcpyDtoH(Pointer.to(threadDepths), d_threadDepths, (long) numBoards * Sizeof.INT);

        double timeSeconds = Math.max(timeTaken / 1000.0, 0.001);
        double speed = totalSteps[0] / timeSeconds;

        long sumDepth = 0;
        int maxInBatch = 0;
        int deadOnArrival = 0;
        for (int d : threadDepths) {
            sumDepth += d;
            if (d > maxInBatch) {
                maxInBatch = d;
            }
            if (d <= startingStep + 5) {
                deadOnArrival++;
            }
        }

        System.out.printf("%sGPU Batch | %d ms | %,.0f pcs/s | Avg Depth: %.1f | Max: %d | Dead < 5: %d/%d\n",
                timestamp(), timeTaken, speed, (double) sumDepth / numBoards, maxInBatch, deadOnArrival, numBoards);

        if (numBoards > 0 && (double) deadOnArrival / numBoards >= extinctionThreshold) {
            cuFreeResources(d_partialBoards, d_buildOrder, d_allOrientations, d_physicalMapping, d_solution,
                    d_solvedFlag, d_gpuHighScore, d_bestBoardOut, d_totalSteps, d_threadDepths);
            throw new PoisonedBaseCampException();
        }

        cuMemcpyDtoH(Pointer.to(solvedFlag), d_solvedFlag, Sizeof.INT);
        if (solvedFlag[0] == 1) {
            System.out.println(timestamp() + ">>> GPU FOUND THE SOLUTION! <<<");
            int[] winningBoard = new int[256];
            cuMemcpyDtoH(Pointer.to(winningBoard), d_solution, 256L * Sizeof.INT);
            updateDisplay(buildDisplayBoard(winningBoard));
            RecordManager.saveRecord(buildDisplayBoard(winningBoard), 256, saveProfile);
            System.exit(0);
        }

        cuMemcpyDtoH(Pointer.to(gpuScore), d_gpuHighScore, Sizeof.INT);

        if (gpuScore[0] > deepestStep) {
            synchronized (displayLock) {
                if (gpuScore[0] > deepestStep) {
                    deepestStep = gpuScore[0];
                    int[] gpuWinningBoard = new int[256];
                    cuMemcpyDtoH(Pointer.to(gpuWinningBoard), d_bestBoardOut, 256L * Sizeof.INT);
                    System.arraycopy(gpuWinningBoard, 0, bestBoard, 0, 256);

                    int[][] displayBoard = buildDisplayBoard(bestBoard);
                    updateDisplay(displayBoard);

                    if (deepestStep > absoluteHighScore) {
                        absoluteHighScore = deepestStep;
                        RecordManager.saveRecord(displayBoard, absoluteHighScore, saveProfile);
                        CheckpointManager.save(displayBoard, saveProfile);
                        System.out.println(timestamp() + ">>> NY ALL-TIME HIGH SCORE (GPU): " + absoluteHighScore +
                                " PIECES! <<<");
                    }
                }
            }

            if (currentStrategy == BuildStrategy.SPIRAL && deepestStep > handoffDepth + 30) {
                cuFreeResources(d_partialBoards, d_buildOrder, d_allOrientations, d_physicalMapping, d_solution,
                        d_solvedFlag, d_gpuHighScore, d_bestBoardOut, d_totalSteps, d_threadDepths);
                throw new EvolutionLeapException();
            }
        }

        cuFreeResources(d_partialBoards, d_buildOrder, d_allOrientations, d_physicalMapping, d_solution, d_solvedFlag
                , d_gpuHighScore, d_bestBoardOut, d_totalSteps, d_threadDepths);
    }

    private void cuFreeResources(CUdeviceptr... pointers) {
        for (CUdeviceptr ptr : pointers) {
            cuMemFree(ptr);
        }
    }

    @Override
    public void run() {
        System.out.println(timestamp() + "Starting Multithreaded Engine (Profile: " + saveProfile + ")...");

        if (useGpu) {
            initCUDA();
        }

        while (true) {
            Arrays.fill(flatBoard, -1);
            Arrays.fill(flatResumeBoard, -1);
            Arrays.fill(usedPhysicalPieces, false);

            if (deepestStep > 0) {
                updateDisplay(buildDisplayBoard(bestBoard));
            }

            int lockedPieces = 0;
            if (deepestStep > 0 && currentStrategy == BuildStrategy.SPIRAL) {
                lockedPieces = Math.max(0, deepestStep - 30);

                int gap;
                if (lockedPieces > 180) {
                    gap = 8;
                    targetBatchSize = 5000;
                } else if (lockedPieces > 150) {
                    gap = 10;
                    targetBatchSize = 10000;
                } else {
                    gap = 15;
                    targetBatchSize = 10000;
                }

                handoffDepth = lockedPieces + gap;
                int activeBatch = (userBatchSizeOverride > 0) ? userBatchSizeOverride : targetBatchSize;

//                System.out.println("\n" + timestamp() + "=======================================================");
//                System.out.println(timestamp() + ">>> EVOLUTION! Base Camp locked to step " + lockedPieces);
//                System.out.println(timestamp() + ">>> CPU build to Handoff at step " + HANDOFF_DEPTH);
//                System.out.println(timestamp() + ">>> Goal: " + activeBatch + " seeds (" + numCores + " kernels
//                work parallel)");
//                System.out.println(timestamp() + "=======================================================\n");

                for (int step = 0; step < lockedPieces; step++) {
                    int boardIdx = buildOrder[step];
                    if (lockCenter && boardIdx == 135) {
                        continue;
                    }
                    flatResumeBoard[boardIdx] = bestBoard[boardIdx];
                }
            }

            gpuSeedBoards.clear();
            currentBatchSize.set(0);
            int activeBatch = (userBatchSizeOverride > 0) ? userBatchSizeOverride : targetBatchSize;

            List<SearchWorker> workers = new ArrayList<>();
            for (int i = 0; i < numCores; i++) {
                workers.add(new SearchWorker(activeBatch));
            }

            boolean spaceExhausted = true;
            try {
                List<Future<Boolean>> results = executor.invokeAll(workers);
                for (Future<Boolean> f : results) {
                    if (f.get()) {
                        System.out.println(timestamp() + "SOLVED BY CPU!");
                        return;
                    }
                }

                if (currentBatchSize.get() >= activeBatch) {
                    runGpuHandoff(new ArrayList<>(gpuSeedBoards), handoffDepth);
                    spaceExhausted = false;
                } else if (currentBatchSize.get() > 0) {
                    System.out.println(timestamp() + ">>> Exhaustion! Send " + currentBatchSize.get() + " seeds to " +
                            "GPU...");
                    runGpuHandoff(new ArrayList<>(gpuSeedBoards), handoffDepth);
                }

            } catch (EvolutionLeapException e) {
                spaceExhausted = false;
            } catch (PoisonedBaseCampException e) {
                spaceExhausted = false;
                if (lockedPieces > 70) {
                    deepestStep = Math.max(70, lockedPieces - 8);
                    System.out.println("\n" + timestamp() + "[!] EXTINCTION EVENT! " + (int) (extinctionThreshold * 100) + "%+ seeds died instantly.");
                    System.out.println(timestamp() + "[!] Poisoned Base Camp detected. Retreating to step " + deepestStep + "...\n");
                }
            } catch (ExecutionException e) {
                Throwable cause = e.getCause();
                if (cause instanceof ManualOverrideException) {
                    spaceExhausted = false;
                    manualOverrideRequested = false;
                    deepestStep = manualBaseCampTarget + 30;
                    System.out.println("\n" + timestamp() + "[!] MANUAL OVERRIDE! User forced base camp jump to step "
                            + manualBaseCampTarget + ".\n");
                } else {
                    e.printStackTrace();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }

            if (spaceExhausted) {
                if (currentStrategy == BuildStrategy.SPIRAL && lockedPieces > 0) {
                    deepestStep = Math.max(0, lockedPieces - 10);
                    System.out.println(timestamp() + ">>> Base camp exhausted. Falling back to search new path...");
                } else {
                    System.out.println(timestamp() + "Total search space exhausted. No solution found.");
                    return;
                }
            }
        }
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
        Main.updateDisplay(absoluteHighScore, displayBoard);
    }


    public enum BuildStrategy {
        TYPEWRITER, SPIRAL
    }

    private static class EvolutionLeapException extends RuntimeException {
    }

    private static class PoisonedBaseCampException extends RuntimeException {
    }

    private static class ManualOverrideException extends RuntimeException {
    }

    private class SearchWorker implements Callable<Boolean> {
        private final int[] localBoard = new int[256];
        private final int[] localResumeBoard = new int[256];
        private final boolean[] localUsed = new boolean[256];
        private final Random rnd = new Random();
        private final int activeBatch;

        public SearchWorker(int activeBatch) {
            this.activeBatch = activeBatch;
            System.arraycopy(flatBoard, 0, localBoard, 0, 256);
            System.arraycopy(flatResumeBoard, 0, localResumeBoard, 0, 256);
            System.arraycopy(usedPhysicalPieces, 0, localUsed, 0, 256);

            if (lockCenter) {
                localBoard[135] = targetPiece;
                if (centerPhysicalIdx != -1) {
                    localUsed[centerPhysicalIdx] = true;
                }
            }
        }

        @Override
        public Boolean call() {
            return solve(0);
        }

        private boolean solve(int step) {
            if (manualOverrideRequested) {
                throw new ManualOverrideException();
            }

            if (currentBatchSize.get() >= activeBatch) {
                return false;
            }

            if (step == 256) {
                return true;
            }

            int boardIdx = buildOrder[step];

            if (lockCenter && boardIdx == 135) {
                return solve(step + 1);
            }

            if (useGpu && step == handoffDepth) {
                if (currentBatchSize.get() >= activeBatch) {
                    return false;
                }

                int[] clonedBoard = new int[256];
                System.arraycopy(localBoard, 0, clonedBoard, 0, 256);
                gpuSeedBoards.add(clonedBoard);

                int size = currentBatchSize.incrementAndGet();
                int printInterval = (activeBatch <= 250) ? 10 : (activeBatch <= 1000 ? 100 : 1000);

//                if (size % printInterval == 0) {
//                    System.out.println(timestamp() + "   [CPU Pool] Found " + size + " / " + activeBatch + " seeds.
//                    ..");
//                }
                return false;
            }

            if (step > deepestStep) {
                synchronized (displayLock) {
                    if (step > deepestStep) {
                        deepestStep = step;
                        System.arraycopy(localBoard, 0, bestBoard, 0, 256);
                        int[][] displayBoard = buildDisplayBoard(bestBoard);
                        updateDisplay(displayBoard);

                        if (deepestStep > absoluteHighScore) {
                            absoluteHighScore = deepestStep;
                            RecordManager.saveRecord(displayBoard, absoluteHighScore, saveProfile);
                            CheckpointManager.save(displayBoard, saveProfile);
                            System.out.println(timestamp() + ">>> NY ALL-TIME HIGH SCORE: " + absoluteHighScore + " " +
                                    "PIECES! <<<");
                        }
                    }
                }
            }

            int row = boardIdx / 16;
            int col = boardIdx % 16;

            int n_req = (row == 0) ? 0 : (localBoard[boardIdx - 16] != -1 ?
                    PieceUtils.getSouth(localBoard[boardIdx - 16]) : PieceUtils.WILDCARD);
            int s_req = (row == 15) ? 0 : (localBoard[boardIdx + 16] != -1 ?
                    PieceUtils.getNorth(localBoard[boardIdx + 16]) : PieceUtils.WILDCARD);
            int w_req = (col == 0) ? 0 : (localBoard[boardIdx - 1] != -1 ?
                    PieceUtils.getEast(localBoard[boardIdx - 1]) : PieceUtils.WILDCARD);
            int e_req = (col == 15) ? 0 : (localBoard[boardIdx + 1] != -1 ?
                    PieceUtils.getWest(localBoard[boardIdx + 1]) : PieceUtils.WILDCARD);

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
                if (currentBatchSize.get() >= activeBatch) {
                    return false;
                }

                int orientationIdx = pool.get((i + offset) % size);
                int p = inventory.allOrientations[orientationIdx];
                int physicalIdx = inventory.physicalMapping[orientationIdx];

                if (localResumeBoard[boardIdx] != -1) {
                    if (p != localResumeBoard[boardIdx]) {
                        continue;
                    }
                }

                if (localUsed[physicalIdx]) {
                    continue;
                }

                if (matches(p, n_req, e_req, s_req, w_req)) {

                    localBoard[boardIdx] = p;
                    localUsed[physicalIdx] = true;

                    int ghostPiece = localResumeBoard[boardIdx];
                    localResumeBoard[boardIdx] = -1;

                    if (solve(step + 1)) {
                        return true;
                    }

                    localBoard[boardIdx] = -1;
                    localUsed[physicalIdx] = false;
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
    }
}
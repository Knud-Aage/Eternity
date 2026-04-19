package dk.puzzle;

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
    private final java.util.Set<Long> structuralDiversityFilter = java.util.concurrent.ConcurrentHashMap.newKeySet();
    private final int[] buildOrder;
    private final int[][] piecesByNorth = new int[256][];
    private final int[][] piecesByEast = new int[256][];
    private final int[][] piecesBySouth = new int[256][];
    private final int[][] piecesByWest = new int[256][];
    private final int[] recordBoard = new int[256];
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
    private long lastProgressTimestamp = System.currentTimeMillis();
    private volatile int stagnationLimitMinutes = 20;
    private int consecutiveExtinctions = 0;
    private int consecutiveExhaustions = 0;

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
                    // Trust the checkpoint file; if it's not -1/0, it's a piece
                    if (p != -1 && p != 0) {
                        bestBoard[boardIdx] = p;
                        highestStepLoaded = step; // Track the build-order depth
                    }
                }
            }
            if (highestStepLoaded >= 0) {
                this.deepestStep = highestStepLoaded;
                this.absoluteHighScore = highestStepLoaded;
                if (lockCenter) {
                    bestBoard[135] = targetPiece;
                }
                // Visual fix: Show the loaded highscore immediately on startup
                System.arraycopy(bestBoard, 0, flatBoard, 0, 256);
                System.arraycopy(bestBoard, 0, flatResumeBoard, 0, 256);
                updateDisplay(absoluteHighScore, buildDisplayBoard(bestBoard));
                System.out.println(timestamp() + ">>> Loaded checkpoint: " + (highestStepLoaded + 1) + " pieces.");
            }
        }
        java.util.Set<Integer>[] tempNorth = new java.util.HashSet[256];
        java.util.Set<Integer>[] tempEast = new java.util.HashSet[256];
        java.util.Set<Integer>[] tempSouth = new java.util.HashSet[256];
        java.util.Set<Integer>[] tempWest = new java.util.HashSet[256];

        for (int i = 0; i < 256; i++) {
            tempNorth[i] = new java.util.HashSet<>();
            tempEast[i] = new java.util.HashSet<>();
            tempSouth[i] = new java.util.HashSet<>();
            tempWest[i] = new java.util.HashSet<>();
        }

        for (int i = 0; i < 1024; i++) {
            int p = inventory.allOrientations[i];
            int physId = inventory.physicalMapping[i];

            tempNorth[PieceUtils.getNorth(p)].add(physId);
            tempEast[PieceUtils.getEast(p)].add(physId);
            tempSouth[PieceUtils.getSouth(p)].add(physId);
            tempWest[PieceUtils.getWest(p)].add(physId);
        }

        // Convert to fast native int arrays
        for (int i = 0; i < 256; i++) {
            piecesByNorth[i] = tempNorth[i].stream().mapToInt(Integer::intValue).toArray();
            piecesByEast[i] = tempEast[i].stream().mapToInt(Integer::intValue).toArray();
            piecesBySouth[i] = tempSouth[i].stream().mapToInt(Integer::intValue).toArray();
            piecesByWest[i] = tempWest[i].stream().mapToInt(Integer::intValue).toArray();
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

    public void setStagnationLimit(int minutes) {
        this.stagnationLimitMinutes = minutes;
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

    private void runGpuHandoff(List<int[]> partialBoardsList, int startingStep, int radarLimit) {
        int numBoards = partialBoardsList.size();
        if (numBoards == 0) return;

        List<CUdeviceptr> gpuPointers = new ArrayList<>();
        try {
            // 1. Setup and Upload
            int[] flatBoards = new int[numBoards * 256];
            for (int i = 0; i < numBoards; i++) System.arraycopy(partialBoardsList.get(i), 0, flatBoards, i * 256, 256);

            CUdeviceptr d_partialBoards = gpuAllocAndUpload(flatBoards, gpuPointers);
            CUdeviceptr d_buildOrder = gpuAllocAndUpload(buildOrder, gpuPointers);
            CUdeviceptr d_allOrientations = gpuAllocAndUpload(inventory.allOrientations, gpuPointers);
            CUdeviceptr d_physicalMapping = gpuAllocAndUpload(inventory.physicalMapping, gpuPointers);
            CUdeviceptr d_solution = gpuAlloc(256 * Sizeof.INT, gpuPointers);
            CUdeviceptr d_solvedFlag = gpuAllocAndUpload(new int[]{0}, gpuPointers);
            CUdeviceptr d_gpuHighScore = gpuAllocAndUpload(new int[]{0}, gpuPointers);
            CUdeviceptr d_bestBoardOut = gpuAlloc(256 * Sizeof.INT, gpuPointers);
            CUdeviceptr d_totalSteps = gpuAlloc(8, gpuPointers);
            cuMemcpyHtoD(d_totalSteps, Pointer.to(new long[]{0L}), 8L);
            CUdeviceptr d_threadDepths = gpuAlloc(numBoards * Sizeof.INT, gpuPointers);
            CUdeviceptr d_radarLimit = gpuAllocAndUpload(new int[]{radarLimit}, gpuPointers);

            // 2. Kernel Launch
            Pointer kernelParams = Pointer.to(
                    Pointer.to(d_partialBoards), Pointer.to(new int[]{numBoards}), Pointer.to(new int[]{startingStep}),
                    Pointer.to(d_buildOrder), Pointer.to(d_allOrientations), Pointer.to(d_physicalMapping),
                    Pointer.to(d_solution), Pointer.to(d_solvedFlag), Pointer.to(d_gpuHighScore),
                    Pointer.to(d_bestBoardOut), Pointer.to(d_totalSteps), Pointer.to(new int[]{lockCenter ? 1 : 0}),
                    Pointer.to(d_threadDepths), Pointer.to(d_radarLimit)
            );

            long startTime = System.currentTimeMillis();
            cuLaunchKernel(cuFunction, (int)Math.ceil(numBoards/256.0), 1, 1, 256, 1, 1, 0, null, kernelParams, null);
            cuCtxSynchronize();
            long timeTaken = System.currentTimeMillis() - startTime;

            // 3. Collect Results
            long[] totalSteps = {0L};
            int[] threadDepths = new int[numBoards];
            int[] solvedFlag = {0}, gpuScore = {0};

            cuMemcpyDtoH(Pointer.to(totalSteps), d_totalSteps, 8L);
            cuMemcpyDtoH(Pointer.to(threadDepths), d_threadDepths, (long) numBoards * Sizeof.INT);
            cuMemcpyDtoH(Pointer.to(solvedFlag), d_solvedFlag, Sizeof.INT);
            cuMemcpyDtoH(Pointer.to(gpuScore), d_gpuHighScore, Sizeof.INT);

            // 4. Reporting and Progress
            processGpuResults(timeTaken, totalSteps[0], threadDepths, startingStep, numBoards);

            if (solvedFlag[0] == 1) {
                int[] winningBoard = new int[256];
                cuMemcpyDtoH(Pointer.to(winningBoard), d_solution, 256L * Sizeof.INT);
                updateDisplay(256, buildDisplayBoard(winningBoard));
                RecordManager.saveRecord(buildDisplayBoard(winningBoard), 256, saveProfile);
                System.exit(0);
            }

            if (gpuScore[0] > deepestStep) {
                synchronized (displayLock) {
                    if (gpuScore[0] > deepestStep) {
                        deepestStep = gpuScore[0];
                        int[] gpuWinningBoard = new int[256];
                        cuMemcpyDtoH(Pointer.to(gpuWinningBoard), d_bestBoardOut, 256L * Sizeof.INT);
                        System.arraycopy(gpuWinningBoard, 0, bestBoard, 0, 256);
                        updateDisplay(deepestStep, buildDisplayBoard(bestBoard));

                        if (deepestStep > absoluteHighScore) {
                            absoluteHighScore = deepestStep;
                            lastProgressTimestamp = System.currentTimeMillis();
                            RecordManager.saveRecord(buildDisplayBoard(bestBoard), absoluteHighScore, saveProfile);
                            CheckpointManager.save(buildDisplayBoard(bestBoard), saveProfile);
                            System.out.println(timestamp() + ">>> NY ALL-TIME HIGH SCORE (GPU): " + absoluteHighScore + " PIECES! <<<");
                        }
                    }
                }
                if (deepestStep > handoffDepth + 30) throw new EvolutionLeapException();
            }
        } finally {
            gpuPointers.forEach(JCudaDriver::cuMemFree);
        }
    }

    private CUdeviceptr gpuAlloc(long size, List<CUdeviceptr> tracker) {
        CUdeviceptr ptr = new CUdeviceptr();
        cuMemAlloc(ptr, size);
        tracker.add(ptr);
        return ptr;
    }

    private CUdeviceptr gpuAllocAndUpload(int[] data, List<CUdeviceptr> tracker) {
        CUdeviceptr ptr = gpuAlloc((long) data.length * Sizeof.INT, tracker);
        cuMemcpyHtoD(ptr, Pointer.to(data), (long) data.length * Sizeof.INT);
        return ptr;
    }

    private void processGpuResults(long timeMs, long steps, int[] depths, int start, int n) {
        double speed = steps / Math.max(timeMs/1000.0, 0.001);
        long sum = 0; int max = 0, dead = 0;
        for (int d : depths) {
            sum += d;
            if (d > max) max = d;
            if (d <= start + 5) dead++;
        }
        System.out.printf("%sGPU | %d ms | %,.0f pcs/s | Avg Depth: %.1f | Max: %d | Dead < 5: %d/%d\n",
                timestamp(), timeMs, speed, (double)sum/n, max, dead, n);
        if (n > 0 && (double)dead/n >= extinctionThreshold) throw new PoisonedBaseCampException();
    }

    private void cuFreeResources(CUdeviceptr... pointers) {
        for (CUdeviceptr ptr : pointers) {
            cuMemFree(ptr);
        }
    }

    @Override
    public void run() {
        System.out.println(timestamp() + "Starting Multithreaded Engine (Profile: " + saveProfile + ")...");
        if (useGpu) initCUDA();
        System.out.println(timestamp() + "Starting Autonomous Engine...");

        while (true) {
            handleGlobalStagnation();

            int lockedPieces = updateHandoffConfig();
            prepareSearchIteration(lockedPieces);

            boolean spaceExhausted = true;
            try {
                if (executeCpuSeedGeneration()) return;
                spaceExhausted = handleSearchHandoff();
            } catch (EvolutionLeapException e) {
                spaceExhausted = false;
                resetCounters();
            } catch (PoisonedBaseCampException e) {
                spaceExhausted = false;
                handleExtinctionEvent(lockedPieces);
            } catch (ExecutionException e) {
                spaceExhausted = !handleWorkerException(e);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }

            if (spaceExhausted) {
                handleSearchExhaustion(lockedPieces);
            }
        }
    }

    private void handleGlobalStagnation() {
        long minutesSinceProgress = (System.currentTimeMillis() - lastProgressTimestamp) / 60000;
        if (minutesSinceProgress >= stagnationLimitMinutes && deepestStep > 0) {
            int deepRetreat = 40 + new Random().nextInt(41);
            String msg = "\n" + timestamp() + " [!!!] AUTONOMOUS DEEP EXTINCTION [!!!]\n" +
                    timestamp() + " No progression in the last " + minutesSinceProgress + " minutes.\n" +
                    timestamp() + " Forces Base Camp all the way down to tile : " + deepRetreat + "\n";
            retreat(deepRetreat, msg);
            lastProgressTimestamp = System.currentTimeMillis();
            Arrays.fill(flatResumeBoard, -1);
        }
    }

    private int updateHandoffConfig() {
        int lockedPieces = 0;
        if (currentStrategy == BuildStrategy.TYPEWRITER && useGpu) {
            if (deepestStep > 160) {
                lockedPieces = Math.max(0, deepestStep - 60);
                handoffDepth = lockedPieces + 28;
                targetBatchSize = 8000;
            } else {
                lockedPieces = Math.max(0, deepestStep - 45);
                handoffDepth = lockedPieces + 30;
                targetBatchSize = 10000;
            }
            if (lockedPieces == 0 && deepestStep > 0) retreat(0, null);
        } else if (deepestStep > 0) {
            int retreatDistance = (deepestStep > 180) ? 80 : 30;
            lockedPieces = Math.max(0, deepestStep - retreatDistance);
            int gap = (lockedPieces > 180) ? 10 : (lockedPieces > 150) ? 12 : 15;
            targetBatchSize = (lockedPieces > 180) ? 5000 : (lockedPieces > 150) ? 8000 : 10000;
            handoffDepth = lockedPieces + gap;
        }
        return lockedPieces;
    }

    private void prepareSearchIteration(int lockedPieces) {
        if (lockedPieces > 0) {
            for (int step = 0; step < lockedPieces; step++) {
                int idx = buildOrder[step];
                if (lockCenter && idx == 135) continue;
                flatResumeBoard[idx] = bestBoard[idx];
            }
        } else {
            Arrays.fill(flatResumeBoard, -1);
        }
        gpuSeedBoards.clear();
        structuralDiversityFilter.clear();
        currentBatchSize.set(0);
    }

    private boolean executeCpuSeedGeneration() throws InterruptedException, ExecutionException {
        int activeBatch = (userBatchSizeOverride > 0) ? userBatchSizeOverride : targetBatchSize;
        List<SearchWorker> workers = new ArrayList<>();
        for (int i = 0; i < numCores; i++) workers.add(new SearchWorker(activeBatch));

        List<Future<Boolean>> results = executor.invokeAll(workers);
        for (Future<Boolean> f : results) {
            if (f.get()) return true;
        }
        return false;
    }

    private boolean handleSearchHandoff() {
        int activeBatch = (userBatchSizeOverride > 0) ? userBatchSizeOverride : targetBatchSize;
        int radarDistance = (currentStrategy == BuildStrategy.TYPEWRITER) ? 120 : 50;
        int foundSeeds = currentBatchSize.get();

        if (foundSeeds >= activeBatch) {
            runGpuHandoff(new ArrayList<>(gpuSeedBoards), this.handoffDepth, radarDistance);
            resetCounters();
            return false;
        } else if (foundSeeds > 0) {
            if (foundSeeds < activeBatch / 4) {
                System.out.println(timestamp() + ">>> CPU exhausted with only " + foundSeeds + " seeds. Base Camp is likely a dead end.");
                return true;
            }
            System.out.println(timestamp() + ">>> Sending partial batch: " + foundSeeds + " seeds to GPU...");
            runGpuHandoff(new ArrayList<>(gpuSeedBoards), this.handoffDepth, radarDistance);
            consecutiveExhaustions = 0;
            return false;
        }
        return true;
    }

    private void handleExtinctionEvent(int lockedPieces) {
        consecutiveExtinctions++;
        int retreatSize = (consecutiveExtinctions == 1) ? 8 : (consecutiveExtinctions == 2) ? 20 : 40;
        if (lockedPieces > retreatSize) {
            String msg = "\n" + timestamp() + "[!] EXTINCTION EVENT (" + consecutiveExtinctions + " failures)!\n" +
                    timestamp() + "[!] Structural dead-end detected. Retreating " + retreatSize + " steps to " + (lockedPieces - retreatSize);
            retreat(lockedPieces - retreatSize, msg);
        } else {
            retreat(0, null);
        }
        if (consecutiveExtinctions >= 3) consecutiveExtinctions = 0;
    }

    private boolean handleWorkerException(ExecutionException e) {
        Throwable cause = e.getCause();
        if (cause instanceof ManualOverrideException) {
            manualOverrideRequested = false;
            retreat(manualBaseCampTarget, "\n" + timestamp() + "[!] MANUAL OVERRIDE! Jumping to " + manualBaseCampTarget);
            deepestStep = manualBaseCampTarget + 30;
            return true;
        }
        e.printStackTrace();
        return false;
    }

    private void handleSearchExhaustion(int lockedPieces) {
        consecutiveExhaustions++;
        if (lockedPieces > 0) {
            int retreat = (currentStrategy == BuildStrategy.TYPEWRITER &&
                    (System.currentTimeMillis() - lastProgressTimestamp) / 60000 >= 2) ? 50 : 10;
            if (retreat == 50) System.out.println("\n" + timestamp() + ">>> [!!!] TYPEWRITER EXHAUSTION [!!!] Retreating 50 steps.");
            retreat(lockedPieces - retreat, timestamp() + ">>> Base camp exhausted. Searching new path...");
        } else if (currentStrategy == BuildStrategy.TYPEWRITER && useGpu) {
            System.out.println(timestamp() + ">>> [FIXED MODE] Seeds exhausted. Restarting search...");
        } else {
            System.out.println(timestamp() + "Total search space exhausted.");
        }
    }

    private void retreat(int targetStep, String logMessage) {
        deepestStep = Math.max(0, targetStep);
        for (int s = deepestStep; s < 256; s++) bestBoard[buildOrder[s]] = -1;
        if (lockCenter) bestBoard[135] = targetPiece;
        updateDisplay(deepestStep, buildDisplayBoard(bestBoard));
        if (logMessage != null) System.out.println(logMessage);
    }

    private void resetCounters() {
        consecutiveExtinctions = 0;
        consecutiveExhaustions = 0;
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

    private void updateDisplay(int score, int[][] displayBoard) {
        Main.updateDisplay(score, displayBoard);
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
            System.arraycopy(flatResumeBoard, 0, localBoard, 0, 256);
            System.arraycopy(flatResumeBoard, 0, localResumeBoard, 0, 256);
            System.arraycopy(usedPhysicalPieces, 0, localUsed, 0, 256);

            // *** FIX: mark all resumed pieces as used ***
            for (int i = 0; i < 256; i++) {
                int p = localBoard[i];
                if (p != -1) {
                    // Find the physical index for this piece value
                    for (int oi = 0; oi < 1024; oi++) {
                        if (inventory.allOrientations[oi] == p) {
                            localUsed[inventory.physicalMapping[oi]] = true;
                            break;
                        }
                    }
                }
            }

            if (lockCenter) {
                localBoard[135] = targetPiece;
                if (centerPhysicalIdx != -1) {
                    localUsed[centerPhysicalIdx] = true;
                }
            }
        }
        
//        public SearchWorker(int activeBatch) {
//            this.activeBatch = activeBatch;
//            // Initialize localBoard from the CURRENT base camp, not the empty flatBoard
//            System.arraycopy(flatResumeBoard, 0, localBoard, 0, 256);
//            System.arraycopy(flatResumeBoard, 0, localResumeBoard, 0, 256);
//            System.arraycopy(usedPhysicalPieces, 0, localUsed, 0, 256);
//
//            if (lockCenter) {
//                localBoard[135] = targetPiece;
//                if (centerPhysicalIdx != -1) {
//                    localUsed[centerPhysicalIdx] = true;
//                }
//            }
//        }

        private boolean hasAvailablePiece(int[] candidatePhysIds) {
            for (int physId : candidatePhysIds) {
                if (!localUsed[physId]) {
                    return true;
                }
            }
            return false;
        }

        /**
         * 2-Step Lookahead: Checks if a piece matching 'reqNorth' exists,
         * and if that piece leaves its own southern neighbor solvable.
         */
        private boolean isSecondaryNeighborViable(int reqNorth) {
            int[] candidates = piecesByNorth[reqNorth];
            for (int physId : candidates) {
                if (localUsed[physId]) {
                    continue;
                }
                for (int rot = 0; rot < 4; rot++) {
                    int p_orient = inventory.allOrientations[physId * 4 + rot];
                    if (PieceUtils.getNorth(p_orient) == reqNorth) {
                        int s_edge = PieceUtils.getSouth(p_orient);
                        if (hasAvailablePiece(piecesByNorth[s_edge])) {
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        @Override
        public Boolean call() {
            return solve(0);
        }

        private boolean solve(int step) {
            if (manualOverrideRequested) throw new ManualOverrideException();
            if (currentBatchSize.get() >= activeBatch) return false;
            if (step == 256) return true;

            int boardIdx = buildOrder[step];
            if (lockCenter && boardIdx == 135) return solve(step + 1);

            if (useGpu && step == handoffDepth) return registerGpuSeed();

            updateProgress(step);

            int row = boardIdx / 16, col = boardIdx % 16;
            int n = (row == 0) ? 0 : (localBoard[boardIdx - 16] != -1 ? PieceUtils.getSouth(localBoard[boardIdx - 16]) : PieceUtils.WILDCARD);
            int s = (row == 15) ? 0 : (localBoard[boardIdx + 16] != -1 ? PieceUtils.getNorth(localBoard[boardIdx + 16]) : PieceUtils.WILDCARD);
            int w = (col == 0) ? 0 : (localBoard[boardIdx - 1] != -1 ? PieceUtils.getEast(localBoard[boardIdx - 1]) : PieceUtils.WILDCARD);
            int e = (col == 15) ? 0 : (localBoard[boardIdx + 1] != -1 ? PieceUtils.getWest(localBoard[boardIdx + 1]) : PieceUtils.WILDCARD);

            List<Integer> pool = getCandidatePool(row, col);
            int size = pool.size(), offset = rnd.nextInt(size);

            for (int i = 0; i < size; i++) {
                if (currentBatchSize.get() >= activeBatch) return false;

                int orientationIdx = pool.get((i + offset) % size);
                int p = inventory.allOrientations[orientationIdx];
                int physicalIdx = inventory.physicalMapping[orientationIdx];

                if (localUsed[physicalIdx] || (localResumeBoard[boardIdx] != -1 && p != localResumeBoard[boardIdx])) continue;

                if (matches(p, n, e, s, w) && passesLookahead(p, step, row, col, boardIdx)) {
                    localBoard[boardIdx] = p;
                    localUsed[physicalIdx] = true;
                    int ghost = localResumeBoard[boardIdx];
                    localResumeBoard[boardIdx] = -1;

                    if (solve(step + 1)) return true;

                    localBoard[boardIdx] = -1;
                    localUsed[physicalIdx] = false;
                    localResumeBoard[boardIdx] = ghost;
                }
            }
            return false;
        }

        private boolean registerGpuSeed() {
            if (currentBatchSize.get() >= activeBatch) return false;
            int foundationDepth = Math.max(1, handoffDepth - 5);
            long hash = 0;
            for (int i = 0; i < foundationDepth; i++) {
                long p = localBoard[buildOrder[i]];
                hash ^= (p * 0x9e3779b97f4a7c15L) + 0x6c62272e07bb0142L + (hash << 6) + (hash >>> 2);
            }
            if (!structuralDiversityFilter.add(hash)) return false;

            int[] cloned = new int[256];
            System.arraycopy(localBoard, 0, cloned, 0, 256);
            gpuSeedBoards.add(cloned);

            int size = currentBatchSize.incrementAndGet();
            if (size % 1000 == 0) System.out.println(timestamp() + "   [CPU Pool] Found " + size + " / " + activeBatch + " seeds...");
            return false;
        }

        private void updateProgress(int step) {
            if (step <= deepestStep) return;
            synchronized (displayLock) {
                if (step > deepestStep) {
                    deepestStep = step;
                    System.arraycopy(localBoard, 0, bestBoard, 0, 256);
                    updateDisplay(step, buildDisplayBoard(bestBoard));
                    if (step + 1 > absoluteHighScore) {
                        absoluteHighScore = step + 1;
                        System.arraycopy(localBoard, 0, recordBoard, 0, 256);
                        RecordManager.saveRecord(buildDisplayBoard(bestBoard), absoluteHighScore, saveProfile);
                        CheckpointManager.save(buildDisplayBoard(bestBoard), saveProfile);
                        System.out.println(timestamp() + ">>> NY ALL-TIME HIGH SCORE: " + absoluteHighScore + " PIECES! <<<");
                    }
                }
            }
        }

        private List<Integer> getCandidatePool(int row, int col) {
            int b_req = (row == 0 || row == 15 ? 1 : 0) + (col == 0 || col == 15 ? 1 : 0);
            return (b_req == 2) ? inventory.corners : (b_req == 1) ? inventory.edges : inventory.interior;
        }

        private boolean passesLookahead(int p, int step, int row, int col, int idx) {
            if (row < 15 && localBoard[idx + 16] == -1) {
                int reqN = PieceUtils.getSouth(p);
                if (!hasAvailablePiece(piecesByNorth[reqN])) return false;
                if (step > 40 && row < 14 && localBoard[idx + 32] == -1 && !isSecondaryNeighborViable(reqN)) return false;
            }
            if (col < 15 && localBoard[idx + 1] == -1) {
                int reqW = PieceUtils.getEast(p);
                if (!hasAvailablePiece(piecesByWest[reqW])) return false;
            }
            return true;
        }

        private boolean matches(int p, int n, int e, int s, int w) {
            if (n != PieceUtils.WILDCARD && ((p >>> 24) & 0xFF) != n) {
                return false;
            }
            if (e != PieceUtils.WILDCARD && ((p >>> 16) & 0xFF) != e) {
                return false;
            }
            if (s != PieceUtils.WILDCARD && ((p >>> 8) & 0xFF) != s) {
                return false;
            }
            return w == PieceUtils.WILDCARD || PieceUtils.getWest(p) == w;
        }
    }
}
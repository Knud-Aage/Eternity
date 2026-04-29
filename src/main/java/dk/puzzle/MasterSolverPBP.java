package dk.puzzle;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static jcuda.driver.JCudaDriver.*;

public class MasterSolverPBP implements Runnable {

    public static final int SWISS_CHEESE_LEVEL = 214;
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
    private final AtomicLong globalTrialCount = new AtomicLong(0);
    private final int[] buildOrder;
    private final int[][] piecesByNorth = new int[256][];
    private final int[][] piecesByEast = new int[256][];
    private final int[][] piecesBySouth = new int[256][];
    private final int[][] piecesByWest = new int[256][];
    private final int[] recordBoard = new int[256];
    private final CompatibilityIndex compatIndex;
    private final int[] tabuTenure = new int[256]; // Holder styr på hvornår en brik "frigives" fra Tabu
    private long lastThroughputReportTime = System.currentTimeMillis();
    private CUfunction dfsFunction;
    private CUfunction repairFunction;
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
    private long lastRepairPrintTime = 0;
    private long repairLoopsCounter = 0;
    private long totalRepairVariationsTested = 0;
    private long repairStartTime = 0;
    private volatile double targetedHolesPercentage = 0.70;
    private int currentRepairIteration = 0;


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
            handoffDepth = 200;
        }

        int[][] loaded = CheckpointManager.loadSmartCheckpoint(saveProfile);
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
                this.deepestStep = highestStepLoaded + 1;
                this.absoluteHighScore = highestStepLoaded + 1;
                if (lockCenter) {
                    bestBoard[135] = targetPiece;
                }
                System.arraycopy(bestBoard, 0, flatBoard, 0, 256);
                System.arraycopy(bestBoard, 0, flatResumeBoard, 0, 256);
                updateDisplay(absoluteHighScore, buildDisplayBoard(bestBoard));
                System.out.println(timestamp() + ">>> Loaded checkpoint: " + absoluteHighScore + " pieces.");
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

        // Convert to fast native int arrays
        for (int i = 0; i < 256; i++) {
            piecesByNorth[i] = tempNorth[i].stream().mapToInt(Integer::intValue).toArray();
            piecesByEast[i] = tempEast[i].stream().mapToInt(Integer::intValue).toArray();
            piecesBySouth[i] = tempSouth[i].stream().mapToInt(Integer::intValue).toArray();
            piecesByWest[i] = tempWest[i].stream().mapToInt(Integer::intValue).toArray();
        }
        this.compatIndex = new CompatibilityIndex(inventory.allOrientations, inventory.physicalMapping);
        this.handoffDepth = 170;
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

    public void setTargetedHolesPercentage(double percentage) {
        this.targetedHolesPercentage = percentage;
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

        dfsFunction = new CUfunction();
        cuModuleGetFunction(dfsFunction, cuModule, "solvePBP");

        repairFunction = new CUfunction();
        cuModuleGetFunction(repairFunction, cuModule, "solveRepairMode");

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
        if (numBoards == 0) {
            return;
        }

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
            CUdeviceptr d_threadDepths = gpuAlloc((long) numBoards * Sizeof.INT, gpuPointers);
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
            cuLaunchKernel(dfsFunction, (int) Math.ceil(numBoards / 256.0), 1, 1, 256, 1, 1, 0, null, kernelParams,
                    null);
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
                        updateDisplay(countPieces(bestBoard), buildDisplayBoard(bestBoard));

                        if (deepestStep > absoluteHighScore) {
                            absoluteHighScore = deepestStep;
                            lastProgressTimestamp = System.currentTimeMillis();
                            RecordManager.saveRecord(buildDisplayBoard(bestBoard), absoluteHighScore, saveProfile);
                            CheckpointManager.saveRecordCheckpoint(buildDisplayBoard(bestBoard), absoluteHighScore,
                                    saveProfile);
                            System.out.println(timestamp() + ">>> NEW ALL-TIME HIGH SCORE (GPU): " + absoluteHighScore + " PIECES! <<<");
                        }
                    }
                }
                if (deepestStep > handoffDepth + 30) {
                    throw new EvolutionLeapException();
                }
            }
        } finally {
            gpuPointers.forEach(JCudaDriver::cuMemFree);
        }
    }

    private void runRepairGpuHandoff(List<int[]> swissCheeseBoards) {
        int numBoards = swissCheeseBoards.size();
        if (numBoards == 0) {
            return;
        }

        // 1. Flatten the 2D boards into a 1D array for the GPU
        int[] flatBoards = new int[numBoards * 256];
        for (int i = 0; i < numBoards; i++) {
            System.arraycopy(swissCheeseBoards.get(i), 0, flatBoards, i * 256, 256);
        }

        // ==========================================================
        // 2. ALLOCATE AND COPY MEMORY TO GPU
        // ==========================================================
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
        // GPU'en skal vide, hvad highscoren er lige nu, for at kunne slå den!
        cuMemcpyHtoD(d_gpuHighScore, Pointer.to(new int[]{absoluteHighScore}), Sizeof.INT);

        CUdeviceptr d_bestBoardOut = new CUdeviceptr();
        cuMemAlloc(d_bestBoardOut, 256L * Sizeof.INT);

        CUdeviceptr d_totalSteps = new CUdeviceptr();
        cuMemAlloc(d_totalSteps, Sizeof.LONG);
        cuMemcpyHtoD(d_totalSteps, Pointer.to(new long[]{0L}), Sizeof.LONG);

        // ==========================================================
        // 3. SETUP & LAUNCH THE SURGEON KERNEL
        // ==========================================================
        int maxStepsPerThread = 150000; // Limit so GPU doesn't hang on bad holes

        Pointer kernelParameters = Pointer.to(
                Pointer.to(d_partialBoards),
                Pointer.to(new int[]{numBoards}),
                Pointer.to(d_allOrientations),
                Pointer.to(d_physicalMapping),
                Pointer.to(d_solution),
                Pointer.to(d_solvedFlag),
                Pointer.to(d_gpuHighScore),
                Pointer.to(d_bestBoardOut),
                Pointer.to(d_totalSteps),
                Pointer.to(new int[]{maxStepsPerThread})
        );

        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) numBoards / blockSizeX);

        cuLaunchKernel(repairFunction,
                gridSizeX, 1, 1,
                blockSizeX, 1, 1,
                0, null,
                kernelParameters, null
        );
        cuCtxSynchronize();

        // ==========================================================
        // 4. RETRIEVE RESULTS FROM THE GPU
        // ==========================================================

        // Did the surgeon beat the high score?
        int[] resultHighScore = new int[1];
        cuMemcpyDtoH(Pointer.to(resultHighScore), d_gpuHighScore, Sizeof.INT);

        if (resultHighScore[0] > absoluteHighScore) {
            int[] repairedBoard = new int[256];
            cuMemcpyDtoH(Pointer.to(repairedBoard), d_bestBoardOut, 256L * Sizeof.INT);

            synchronized (displayLock) {
                if (resultHighScore[0] > absoluteHighScore) {

                    // =======================================================
                    // TABU SEARCH: IDENTIFICER NYE/FLYTTE BRIKKER OG FRED DEM!
                    // =======================================================
                    int changedPieces = 0;
                    for (int i = 0; i < 256; i++) {
                        // Hvis feltet har ændret sig (f.eks. fået en ny brik)
                        if (repairedBoard[i] != bestBoard[i] && repairedBoard[i] != -1) {
                            // Sæt feltet på Tabu-listen i de næste 100 batches!
                            // (Kirurgen tvinges nu til at bygge videre udenom disse felter)
                            tabuTenure[i] = currentRepairIteration + 100;
                            changedPieces++;
                        }
                    }
                    System.out.println(timestamp() + ">>> TABU SEARCH: Fredet " + changedPieces + " nyligt flyttede brikker for de næste 100 batches.");
                    // =======================================================

                    absoluteHighScore = resultHighScore[0];
                    deepestStep = absoluteHighScore;
                    System.arraycopy(repairedBoard, 0, bestBoard, 0, 256);
                    updateDisplay(countPieces(bestBoard), buildDisplayBoard(bestBoard));

                    RecordManager.saveRecord(buildDisplayBoard(bestBoard), absoluteHighScore, saveProfile);
                    CheckpointManager.saveRecordCheckpoint(buildDisplayBoard(bestBoard), absoluteHighScore, saveProfile);

                    System.out.println("\n" + timestamp() + ">>> KIRURGEN SLOG REKORDEN! NY HIGH SCORE: " + absoluteHighScore + " <<<");
                }
            }
        }
        // Did the surgeon actually SOLVE the whole puzzle?
        int[] solved = new int[1];
        cuMemcpyDtoH(Pointer.to(solved), d_solvedFlag, Sizeof.INT);
        if (solved[0] == 1) {
            int[] winningBoard = new int[256];
            cuMemcpyDtoH(Pointer.to(winningBoard), d_solution, 256L * Sizeof.INT);
            System.out.println("\n" + timestamp() + ">>> ETERNITY II LØST AF REPAIR MODE (GPU)!!! <<<");
            updateDisplay(256, buildDisplayBoard(winningBoard));
            System.exit(0);
        }

        // ==========================================================
        // 5. CLEANUP DEVICE MEMORY (Prevent memory leaks)
        // ==========================================================
        cuMemFree(d_partialBoards);
        cuMemFree(d_allOrientations);
        cuMemFree(d_physicalMapping);
        cuMemFree(d_solution);
        cuMemFree(d_solvedFlag);
        cuMemFree(d_gpuHighScore);
        cuMemFree(d_bestBoardOut);
        cuMemFree(d_totalSteps);
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
        double speed = steps / Math.max(timeMs / 1000.0, 0.001);
        globalTrialCount.addAndGet(steps);
        long sum = 0;
        int max = 0, dead = 0;
        for (int d : depths) {
            sum += d;
            if (d > max) {
                max = d;
            }
            if (d <= start + 5) {
                dead++;
            }
        }
        System.out.printf("%sGPU | %d ms | %,.0f pcs/s | Avg Depth: %.1f | Max: %d | Dead < 5: %d/%d\n",
                timestamp(), timeMs, speed, (double) sum / n, max, dead, n);
        if (n > 0 && (double) dead / n >= extinctionThreshold) {
            throw new PoisonedBaseCampException();
        }
    }

    private void cuFreeResources(CUdeviceptr... pointers) {
        for (CUdeviceptr ptr : pointers) {
            cuMemFree(ptr);
        }
    }

    @Override
    public void run() {
        try {
            // Register a Shutdown Hook to save when the program is closed or stopped
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                System.out.println("\n" + timestamp() + ">>> Shutdown hook: Saving final checkpoint...");
                synchronized (displayLock) {
                    CheckpointManager.saveWorkingState(buildDisplayBoard(bestBoard));
                }
            }));

            System.out.println(timestamp() + "Starting Multithreaded Engine (Profile: " + saveProfile + ")...");
            if (useGpu) {
                initCUDA();
            }
            System.out.println(timestamp() + "Starting Autonomous Engine...");
            long lastPeriodicSave = System.currentTimeMillis();

            while (true) {
                handleGlobalStagnation();

                // --- Throughput Reporting ---
                long now = System.currentTimeMillis();
                long elapsed = now - lastThroughputReportTime;
                if (elapsed >= 5000) { // Report every 5 seconds
                    long trials = globalTrialCount.getAndSet(0);
                    double tps = trials / (elapsed / 1000.0);
                    System.out.printf("%s[STATS] Throughput: %,.0f trials/sec (Total in window: %,d)\n",
                            timestamp(), tps, trials);
                    lastThroughputReportTime = now;
                }

                int lockedPieces = updateHandoffConfig();
                prepareSearchIteration(lockedPieces);

                boolean spaceExhausted = true;
                try {
                    if (executeCpuSeedGeneration()) {
                        return;
                    }
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

                // Safety: Periodic save every 5 minutes
                if (System.currentTimeMillis() - lastPeriodicSave > 300_000) {
                    synchronized (displayLock) {
                        CheckpointManager.saveWorkingState(buildDisplayBoard(bestBoard));
                    }
                    lastPeriodicSave = System.currentTimeMillis();
                }

                if (spaceExhausted) {
                    handleSearchExhaustion(lockedPieces);
                }
            }

        } finally {
            CheckpointManager.saveWorkingState(buildDisplayBoard(bestBoard));
        }
    }

    private void handleGlobalStagnation() {
        if (currentStrategy == BuildStrategy.TYPEWRITER) return;
        long minutesSinceProgress = (System.currentTimeMillis() - lastProgressTimestamp) / 60000;
        if (minutesSinceProgress >= stagnationLimitMinutes && deepestStep > 0 && deepestStep <= SWISS_CHEESE_LEVEL) {
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
                if (deepestStep > 0) {
                    lockedPieces = Math.max(0, deepestStep - 45);
                    handoffDepth = lockedPieces + 30;
                } else {
                    lockedPieces = 0;
                    handoffDepth = 200;
                }
                targetBatchSize = 10000;
            }
            if (lockedPieces == 0 && deepestStep > 0) {
                retreat(0, null);
            }
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
        Arrays.fill(flatResumeBoard, -1);

        Arrays.fill(usedPhysicalPieces, false);

        if (lockedPieces > 0) {
            for (int step = 0; step < lockedPieces; step++) {
                int idx = buildOrder[step];
                if (lockCenter && idx == 135) {
                    continue;
                }
                flatResumeBoard[idx] = bestBoard[idx];
            }
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
            if (f.get()) {
                return true;
            }
        }
        return false;
    }

    private boolean handleSearchHandoff() {
        int activeBatch = (userBatchSizeOverride > 0) ? userBatchSizeOverride : targetBatchSize;
        int radarDistance = (currentStrategy == BuildStrategy.TYPEWRITER) ? 120 : 50;
        int foundSeeds = currentBatchSize.get();
        int numClones = 10000;
        int holesToPunch = 15;

        if (!useGpu) {
            if (foundSeeds > 0) {
                // I CPU-mode betragter vi fundne seeds som "fundne stier".
                // Vi nulstiller bare og lader CPU'en køre videre uden at trigge retreat.
                currentBatchSize.set(0);
                gpuSeedBoards.clear();
                return false; // FALSE betyder: "Bliv hvor du er, ikke træk dig tilbage"
            }
            return false;
        }

        if (repairStartTime == 0) repairStartTime = System.currentTimeMillis();
        totalRepairVariationsTested += numClones;
        repairLoopsCounter++;

        if (useGpu && absoluteHighScore >= SWISS_CHEESE_LEVEL && (consecutiveExtinctions >= 1 || consecutiveExhaustions >= 1 || foundSeeds == 0)) {

            repairLoopsCounter++;
            long now = System.currentTimeMillis();
            List<int[]> swissCheeseBoards = punchHolesInBestBoard(bestBoard, numClones, holesToPunch);
            // Tæl op for hver eneste Tabu-runde
            currentRepairIteration++;

            if (now - lastRepairPrintTime > 2000) {
                long secondsRunning = (now - repairStartTime) / 1000;
                String timeFormatted = String.format("%02d:%02d:%02d", secondsRunning / 3600, (secondsRunning % 3600) / 60, secondsRunning % 60);

                System.out.println(String.format("%s >>> [REPAIR MODE] Uptime: %s | Speed: %d batches/sec | Total variations: %,d",
                        timestamp(), timeFormatted, (repairLoopsCounter / 2), totalRepairVariationsTested));
                // =======================================================
                int[] visualBoard = new int[256];
                System.arraycopy(swissCheeseBoards.get(0), 0, visualBoard, 0, 256);

                for (int i = 0; i < 256; i++) {
                    if (visualBoard[i] == -2) visualBoard[i] = -1;
                }
                updateDisplay(countPieces(visualBoard), buildDisplayBoard(visualBoard));
                // =======================================================

                lastRepairPrintTime = now;
                repairLoopsCounter = 0;
            }
            runRepairGpuHandoff(swissCheeseBoards);
            updateDisplay(absoluteHighScore, buildDisplayBoard(bestBoard));

            consecutiveExtinctions = 0;
            consecutiveExhaustions = 0;
            resetCounters();

            deepestStep = absoluteHighScore;
//            if (repairLoopsCounter == 1) {
//                Main.updateDisplay(absoluteHighScore, buildDisplayBoard(bestBoard));
//            }

            return false;
        }
        // =========================================================

        if (foundSeeds >= activeBatch) {
            if (useGpu) {
                runGpuHandoff(new ArrayList<>(gpuSeedBoards), this.handoffDepth, radarDistance);
            }
            resetCounters();
            return false;
        } else if (foundSeeds > 0) {
//            System.out.println(timestamp() + ">>> CPU space exhausted. Sending partial batch to GPU...");
            runGpuHandoff(new ArrayList<>(gpuSeedBoards), this.handoffDepth, 120);
            resetCounters();
            return true;
        }

//        System.out.println(timestamp() + ">>> ZERO seeds found. Base Camp is a dead end.");
        return true;
    }

    private void handleExtinctionEvent(int lockedPieces) {
        consecutiveExtinctions++;

        // =======================================================
        // ANKER-LOGIK: KUN SPIRAL MÅ TRÆKKE SIG TILBAGE
        // =======================================================
        if (currentStrategy == BuildStrategy.TYPEWRITER) {
            deepestStep = absoluteHighScore;
            if (consecutiveExtinctions >= 3) consecutiveExtinctions = 0;
            return;
        }
        // =======================================================

        // Herunder kører koden kun, hvis strategien er SPIRAL
        int retreatSize = (consecutiveExtinctions == 1) ? 8 : (consecutiveExtinctions == 2) ? 20 : 40;
        if (lockedPieces > retreatSize) {
            String msg = "\n" + timestamp() + "[!] EXTINCTION EVENT (" + consecutiveExtinctions + " failures)!\n" +
                    timestamp() + "[!] Structural dead-end detected. Retreating " + retreatSize + " steps to " + (deepestStep - retreatSize);

            retreat(deepestStep - retreatSize, msg);
        } else {
            retreat(0, null);
        }
        if (consecutiveExtinctions >= 3) consecutiveExtinctions = 0;
    }

    private boolean handleWorkerException(ExecutionException e) {
        Throwable cause = e.getCause();
        if (cause instanceof ManualOverrideException) {
            manualOverrideRequested = false;
            retreat(manualBaseCampTarget,
                    "\n" + timestamp() + "[!] MANUAL OVERRIDE! Jumping to " + manualBaseCampTarget);

            // Fix: ensure absoluteHighScore is updated to reflect the new state,
            // allowing us to exit Surgeon/Repair mode if necessary.
            deepestStep = manualBaseCampTarget + 30;
            absoluteHighScore = deepestStep;
            lastProgressTimestamp = System.currentTimeMillis();
            resetCounters();

            return true;
        }
        e.printStackTrace();
        return false;
    }

    private void handleSearchExhaustion(int lockedPieces) {
        consecutiveExhaustions++;

        // =======================================================
        // ANKER-LOGIK: KUN SPIRAL MÅ TRÆKKE SIG TILBAGE
        // =======================================================
        if (currentStrategy == BuildStrategy.TYPEWRITER) {
            if (consecutiveExhaustions >= 15) {
                // If we are stuck with 0 seeds 3 times in a row, the base camp is "poisoned".
                // We must retreat slightly to find a branch that actually has seeds.
                int retreatAmount = 5 + new Random().nextInt(30);
                retreat(deepestStep - retreatAmount, timestamp() + ">>> [RECOVERY] Zero seeds found repeatedly. Retreating " + retreatAmount + " pieces to find a healthy branch...");
                consecutiveExhaustions = 0;
            } else {
                deepestStep = absoluteHighScore;
            }
            return;
        }
        // =======================================================

        // Herunder kører koden kun, hvis strategien er SPIRAL
        if (lockedPieces > 0) {
            int retreat = 10; // Standard retreat for Spiral
            retreat(deepestStep - retreat, timestamp() + ">>> Base camp exhausted. Searching new path (SPIRAL)...");
        } else {
            System.out.println(timestamp() + "Total search space exhausted.");
        }
    }

    private void retreat(int targetStep, String logMessage) {
        deepestStep = Math.max(0, targetStep);
        for (int s = deepestStep; s < 256; s++) bestBoard[buildOrder[s]] = -1;
        if (lockCenter) {
            bestBoard[135] = targetPiece;
        }
        updateDisplay(countPieces(bestBoard), buildDisplayBoard(bestBoard));
        if (logMessage != null) {
            System.out.println(logMessage);
        }
    }

    private void resetCounters() {
        consecutiveExtinctions = 0;
        consecutiveExhaustions = 0;
    }

    private int countPieces(int[] board) {
        int count = 0;
        for (int p : board) if (p != -1 && p != -2) count++;
        return count;
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

// ==========================================================
    // --- LATE GAME: REPAIR MODE (Large Neighborhood Search) ---
    // ==========================================================

    /**
     * Scores each placed cell by the number of edge mismatches with its neighbors.
     * A score of 0 means the piece fits perfectly on all sides.
     * A score of 4 means all four neighbors are mismatched (maximally conflicted).
     * Empty cells (-1) and the locked center are excluded.
     *
     * @return array of conflict scores, indexed by board position (0..255)
     */
    private int[] scoreConflicts(int[] sourceBoard) {
        int[] conflicts = new int[256];
        for (int idx = 0; idx < 256; idx++) {
            int p = sourceBoard[idx];
            if (p == -1 || p == -2) {
                continue;
            }
            if (lockCenter && idx == 135) {
                continue;
            }

            int row = idx / 16;
            int col = idx % 16;
            int score = 0;

            // North neighbor
            if (row > 0) {
                int northPiece = sourceBoard[idx - 16];
                if (northPiece > 0 && PieceUtils.getSouth(northPiece) != PieceUtils.getNorth(p)) {
                    score++;
                }
            }
            // South neighbor
            if (row < 15) {
                int southPiece = sourceBoard[idx + 16];
                if (southPiece > 0 && PieceUtils.getNorth(southPiece) != PieceUtils.getSouth(p)) {
                    score++;
                }
            }
            // West neighbor
            if (col > 0) {
                int westPiece = sourceBoard[idx - 1];
                if (westPiece > 0 && PieceUtils.getEast(westPiece) != PieceUtils.getWest(p)) {
                    score++;
                }
            }
            // East neighbor
            if (col < 15) {
                int eastPiece = sourceBoard[idx + 1];
                if (eastPiece > 0 && PieceUtils.getWest(eastPiece) != PieceUtils.getEast(p)) {
                    score++;
                }
            }

            conflicts[idx] = score;
        }
        return conflicts;
    }

    /**
     * Takes the best known board and generates thousands of clones using targeted LNS.
     * <p>
     * Hole selection strategy (per clone):
     * - A percentage of holes are drawn from the HIGH-CONFLICT region (cells with the most
     * edge mismatches), focusing repair effort where it matters most.
     * - The remaining holes are drawn randomly from all placed cells, preserving diversity
     * and preventing the search from getting tunnel-visioned on one area.
     * <p>
     * Additionally, the "frontier" cell (the next unplaced position in build order)
     * is always included as a hole, so every clone targets progression.
     */
    private List<int[]> punchHolesInBestBoard(int[] sourceBoard, int numClones, int numHoles) {
        List<int[]> swissCheeseBoards = new ArrayList<>(numClones);
        Random rnd = new Random();

        // --- 1. Collect placed indices and their conflict scores ---
        int[] placedIndices = new int[256];
        int placedCount = 0;

        for (int i = 0; i < 256; i++) {
            if (sourceBoard[i] != -1 && sourceBoard[i] != -2) {
                if (lockCenter && i == 135) continue;

                // ========================================================
                // TABU SEARCH FILTER: Må vi røre denne brik?
                // ========================================================
                if (tabuTenure[i] > currentRepairIteration) {
                    continue; // BRIKKEN ER FREDET! Vi må ikke skyde hul i den.
                }

                placedIndices[placedCount++] = i;
            }
        }

        int actualHoles = Math.min(numHoles, placedCount);

        // --- 2. Score conflicts and build a sorted high-conflict candidate list ---
        int[] conflicts = scoreConflicts(sourceBoard);

        // Collect and sort placed indices by descending conflict score
        Integer[] sortedByConflict = new Integer[placedCount];
        for (int i = 0; i < placedCount; i++) sortedByConflict[i] = placedIndices[i];
        Arrays.sort(sortedByConflict, (a, b) -> Integer.compare(conflicts[b], conflicts[a]));

        // Top 25% most-conflicted cells form the "hot zone"
        int hotZoneSize = Math.max(actualHoles, placedCount / 4);
        int[] hotZone = new int[hotZoneSize];
        for (int i = 0; i < hotZoneSize; i++) hotZone[i] = sortedByConflict[i];

        // --- 3. Generate clones with targeted holes ---
        int targetedHoles = (int) Math.round(actualHoles * targetedHolesPercentage);
        int randomHoles = actualHoles - targetedHoles;

        for (int clone = 0; clone < numClones; clone++) {
            int[] clonedBoard = new int[256];
            System.arraycopy(sourceBoard, 0, clonedBoard, 0, 256);

            // Track which indices we've already punched this clone (avoid double-punching)
            boolean[] punched = new boolean[256];

            // Phase A: targeted holes from the hot zone (shuffle + pick first N)
            int hotPicked = 0;
            int[] hotShuffled = hotZone.clone();
            for (int i = 0; i < hotShuffled.length && hotPicked < targetedHoles; i++) {
                int swapIdx = i + rnd.nextInt(hotShuffled.length - i);
                int tmp = hotShuffled[i];
                hotShuffled[i] = hotShuffled[swapIdx];
                hotShuffled[swapIdx] = tmp;
                int holeIdx = hotShuffled[i];
                if (!punched[holeIdx]) {
                    clonedBoard[holeIdx] = -2;
                    punched[holeIdx] = true;
                    hotPicked++;
                }
            }

            // Phase B: random holes from all placed cells for diversity
            int randPicked = 0;
            int[] allShuffled = Arrays.copyOf(placedIndices, placedCount);
            for (int i = 0; i < allShuffled.length && randPicked < randomHoles; i++) {
                int swapIdx = i + rnd.nextInt(allShuffled.length - i);
                int tmp = allShuffled[i];
                allShuffled[i] = allShuffled[swapIdx];
                allShuffled[swapIdx] = tmp;
                int holeIdx = allShuffled[i];
                if (!punched[holeIdx]) {
                    clonedBoard[holeIdx] = -2;
                    punched[holeIdx] = true;
                    randPicked++;
                }
            }

            // Always punch the frontier cell (next unplaced position in build order)
            if (absoluteHighScore < 256) {
                int nextTargetIdx = buildOrder[absoluteHighScore];
                clonedBoard[nextTargetIdx] = -2;
            }

            swissCheeseBoards.add(clonedBoard);
        }

        return swissCheeseBoards;
    }

    //    private void updateDisplay(int score, int[][] displayBoard) {
//        Main.updateDisplay(score, displayBoard);
//    }
    private void updateDisplay(int score, int[][] displayBoard) {
        // Nu sender den altid både det aktuelle antal brikker og rekorden til titelbaren!
        Main.updateDisplay(score, this.absoluteHighScore, displayBoard);
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
        private long localTrialCount = 0;

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
            if (manualOverrideRequested) {
                throw new ManualOverrideException();
            }
            if (currentBatchSize.get() >= activeBatch) {
                return false;
            }

            localTrialCount++;
            if (localTrialCount >= 100000) { // Batch updates to avoid memory contention
                globalTrialCount.addAndGet(localTrialCount);
                localTrialCount = 0;
            }

            if (step == 256) {
                return true;
            }

            int boardIdx = buildOrder[step];
            if (localBoard[boardIdx] != -1) {
                if (useGpu && step == handoffDepth) return registerGpuSeed();
                return solve(step + 1);
            }

            if (lockCenter && boardIdx == 135) {
                return solve(step + 1);
            }

            if (useGpu && step == handoffDepth) {
                return registerGpuSeed();
            }

            updateProgress(step);

            int row = boardIdx / 16;
            int col = boardIdx % 16;
            int n = (row == 0) ? 0 : (localBoard[boardIdx - 16] != -1 ?
                    PieceUtils.getSouth(localBoard[boardIdx - 16]) : PieceUtils.WILDCARD);
            int s = (row == 15) ? 0 : (localBoard[boardIdx + 16] != -1 ?
                    PieceUtils.getNorth(localBoard[boardIdx + 16]) : PieceUtils.WILDCARD);
            int w = (col == 0) ? 0 : (localBoard[boardIdx - 1] != -1 ? PieceUtils.getEast(localBoard[boardIdx - 1]) :
                                      PieceUtils.WILDCARD);
            int e = (col == 15) ? 0 : (localBoard[boardIdx + 1] != -1 ? PieceUtils.getWest(localBoard[boardIdx + 1])
                                       : PieceUtils.WILDCARD);

            int northReq = (row == 0) ? 0 : (localBoard[boardIdx - 16] != -1 ? PieceUtils.getSouth(localBoard[boardIdx - 16]) : CompatibilityIndex.WILDCARD);
            int southReq = (row == 15) ? 0 : (localBoard[boardIdx + 16] != -1 ? PieceUtils.getNorth(localBoard[boardIdx + 16]) : CompatibilityIndex.WILDCARD);
            int westReq  = (col == 0) ? 0 : (localBoard[boardIdx - 1] != -1 ? PieceUtils.getEast(localBoard[boardIdx - 1])  : CompatibilityIndex.WILDCARD);
            int eastReq  = (col == 15) ? 0 : (localBoard[boardIdx + 1] != -1 ? PieceUtils.getWest(localBoard[boardIdx + 1]) : CompatibilityIndex.WILDCARD);

            // 1. Slå op i BitMask indekset!
            java.util.BitSet candidates = compatIndex.candidatesFor(northReq, eastReq, southReq, westReq);
            compatIndex.andNotUsed(candidates, localUsed);

            // 2. Hvis vi har en ghost-brik (resumeBoard), tvinger vi den til at bruge den
            if (localResumeBoard[boardIdx] != -1) {
                int resumeP = localResumeBoard[boardIdx];
                for (int oi = candidates.nextSetBit(0); oi >= 0; oi = candidates.nextSetBit(oi + 1)) {
                    if (inventory.allOrientations[oi] != resumeP) candidates.clear(oi);
                }
            }

            int candidateCount = candidates.cardinality();
            if (candidateCount == 0) return false;

            // 3. Konverter de overlevende bits til et randomiseret array
            int[] orientIdxs = new int[candidateCount];
            int k = 0;
            for (int oi = candidates.nextSetBit(0); oi >= 0; oi = candidates.nextSetBit(oi + 1)) {
                orientIdxs[k++] = oi;
            }
            int offset = rnd.nextInt(candidateCount);

            // 4. Test brikkerne
            for (int i = 0; i < candidateCount; i++) {
                if (currentBatchSize.get() >= activeBatch) return false;

                int orientationIdx = orientIdxs[(i + offset) % candidateCount];
                int p = inventory.allOrientations[orientationIdx];
                int physicalIdx = inventory.physicalMapping[orientationIdx];

                // PassesLookahead tjekker "The Future"
                if (passesLookahead(p, step, row, col, boardIdx)) {
                    localBoard[boardIdx] = p;
                    localUsed[physicalIdx] = true;
                    int ghost = localResumeBoard[boardIdx];
                    localResumeBoard[boardIdx] = -1;

                    if (solve(step + 1)) return true;

                    // Backtrack
                    localBoard[boardIdx] = -1;
                    localUsed[physicalIdx] = false;
                    localResumeBoard[boardIdx] = ghost;
                }
            }
            return false;
//            List<Integer> pool = getCandidatePool(row, col);
//            int size = pool.size();
//            int offset = rnd.nextInt(size);
//
//            for (int i = 0; i < size; i++) {
//                if (currentBatchSize.get() >= activeBatch) {
//                    return false;
//                }
//
//                int orientationIdx = pool.get((i + offset) % size);
//                int p = inventory.allOrientations[orientationIdx];
//                int physicalIdx = inventory.physicalMapping[orientationIdx];
//
//                if (localUsed[physicalIdx] || (localResumeBoard[boardIdx] != -1 && p != localResumeBoard[boardIdx])) {
//                    continue;
//                }
//
//                if (matches(p, n, e, s, w) && passesLookahead(p, step, row, col, boardIdx)) {
//                    localBoard[boardIdx] = p;
//                    localUsed[physicalIdx] = true;
//                    int ghost = localResumeBoard[boardIdx];
//                    localResumeBoard[boardIdx] = -1;
//
//                    if (solve(step + 1)) {
//                        return true;
//                    }
//
//                    localBoard[boardIdx] = -1;
//                    localUsed[physicalIdx] = false;
//                    localResumeBoard[boardIdx] = ghost;
//                }
//            }
        }

        private boolean registerGpuSeed() {
            if (currentBatchSize.get() >= activeBatch) {
                return false;
            }
            int foundationDepth = Math.max(1, handoffDepth - 5);
            long hash = 0;
            for (int i = 0; i < foundationDepth; i++) {
                long p = localBoard[buildOrder[i]];
                hash ^= (p * 0x9e3779b97f4a7c15L) + 0x6c62272e07bb0142L + (hash << 6) + (hash >>> 2);
            }
            if (!structuralDiversityFilter.add(hash)) {
                return false;
            }

            int[] cloned = new int[256];
            System.arraycopy(localBoard, 0, cloned, 0, 256);
            gpuSeedBoards.add(cloned);

            int size = currentBatchSize.incrementAndGet();
            if (size % 1000 == 0) {
                System.out.println(timestamp() + "   [CPU Pool] Found " + size + " / " + activeBatch + " seeds...");
            }
            return false;
        }

        private void updateProgress(int step) {
            if (step > deepestStep) {
                synchronized (displayLock) {
                    if (step > deepestStep) {
                        deepestStep = step;

                        updateDisplay(countPieces(localBoard), buildDisplayBoard(localBoard));

                        if (deepestStep > absoluteHighScore) {
                            absoluteHighScore = deepestStep;

                            System.arraycopy(localBoard, 0, bestBoard, 0, 256);

                            RecordManager.saveRecord(buildDisplayBoard(bestBoard), absoluteHighScore, saveProfile);
                            CheckpointManager.saveRecordCheckpoint(buildDisplayBoard(bestBoard), absoluteHighScore,
                                    saveProfile);
                            System.out.println(timestamp() + ">>> NY ALL-TIME HIGH SCORE: " + absoluteHighScore + " " +
                                    "PIECES! <<<");
                        }
                    }
                }
            }
        }

        private List<Integer> getCandidatePool(int row, int col) {
            int b_req = (row == 0 || row == 15 ? 1 : 0) + (col == 0 || col == 15 ? 1 : 0);
            return (b_req == 2) ? inventory.corners : (b_req == 1) ? inventory.edges : inventory.interior;
        }

        private boolean passesLookahead(int p, int step, int row, int col, int idx) {
            // --- Check south neighbor ---
            if (row < 15 && localBoard[idx + 16] == -1) {
                int reqN = PieceUtils.getSouth(p);
                int reqE = (col == 15) ? 0 : (localBoard[idx + 17] != -1 ? PieceUtils.getWest(localBoard[idx + 17]) : CompatibilityIndex.WILDCARD);
                int reqW = (col == 0)  ? 0 : (localBoard[idx + 15] != -1 ? PieceUtils.getEast(localBoard[idx + 15]) : CompatibilityIndex.WILDCARD);
                int reqS = (row == 14) ? 0 : (localBoard[idx + 32] != -1 ? PieceUtils.getNorth(localBoard[idx + 32]) : CompatibilityIndex.WILDCARD);

                // 1. Hent alle gyldige kandidater til naboen direkte mod syd
                java.util.BitSet southCandidates = compatIndex.candidatesFor(reqN, reqE, reqS, reqW);
                compatIndex.andNotUsed(southCandidates, localUsed);

                if (southCandidates.isEmpty()) {
                    return false;
                }

                // =======================================================
                // NYT BITMASK SECONDARY LOOKAHEAD (Løser 41-brikkers muren!)
                // =======================================================
                if (step > 40 && row < 14 && localBoard[idx + 32] == -1) {
                    boolean secondaryFound = false;
                    // Tjek om bare én af syd-kandidaterne OGSÅ har en gyldig nabo under sig
                    for (int i = southCandidates.nextSetBit(0); i >= 0; i = southCandidates.nextSetBit(i + 1)) {
                        int testPiece = inventory.allOrientations[i];
                        int nextReqN = PieceUtils.getSouth(testPiece);

                        if (compatIndex.hasAnyCandidate(nextReqN, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, localUsed)) {
                            secondaryFound = true;
                            break; // Vi fandt en vej frem, afbryd søgningen (lynhurtigt!)
                        }
                    }
                    if (!secondaryFound) {
                        return false; // Død ende opdaget i række 2!
                    }
                }
                // =======================================================
            }

            // --- Check east neighbor ---
            if (col < 15 && localBoard[idx + 1] == -1) {
                int reqW = PieceUtils.getEast(p);
                int reqN = (row == 0)  ? 0 : (localBoard[idx - 15] != -1 ? PieceUtils.getSouth(localBoard[idx - 15]) : CompatibilityIndex.WILDCARD);
                int reqE = (col == 14) ? 0 : (localBoard[idx + 2]  != -1 ? PieceUtils.getWest(localBoard[idx + 2])   : CompatibilityIndex.WILDCARD);
                int reqS = (row == 15) ? 0 : (localBoard[idx + 17] != -1 ? PieceUtils.getNorth(localBoard[idx + 17]) : CompatibilityIndex.WILDCARD);

                if (!compatIndex.hasAnyCandidate(reqN, reqE, reqS, reqW, localUsed)) {
                    return false;
                }
            }
            return true;
        }

//        private boolean passesLookahead(int p, int step, int row, int col, int idx) {
//            if (row < 15 && localBoard[idx + 16] == -1) {
//                int reqN = PieceUtils.getSouth(p);
//                if (!hasAvailablePiece(piecesByNorth[reqN])) {
//                    return false;
//                }
//                if (step > 40 && row < 14 && localBoard[idx + 32] == -1 && !isSecondaryNeighborViable(reqN)) {
//                    return false;
//                }
//            }
//            if (col < 15 && localBoard[idx + 1] == -1) {
//                int reqW = PieceUtils.getEast(p);
//                return hasAvailablePiece(piecesByWest[reqW]);
//            }
//            return true;
//        }

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

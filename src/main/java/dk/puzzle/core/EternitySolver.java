package dk.puzzle.core;

import dk.puzzle.ai.SeedSelector;
import dk.puzzle.ai.SurgeonHeuristics;
import dk.puzzle.gpu.GpuEngine;
import dk.puzzle.io.BucasExporter;
import dk.puzzle.io.CheckpointManager;
import dk.puzzle.io.RecordManager;
import static dk.puzzle.io.RecordManager.uploadToDrive;
import dk.puzzle.model.CompatibilityIndex;
import dk.puzzle.model.PieceInventory;
import dk.puzzle.util.PieceUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The central orchestrator for the Eternity II puzzle solver, managing a multiphase
 * hybrid search strategy involving CPU-based seed generation and GPU-accelerated
 * deep search and repair.
 */
public class EternitySolver implements Runnable {

    public static final int SEED_DEPTH = 80;
    public static final int LNS_THRESHOLD = 200;

    private static final Logger logger = LogManager.getFormatterLogger(EternitySolver.class);

    // GUI configurable parameters
    public static volatile int userCpuHandoffDepth = 8;
    public static volatile int userSurgeonHoles = 20;

    // Core Solver Components
    private final PieceInventory inventory;
    private final CompatibilityIndex compatIndex;
    final SurgeonHeuristics surgeon;
    private GpuEngine gpuEngine;

    // Configuration Flags
    private boolean useGpu;
    private final boolean lockCenter;

    // Threading and Concurrency
    private final ExecutorService executor;
    private ExecutorService gpuExecutor; // Dedicated GPU thread for Context safety
    private final int numCores;
    private final int targetBatchSize = 50000;
    volatile int userBatchSizeOverride = -1;
    private final ConcurrentLinkedQueue<int[]> seedPool = new ConcurrentLinkedQueue<>();
    private final AtomicInteger currentBatchSize = new AtomicInteger(0);
    private int fatalGpuBugCount = 0;
    private final Object displayLock = new Object();

    // Board State
    private final int[] flatResumeBoard = new int[256];
    private final boolean[] usedPhysicalPieces = new boolean[256];
    final int[] tabuTenure = new int[256];
    final int[] buildOrder = new int[256];
    final int[] bestBoard = new int[256];
    final int[] globalBestBoard = new int[256];
    final TopBoardRegistry topBoards = new TopBoardRegistry();

    // Puzzle Specifics
    private final int targetPiece;
    private int centerPhysicalIdx = -1;

    // Solver Profile
    private final String saveProfile;
    private final BuildStrategy currentStrategy;

    // Metrics and Progress Tracking
    private final AtomicLong globalCpuTrialCount = new AtomicLong(0);
    private final AtomicLong globalGpuTrialCount = new AtomicLong(0);
    private final Set<Integer> uniqueMaxScoreHashes = ConcurrentHashMap.newKeySet();
    private volatile int currentSeedDepth = SEED_DEPTH;
    volatile int absoluteHighScore = 0;
    volatile int deepestStep = 0;
    private int currentRepairIteration = 0;
    private int consecutiveExtinctions = 0;
    volatile boolean manualOverrideRequested = false;
    volatile int manualBaseCampTarget = 0;
    volatile double extinctionThreshold = 0.98;
    private volatile int lastReportedDepth = 0;
    private int consecutiveGpuStagnation = 0;
    private int phaseCounter = 0; // For GPU Scheduling
    private long lastThroughputReportTime = System.currentTimeMillis();
    private long totalRepairVariationsTested = 0;
    private long repairLoopsCounter = 0;
    private volatile boolean isGpuBusy = false;
    volatile int stagnationLimitMinutes = 20;
    private volatile int poisonedIndex = -1;
    private volatile int poisonedPiece = -1;
    private ScheduledExecutorService repairReporterScheduler;

    // HINT STRATEGY FIELDS
    private static final int[] HINT_POSITIONS = {221, 45, 210, 34};
    private static final int[] HINT_PHYSICAL_NUMBERS = {249, 181, 255, 208};
    private static final int[] HINT_ROTATIONS = {1, 0, 0, 0};
    private final int[] hintPackedValues = new int[4];
    private final int[] hintPhysicalIndices = new int[4];

    private int findPackedPiece(int physicalNumber, int targetRotation) {
        int physIdx = physicalNumber - 1;
        int foundCount = 0;
        for (int oi = 0; oi < 1024; oi++) {
            if (inventory.physicalMapping[oi] == physIdx) {
                if (foundCount == targetRotation) {
                    return inventory.allOrientations[oi];
                }
                foundCount++;
            }
        }
        return -1;
    }

    public EternitySolver(PieceInventory inventory, int trueCenterPiece, boolean useGpu, BuildStrategy strategy, boolean lockCenter) {
        Arrays.fill(bestBoard, -1);
        Arrays.fill(globalBestBoard, -1);
        Arrays.fill(flatResumeBoard, -1);

        this.inventory = inventory;
        this.targetPiece = trueCenterPiece;
        this.currentStrategy = strategy;
        this.lockCenter = lockCenter;
        this.useGpu = useGpu;

        this.saveProfile = strategy.name() + (lockCenter ? "_LOCKED" : "_UNLOCKED");
        this.numCores = Runtime.getRuntime().availableProcessors();
        this.executor = Executors.newFixedThreadPool(numCores);
        logger.info(">>> Multithreading active with " + numCores + " cores.");

        for (int i = 0; i < 1024; i++) {
            if (inventory.allOrientations[i] == targetPiece) {
                this.centerPhysicalIdx = inventory.physicalMapping[i];
                break;
            }
        }

        if (strategy == BuildStrategy.SPIRAL) {
            generateSpiralOrder();
        } else {
            for (int i = 0; i < 256; i++) {
                buildOrder[i] = i;
            }
        }

        this.compatIndex = new CompatibilityIndex(inventory.allOrientations, inventory.physicalMapping);
        this.surgeon = new SurgeonHeuristics(lockCenter, 0.70);

        loadCheckpointAndHints();
    }

    private void loadCheckpointAndHints() {
        SolverState loadedState = CheckpointManager.loadSmartState(saveProfile);

        if (loadedState != null) {
            restoreBoardState(loadedState.bestBoard);
            System.arraycopy(loadedState.tabuTenure, 0, this.tabuTenure, 0, 256);
            this.uniqueMaxScoreHashes.addAll(loadedState.uniqueMaxScoreHashes);
            for (int[] historicBoard : loadedState.topBoardsRegistry) {
                this.topBoards.offer(historicBoard, loadedState.score);
            }
            logger.info(">>> SUCCESS: Loaded checkpoint AND restored historic solver memory!");
        } else {
            int[][] loaded = CheckpointManager.loadSmartCheckpoint(saveProfile);
            if (loaded != null) {
                restoreBoardState(loaded);
                logger.info(">>> SUCCESS: Loaded legacy checkpoint fully!");
            } else {
                logger.info(">>> [WARNING] Smart Load failed and returned null. Starting from scratch.");
            }
        }

        // RESOLVE AND PRE-LOCK HINTS
        for (int h = 0; h < 4; h++) {
            int packed = findPackedPiece(HINT_PHYSICAL_NUMBERS[h], HINT_ROTATIONS[h]);
            hintPackedValues[h] = packed;
            hintPhysicalIndices[h] = HINT_PHYSICAL_NUMBERS[h] - 1;

            if (packed != -1) {
                bestBoard[HINT_POSITIONS[h]] = packed;
                globalBestBoard[HINT_POSITIONS[h]] = packed;
                flatResumeBoard[HINT_POSITIONS[h]] = packed;
                logger.info(">>> [HINTS] Hint piece #%d (physical #%d) locked at position %d", h, HINT_PHYSICAL_NUMBERS[h], HINT_POSITIONS[h]);
            }
        }
        if (lockCenter) flatResumeBoard[135] = targetPiece;
    }

    private void restoreBoardState(int[][] loaded) {
        if (loaded == null) return;
        int loadedCount = 0;
        for (int r = 0; r < 16; r++) {
            if (loaded[r] == null) continue;
            for (int c = 0; c < 16; c++) {
                int p = loaded[r][c];
                if (p != -1 && p != 0 && p != -2) {
                    bestBoard[r * 16 + c] = p;
                    globalBestBoard[r * 16 + c] = p;
                    loadedCount++;
                }
            }
        }
        if (loadedCount > 0) {
            this.absoluteHighScore = loadedCount;
            this.deepestStep = loadedCount;
            if (lockCenter) {
                bestBoard[135] = targetPiece;
                globalBestBoard[135] = targetPiece;
            }
            System.arraycopy(bestBoard, 0, flatResumeBoard, 0, 256);
            updateDisplay(absoluteHighScore, buildDisplayBoard(globalBestBoard));
        }
    }

    public void setStagnationLimit(int minutes) { this.stagnationLimitMinutes = minutes; }
    public void setBatchSizeOverride(int size) { this.userBatchSizeOverride = size; }
    public void setExtinctionThreshold(double threshold) { this.extinctionThreshold = threshold; }
    public void setTargetedHolesPercentage(double percentage) {
        if (this.surgeon != null) this.surgeon.setTargetedHolesPercentage(percentage);
    }
    public void triggerManualOverride(int targetBaseCamp) {
        this.manualBaseCampTarget = targetBaseCamp;
        this.manualOverrideRequested = true;
    }

    @Override
    public void run() {
        initializeGpuEngine();
        setupShutdownHook();
        startReporterThread();

        logger.info("Starting Solver Orchestrator...");

        if (this.absoluteHighScore > 0) {
            logger.info(">>> [BOOT] Checkpoint detected! Setting up Base Camp to resume search...");
            triggerBranchScrap();
        }

        run3PhasePipeline();
    }

    private int getDynamicBatchSize() {
        if (deepestStep >= 200) return 250;
        if (deepestStep >= 180) return 1000;
        if (deepestStep >= 100) return 5000;
        if (deepestStep < 50) return 5000;
        return 15000;
    }

    private void initializeGpuEngine() {
        if (this.useGpu) {
            this.gpuExecutor = Executors.newSingleThreadExecutor();
            try {
                this.gpuEngine = this.gpuExecutor.submit(() -> new GpuEngine(inventory, lockCenter)).get();
                logger.info(">>> [HARDWARE] NVIDIA CUDA GPU detected and initialized successfully on dedicated thread.");
            } catch (Throwable t) {
                this.useGpu = false;
                this.gpuEngine = null;
                if (this.gpuExecutor != null) this.gpuExecutor.shutdownNow();
                logger.warn(">>> [HARDWARE] GPU initialization failed: " + t.getMessage());
            }
        }
    }

    private void rebootGpuEngine() {
        logger.warn(">>> [HARDWARE] Rebooting GPU Engine to escape kernel deadlock...");

        if (this.useGpu) {
            try {
                // 1. The Orchestrator thread MUST claim the device before resetting it!
                jcuda.runtime.JCuda.cudaSetDevice(0);

                // 2. Nuke the VRAM memory and destroy the deadlocked context
                jcuda.runtime.JCuda.cudaDeviceReset();
            } catch (Exception e) {
                logger.warn(">>> [HARDWARE] Failed to forcefully reset CUDA device: " + e.getMessage());
            }
        }

        if (this.gpuExecutor != null) {
            this.gpuExecutor.shutdownNow();
        }

        // 3. Hardware Buffer: Give the NVIDIA driver 500ms to physically flush the VRAM
        try {
            Thread.sleep(500);
        } catch (InterruptedException ignored) {}

        // 4. Force Java to delete the old GpuEngine object from RAM
        System.gc();

        // 5. Spin up a fresh, clean GPU Engine
        initializeGpuEngine();
    }

    private void setupShutdownHook() {
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            logger.info(">>> Shutdown hook: Saving final checkpoint...");
            synchronized (displayLock) {
                if (repairReporterScheduler != null) repairReporterScheduler.shutdownNow();
                SolverState memoryToSave = new SolverState(
                        buildDisplayBoard(globalBestBoard),
                        absoluteHighScore,
                        this.tabuTenure,
                        this.uniqueMaxScoreHashes,
                        this.topBoards.getRawRegistry()
                );
                CheckpointManager.saveSmartState(memoryToSave, saveProfile);
            }
        }));
    }

    private void startReporterThread() {
        Thread reporterThread = new Thread(() -> {
            while (true) {
                try {
                    Thread.sleep(5000);
                } catch (InterruptedException ignored) {
                    Thread.currentThread().interrupt();
                    return;
                }
                reportSpeed();
            }
        });
        reporterThread.setDaemon(true);
        reporterThread.start();
    }

    private void run3PhasePipeline() {
        logger.info(">>> Running 3-Phase Pipeline Orchestrator...");
        long lastPeriodicSave = System.currentTimeMillis();

        while (true) {
            try {
                if (manualOverrideRequested) {
                    manualOverrideRequested = false;
                    retreat(manualBaseCampTarget, ">>> User Override...");
                    try { Thread.sleep(1000); } catch (InterruptedException ignored) {}
                }

                int activeBatch = getDynamicBatchSize();
                if (userBatchSizeOverride > 0) activeBatch = userBatchSizeOverride;

                if (this.useGpu && this.gpuEngine != null) {
                    if (deepestStep >= LNS_THRESHOLD) {
                        phaseCounter++;
                        boolean seedsReady = seedPool.size() >= activeBatch;

                        // Claude's Phase Scheduler: Force Deep DFS every 5th round!
                        if (phaseCounter % 4 == 0 && seedsReady) {
                            logger.info(">>> [SCHEDULER] Phase 2 refresh (round %d of 5)", phaseCounter);
                            runPhase2_GpuDeepDfs();
                        } else if (!seedsReady) {
                            runPhase1_CpuSeedGen();
                        } else {
                            runPhase3_GpuSurgeon();
                        }
                    } else if (seedPool.size() >= activeBatch) {
                        runPhase2_GpuDeepDfs();
                    } else {
                        runPhase1_CpuSeedGen();
                    }
                } else {
                    if (deepestStep >= LNS_THRESHOLD) {
                        logger.info(">>> [CPU MODE] Skipping Surgeon mode (GPU only). Triggering CPU Teardown...");
                        consecutiveExtinctions += 15;
                        triggerBranchScrap();
                    } else {
                        runPhase1_CpuSeedGen();
                    }
                }

                if (System.currentTimeMillis() - lastPeriodicSave > 300_000) {
                    synchronized (displayLock) {
                        SolverState memoryToSave = new SolverState(
                                buildDisplayBoard(globalBestBoard),
                                absoluteHighScore,
                                this.tabuTenure,
                                this.uniqueMaxScoreHashes,
                                this.topBoards.getRawRegistry()
                        );
                        CheckpointManager.saveSmartState(memoryToSave, saveProfile);
                    }
                    lastPeriodicSave = System.currentTimeMillis();
                }
            } catch (Exception e) {
                logger.error(">>> [FATAL ERROR] PIPELINE CRASHED: ", e);
                try { Thread.sleep(5000); } catch (InterruptedException ignored) {}
            }
        }
    }

    private void runPhase1_CpuSeedGen() {
        Arrays.fill(usedPhysicalPieces, false);

        int lockedPieces = 0;
        for (int p : flatResumeBoard) {
            if (p != -1 && p != -2) lockedPieces++;
        }

        // --- FAST GPU HANDOFF LOGIC ---
        if (lockedPieces == 0) {
            currentSeedDepth = 14;
            logger.info(">>> [PHASE 1] Empty board. Setting fast GPU handoff depth to: " + currentSeedDepth);
        } else if (lockedPieces < SEED_DEPTH) {
            currentSeedDepth = lockedPieces + userCpuHandoffDepth;
        } else {
            currentSeedDepth = Math.max(SEED_DEPTH, lockedPieces + userCpuHandoffDepth);
        }

        currentBatchSize.set(seedPool.size());

        try {
            int activeBatch = getDynamicBatchSize();
            if (userBatchSizeOverride > 0) activeBatch = userBatchSizeOverride;

            List<CpuSearchWorker> workers = new ArrayList<>();
            for (int i = 0; i < numCores; i++) {
                workers.add(new CpuSearchWorker(activeBatch));
            }

            List<Future<Boolean>> futures = executor.invokeAll(workers);

            for (Future<Boolean> f : futures) {
                try { f.get(); } catch (ExecutionException e) {}
            }

            if (this.useGpu && seedPool.isEmpty()) {
                logger.warn(">>> [PHASE 1 DEADLOCK] CPU proved that NO alternative seeds exist for this Base Camp!");
                consecutiveExtinctions++;
                triggerBranchScrap();
            } else if (!this.useGpu) {
                logger.warn(">>> [CPU DEADLOCK] Exhausted branch without finding a solution. Tearing down...");
                consecutiveExtinctions++;
                triggerBranchScrap();
            } else {
                if (poisonedIndex != -1) {
                    logger.info(">>> [TABU] CPU successfully bypassed the dead end. Global Tabu lifted.");
                    poisonedIndex = -1;
                    poisonedPiece = -1;
                }
            }
        } catch (InterruptedException e) {}
    }

    private void runPhase2_GpuDeepDfs() {
        int activeBatch = getDynamicBatchSize();
        if (userBatchSizeOverride > 0) activeBatch = userBatchSizeOverride;

        List<int[]> seeds = new ArrayList<>();
        for (int i = 0; i < activeBatch; i++) {
            int[] s = seedPool.poll();
            if (s != null) seeds.add(s);
        }
        if (seeds.isEmpty()) return;

        int scoreBefore = deepestStep;
        int[] bestBoardOut = new int[256];
        isGpuBusy = true;
        long start = System.currentTimeMillis();
        GpuEngine.GpuResult result = null;

        try {
            Future<GpuEngine.GpuResult> future = gpuExecutor.submit(() ->
                    gpuEngine.runDeepDfs(seeds, currentSeedDepth, deepestStep, bestBoardOut, buildOrder)
            );
            result = future.get(10, java.util.concurrent.TimeUnit.SECONDS);
        } catch (java.util.concurrent.TimeoutException e) {
            logger.warn(">>> [GPU TIMEOUT] Phase 2 Kernel locked up! Triggering structural teardown and rebooting GPU...");
            isGpuBusy = false;
            rebootGpuEngine();
            triggerBranchScrap();
            return;
        } catch (Exception e) {
            logger.error(">>> [GPU ERROR] Pipeline crashed during Phase 2: " + e.getMessage());
            isGpuBusy = false;
            rebootGpuEngine();
            triggerBranchScrap();
            return;
        }

        if (result.newHighScore() > lastReportedDepth) {
            logger.info(">>> [CLIMBING] Current pieces placed: %d / 256", result.newHighScore());
            lastReportedDepth = result.newHighScore();
        }
        globalGpuTrialCount.addAndGet(result.stepsTaken());
        isGpuBusy = false;

        long elapsed = System.currentTimeMillis() - start;
        logger.info(">>> GPU Phase 2 complete. Steps taken per second: %,d", Math.round((double) result.stepsTaken() * 1000) / Math.max(1, elapsed));

        if (result.solved()) handleVictory(bestBoardOut);

        if (result.newHighScore() > scoreBefore) {
            if (!verifyBoardStrict(bestBoardOut)) {
                logger.error(">>> [FATAL GPU BUG] The GPU returned a board with an illegal edge conflict!");
                int piecesPlaced = countPieces(bestBoardOut);

                if (piecesPlaced >= 240) {
                    logger.info(">>> [RESCUE] High-score board (" + piecesPlaced + ") detected. Executing emergency surgery...");
                    int validPieces = rescueBoard(bestBoardOut);
                    if (validPieces > deepestStep) {
                        deepestStep = validPieces;
                        System.arraycopy(bestBoardOut, 0, bestBoard, 0, 256);
                        topBoards.offer(bestBoardOut, deepestStep);
                        updateDisplay(deepestStep, buildDisplayBoard(bestBoard));
                    }
                } else {
                    fatalGpuBugCount++;
                    if (fatalGpuBugCount > 50) {
                        triggerBranchScrap();
                        fatalGpuBugCount = 0;
                    }
                }
                return;
            }
            deepestStep = result.newHighScore();
            consecutiveGpuStagnation = 0;

            System.arraycopy(bestBoardOut, 0, bestBoard, 0, 256);
            topBoards.offer(bestBoardOut, deepestStep);
            updateDisplay(deepestStep, buildDisplayBoard(bestBoard));

            if (deepestStep > absoluteHighScore) {
                absoluteHighScore = deepestStep;
                System.arraycopy(bestBoardOut, 0, globalBestBoard, 0, 256);
                RecordManager.saveRecord(buildDisplayBoard(globalBestBoard), absoluteHighScore, saveProfile);
                consecutiveExtinctions = 0;
                uniqueMaxScoreHashes.clear();
                uniqueMaxScoreHashes.add(Arrays.hashCode(bestBoardOut));
                saveAndUploadBucasLink(globalBestBoard, absoluteHighScore);
                analyzeFullBoardPotential(globalBestBoard);

            } else if (deepestStep == absoluteHighScore) {
                if (uniqueMaxScoreHashes.add(Arrays.hashCode(bestBoardOut))) {
                    analyzeFullBoardPotential(bestBoardOut);
                }
            }
        } else {
            consecutiveGpuStagnation++;
        }

        if (consecutiveGpuStagnation < 10) {
            List<int[]> nextSeeds = SeedSelector.selectBest(
                    seeds, result.threadDepths(), getDynamicBatchSize(), new Random());
            seedPool.addAll(nextSeeds);
            currentBatchSize.set(seedPool.size());
        } else {
            logger.warn(">>> [!!!] Phase 2 GPU stagnated! Trapped below LNS threshold. Activating Teardown...");
            triggerBranchScrap();
        }
    }

    private void runPhase3_GpuSurgeon() {
        int numClones = 50000;
        int holesToPunch = userSurgeonHoles;

        currentRepairIteration++;
        repairLoopsCounter++;
        totalRepairVariationsTested += numClones;

        int[] sourceBoard = topBoards.nextForRepair();
        if (sourceBoard == null) sourceBoard = bestBoard;

        int actualHoles = (consecutiveExtinctions > 20) ? ((deepestStep > 250) ? 5 : Math.round(holesToPunch / 2.5f)) : ((deepestStep > 200) ? 5 : holesToPunch);
        List<int[]> variations = surgeon.excavateFrontier(
                sourceBoard, numClones, actualHoles, tabuTenure, currentRepairIteration, deepestStep, buildOrder);

        int scoreBefore = deepestStep;
        int[] bestBoardOut = new int[256];
        GpuEngine.GpuResult result = null;

        try {
            Future<GpuEngine.GpuResult> future = gpuExecutor.submit(() ->
                    gpuEngine.runRepairMode(variations, deepestStep, bestBoardOut)
            );
            result = future.get(10, java.util.concurrent.TimeUnit.SECONDS);
        } catch (java.util.concurrent.TimeoutException e) {
            logger.warn(">>> [GPU TIMEOUT] Phase 3 Surgeon locked up! Triggering teardown and rebooting GPU...");
            rebootGpuEngine();
            triggerBranchScrap();
            return;
        } catch (Exception e) {
            logger.error(">>> [GPU ERROR] Pipeline crashed during Phase 3: " + e.getMessage());
            rebootGpuEngine();
            triggerBranchScrap();
            return;
        }

        if (result.newHighScore() > lastReportedDepth) {
            logger.info(">>> [CLIMBING] Current pieces placed: %d / 256", result.newHighScore());
            lastReportedDepth = result.newHighScore();
        }
        globalGpuTrialCount.addAndGet(result.stepsTaken());

        if (result.solved()) handleVictory(bestBoardOut);

        if (result.newHighScore() > scoreBefore) {
            if (!verifyBoardStrict(bestBoardOut)) {
                int piecesPlaced = countPieces(bestBoardOut);
                if (piecesPlaced >= 240) {
                    int validPieces = rescueBoard(bestBoardOut);
                    if (validPieces > deepestStep) {
                        deepestStep = validPieces;
                        System.arraycopy(bestBoardOut, 0, bestBoard, 0, 256);
                        topBoards.offer(bestBoardOut, deepestStep);
                        updateDisplay(deepestStep, buildDisplayBoard(bestBoard));
                    }
                } else {
                    fatalGpuBugCount++;
                    if (fatalGpuBugCount > 50) {
                        triggerBranchScrap();
                        fatalGpuBugCount = 0;
                    }
                }
                return;
            }
            deepestStep = result.newHighScore();
            updateTabuList(bestBoardOut);
            System.arraycopy(bestBoardOut, 0, bestBoard, 0, 256);
            topBoards.offer(bestBoardOut, deepestStep);
            updateDisplay(deepestStep, buildDisplayBoard(bestBoard));

            if (deepestStep > absoluteHighScore) {
                absoluteHighScore = deepestStep;
                System.arraycopy(bestBoardOut, 0, globalBestBoard, 0, 256);
                RecordManager.saveRecord(buildDisplayBoard(globalBestBoard), absoluteHighScore, saveProfile);
                consecutiveExtinctions = 0;
                saveAndUploadBucasLink(globalBestBoard, absoluteHighScore);
                analyzeFullBoardPotential(globalBestBoard);
            } else if (deepestStep == absoluteHighScore) {
                if (uniqueMaxScoreHashes.add(Arrays.hashCode(bestBoardOut))) {
                    analyzeFullBoardPotential(bestBoardOut);
                } else {
                    consecutiveExtinctions++;
                    updateTabuList(bestBoardOut);
                }
            } else {
                consecutiveExtinctions++;
                updateTabuList(bestBoardOut);
            }
        } else {
            consecutiveExtinctions++;
            if (consecutiveExtinctions >= 20) triggerBranchScrap();
        }
    }

    void updateTabuList(int[] newBoard) {
        int tenureLength = 5 + (absoluteHighScore / 25);
        for (int i = 0; i < 256; i++) {
            if (i == 135 || i == 221 || i == 45 || i == 210 || i == 34) {
                tabuTenure[i] = Integer.MAX_VALUE;
                continue;
            }
            if (newBoard[i] != bestBoard[i] && newBoard[i] != -1) {
                tabuTenure[i] = currentRepairIteration + tenureLength;
            }
        }
    }

    void clearTabu() {
        Arrays.fill(tabuTenure, 0);
    }

    private void triggerBranchScrap() {
        consecutiveExtinctions++;
        String phaseName = this.useGpu ? "PHASE 3 (GPU)" : "CPU DEEP SEARCH";
        logger.info(">>> [!!!] DEAD END AT %s (Stagnation level: %d) [!!!]", phaseName, consecutiveExtinctions);

        int minRetreat = 10 + (consecutiveExtinctions * 5);
        int maxRetreat = minRetreat + 15;
        int retreatAmount;

        if (consecutiveExtinctions > 25 || absoluteHighScore == 0) {
            retreatAmount = absoluteHighScore;
        } else {
            Random rand = new Random();
            retreatAmount = minRetreat + rand.nextInt(maxRetreat - minRetreat + 1);
        }

        int lockedPieces = Math.max(0, absoluteHighScore - retreatAmount);

        if (consecutiveExtinctions > 25 || absoluteHighScore == 0) {
            lockedPieces = 0;
            retreatAmount = absoluteHighScore;
            consecutiveExtinctions = 0;
        }

        lastReportedDepth = lockedPieces;

        if (lockedPieces == 0) {
            poisonedIndex = buildOrder[0];
            poisonedPiece = globalBestBoard[poisonedIndex];

            if (poisonedPiece != -1) {
                int physId = -1;
                for (int i = 0; i < 1024; i++) {
                    if (inventory.allOrientations[i] == poisonedPiece) {
                        physId = inventory.physicalMapping[i] + 1;
                        break;
                    }
                }
                logger.info(">>> [GLOBAL TABU] Board reset! Banning physical piece #%d (packed: %d) at index %d.", physId, poisonedPiece, poisonedIndex);
            }
        } else if (absoluteHighScore > lockedPieces + 5) {
            poisonedIndex = buildOrder[lockedPieces + 2];
            poisonedPiece = globalBestBoard[poisonedIndex];
            logger.info(">>> [GLOBAL TABU] Poisoned Square Active! Piece %d is banned at index %d.", poisonedPiece, poisonedIndex);
        } else {
            poisonedIndex = -1;
            poisonedPiece = -1;
        }

        Arrays.fill(flatResumeBoard, -1);
        for (int step = 0; step < lockedPieces; step++) {
            int idx = buildOrder[step];
            if (lockCenter && idx == 135) continue;
            flatResumeBoard[idx] = globalBestBoard[idx];
        }

        deepestStep = Math.max(0, lockedPieces);
        seedPool.clear();
        currentBatchSize.set(0);
        consecutiveGpuStagnation = 0;
        phaseCounter = 0; // RESET PHASE SCHEDULER
        clearTabu();
    }

    private void analyzeFullBoardPotential(int[] recordBoard) {
        int[] simulatedBoard = Arrays.copyOf(recordBoard, 256);
        List<Integer> emptySpots = new ArrayList<>();
        boolean[] usedPhysical = new boolean[256];

        for (int i = 0; i < 256; i++) {
            int p = simulatedBoard[i];
            if (p != -1 && p != -2) {
                int physicalId = -1;
                for (int oi = 0; oi < 1024; oi++) {
                    if (inventory.allOrientations[oi] == p) {
                        physicalId = inventory.physicalMapping[oi];
                        break;
                    }
                }
                if (physicalId != -1) usedPhysical[physicalId] = true;
            } else {
                emptySpots.add(i);
            }
        }

        List<Integer> unusedPhysIds = new ArrayList<>();
        for (int physId = 0; physId < 256; physId++) {
            if (!usedPhysical[physId]) unusedPhysIds.add(physId);
        }

        Collections.shuffle(unusedPhysIds);

        for (int spot : emptySpots) {
            int row = spot / 16;
            int col = spot % 16;
            boolean atNorthEdge = (row == 0);
            boolean atSouthEdge = (row == 15);
            boolean atWestEdge = (col == 0);
            boolean atEastEdge = (col == 15);

            int bestBrikIndex = -1;
            int bestOrientedPiece = -1;

            for (int i = 0; i < unusedPhysIds.size(); i++) {
                int physId = unusedPhysIds.get(i);
                for (int oi = 0; oi < 1024; oi++) {
                    if (inventory.physicalMapping[oi] == physId) {
                        int p = inventory.allOrientations[oi];
                        boolean match = true;
                        if (atNorthEdge && PieceUtils.getNorth(p) != 0) match = false;
                        if (atSouthEdge && PieceUtils.getSouth(p) != 0) match = false;
                        if (atWestEdge && PieceUtils.getWest(p) != 0) match = false;
                        if (atEastEdge && PieceUtils.getEast(p) != 0) match = false;
                        if (match && row > 0 && simulatedBoard[spot - 16] != -1) {
                            if (PieceUtils.getNorth(p) != PieceUtils.getSouth(simulatedBoard[spot - 16])) match = false;
                        }
                        if (match && col > 0 && simulatedBoard[spot - 1] != -1) {
                            if (PieceUtils.getWest(p) != PieceUtils.getEast(simulatedBoard[spot - 1])) match = false;
                        }
                        if (match) {
                            bestBrikIndex = i;
                            bestOrientedPiece = p;
                            break;
                        }
                    }
                }
                if (bestBrikIndex != -1) break;
            }

            // PASS 2: Fallback to EDGE ONLY match (Ignore internal neighbors, just keep borders flat)
            if (bestBrikIndex == -1) {
                for (int i = 0; i < unusedPhysIds.size(); i++) {
                    int physId = unusedPhysIds.get(i);
                    for (int oi = 0; oi < 1024; oi++) {
                        if (inventory.physicalMapping[oi] == physId) {
                            int p = inventory.allOrientations[oi];
                            boolean match = true;
                            if (atNorthEdge && PieceUtils.getNorth(p) != 0) match = false;
                            if (atSouthEdge && PieceUtils.getSouth(p) != 0) match = false;
                            if (atWestEdge && PieceUtils.getWest(p) != 0) match = false;
                            if (atEastEdge && PieceUtils.getEast(p) != 0) match = false;

                            if (match) {
                                bestBrikIndex = i;
                                bestOrientedPiece = p;
                                break;
                            }
                        }
                    }
                    if (bestBrikIndex != -1) break;
                }
            }

            // PASS 3: Ultimate Fallback - Just place anything so there isn't a hole!
            if (bestBrikIndex != -1) {
                simulatedBoard[spot] = bestOrientedPiece;
                unusedPhysIds.remove(bestBrikIndex);
            } else {
                int fallbackPhysId = unusedPhysIds.remove(0);
                for (int oi = 0; oi < 1024; oi++) {
                    if (inventory.physicalMapping[oi] == fallbackPhysId) {
                        simulatedBoard[spot] = inventory.allOrientations[oi];
                        break;
                    }
                }
            }
        }

        int totalConflicts = 0;
        for (int row = 0; row < 16; row++) {
            for (int col = 0; col < 16; col++) {
                int idx = row * 16 + col;
                int currentPiece = simulatedBoard[idx];
                if (currentPiece == -1) continue;
                if (col < 15) {
                    int eastNeighbor = simulatedBoard[idx + 1];
                    if (eastNeighbor != -1 && PieceUtils.getEast(currentPiece) != PieceUtils.getWest(eastNeighbor)) totalConflicts++;
                }
                if (row < 15) {
                    int southNeighbor = simulatedBoard[idx + 16];
                    if (southNeighbor != -1 && PieceUtils.getSouth(currentPiece) != PieceUtils.getNorth(southNeighbor)) totalConflicts++;
                }
            }
        }

        logger.info(">>> [FULL BOARD SCAN] Simulated an edge-aware fully laid board. Total internal edge conflicts: %d / 480", totalConflicts);
        if (totalConflicts < 30) logger.warn(">>> [!!!] WOW! You are mathematically incredibly close to a full solution!");

        saveFullBoardVariant(simulatedBoard, absoluteHighScore, totalConflicts);
    }

    private void handleVictory(int[] winningBoard) {
        logger.info(">>> ETERNITY II SOLVED BY GPU PIPELINE!!! <<<");
        updateDisplay(256, buildDisplayBoard(winningBoard));
        RecordManager.saveRecord(buildDisplayBoard(winningBoard), 256, saveProfile);
        consecutiveExtinctions = 0;
        saveAndUploadBucasLink(winningBoard, 256);
        System.exit(0);
    }

    private void reportSpeed() {
        if (isGpuBusy) return;
        long now = System.currentTimeMillis();
        long elapsed = now - lastThroughputReportTime;
        if (elapsed < 2000) return;

        long cpuTrials = globalCpuTrialCount.getAndSet(0);
        long gpuTrials = globalGpuTrialCount.getAndSet(0);
        double cpuTps = cpuTrials / (elapsed / 1000.0);
        double gpuTps = gpuTrials / (elapsed / 1000.0);

        if (cpuTps != 0 || gpuTps != 0) {
            logger.info("[SPEED] CPU Phase 1: %,.0f/s  |  GPU Phase 2/3: %,.0f/s", cpuTps, gpuTps);
        }
        lastThroughputReportTime = now;
    }

    void retreat(int targetStep, String logMessage) {
        deepestStep = Math.max(0, targetStep);
        for (int s = deepestStep; s < 256; s++) {
            bestBoard[buildOrder[s]] = -1;
        }
        if (lockCenter) bestBoard[135] = targetPiece;
        for (int h = 0; h < 4; h++) {
            if (hintPackedValues[h] != -1) bestBoard[HINT_POSITIONS[h]] = hintPackedValues[h];
        }
        updateDisplay(countPieces(bestBoard), buildDisplayBoard(bestBoard));
        if (logMessage != null) logger.info(logMessage);
    }

    int countPieces(int[] board) {
        int count = 0;
        for (int p : board) {
            if (p != -1 && p != -2) count++;
        }
        return count;
    }

    int[][] buildDisplayBoard(int[] sourceArray) {
        int[][] displayBoard = new int[16][16];
        for (int i = 0; i < 16; i++) Arrays.fill(displayBoard[i], -1);
        for (int i = 0; i < 256; i++) {
            if (sourceArray[i] == -1) continue;
            displayBoard[i / 16][i % 16] = sourceArray[i];
        }
        return displayBoard;
    }

    private void updateDisplay(int score, int[][] displayBoard) {
        Eternity.updateDisplay(score, this.absoluteHighScore, displayBoard);
    }

    private void saveAndUploadBucasLink(int[] board, int score) {
        try {
            String bucasLink = BucasExporter.exportBoard(board);
            java.io.File folder = new java.io.File(saveProfile);
            if (!folder.exists()) folder.mkdirs();
            java.io.File linkFile = new java.io.File(folder, "bucas_link_" + score + ".txt");
            try (java.io.FileWriter writer = new java.io.FileWriter(linkFile)) {
                writer.write("Eternity II Record: " + score + " pieces\n");
                writer.write("Time: " + LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss")) + "\n");
                writer.write("Strategy: " + saveProfile + "\n\n");
                writer.write(bucasLink + "\n");
            }
            uploadToDrive(linkFile, "text/plain", saveProfile);
        } catch (Exception e) {}
    }

    private void saveFullBoardVariant(int[] simulatedBoard, int baseScore, int conflicts) {
        String fullProfileFolder = saveProfile + "_FULL";
        java.io.File folder = new java.io.File(fullProfileFolder);
        if (!folder.exists()) folder.mkdirs();

        String timeId = java.time.LocalTime.now().format(java.time.format.DateTimeFormatter.ofPattern("HHmmss_SSS"));
        String baseFilename = "Errors" + conflicts + "_Base" + baseScore + "_" + timeId;

        try (java.io.PrintWriter writer = new java.io.PrintWriter(new java.io.File(folder, baseFilename + ".csv"))) {
            for (int row = 0; row < 16; row++) {
                StringBuilder line = new StringBuilder();
                for (int col = 0; col < 16; col++) {
                    int p = simulatedBoard[row * 16 + col];
                    int physId = -1;
                    for (int oi = 0; oi < 1024; oi++) {
                        if (inventory.allOrientations[oi] == p) {
                            physId = inventory.physicalMapping[oi];
                            break;
                        }
                    }
                    line.append(physId + 1);
                    if (col < 15) line.append(",");
                }
                writer.println(line);
            }
        } catch (Exception e) {
            logger.error(">>> Error saving Full Board CSV: " + e.getMessage());
        }

        try (java.io.PrintWriter writer = new java.io.PrintWriter(new java.io.File(folder, baseFilename + "_link.txt"))) {
            writer.println("Base Score: " + baseScore);
            writer.println("Edge Conflicts: " + conflicts);
            writer.println(BucasExporter.exportBoard(simulatedBoard));
            logger.info(">>> Saved full board Bucas link: " + baseFilename + "_link.txt");
        } catch (Exception e) {
            logger.error(">>> Error saving Full Board Bucas Link: " + e.getMessage());
        }

        // SAFELY Save PNG Image
        try {
            RecordManager.saveImage(buildDisplayBoard(simulatedBoard), new java.io.File(folder, baseFilename + ".png").getAbsolutePath());
            logger.info(">>> Saved full board PNG image: " + baseFilename + ".png");
        } catch (Exception e) {
            logger.error(">>> Error saving PNG image: " + e.getMessage());
        }
    }

    boolean verifyBoardStrict(int[] board) {
        for (int row = 0; row < 16; row++) {
            for (int col = 0; col < 16; col++) {
                int idx = row * 16 + col;
                int p = board[idx];
                if (p == -1 || p == -2) continue;
                if (col < 15) {
                    int eastP = board[idx + 1];
                    if (eastP != -1 && eastP != -2) {
                        if (PieceUtils.getEast(p) != PieceUtils.getWest(eastP)) return false;
                    }
                }
                if (row < 15) {
                    int southP = board[idx + 16];
                    if (southP != -1 && southP != -2) {
                        if (PieceUtils.getSouth(p) != PieceUtils.getNorth(southP)) return false;
                    }
                }
            }
        }
        return true;
    }

    private int rescueBoard(int[] corruptedBoard) {
        int[] rescuedBoard = Arrays.copyOf(corruptedBoard, 256);
        for (int row = 0; row < 16; row++) {
            for (int col = 0; col < 16; col++) {
                int idx = row * 16 + col;
                int p = rescuedBoard[idx];
                if (p == -1 || p == -2) continue;

                boolean conflict = false;
                if (col < 15) {
                    int eastP = rescuedBoard[idx + 1];
                    if (eastP != -1 && eastP != -2 && PieceUtils.getEast(p) != PieceUtils.getWest(eastP)) conflict = true;
                }
                if (row < 15) {
                    int southP = rescuedBoard[idx + 16];
                    if (southP != -1 && southP != -2 && PieceUtils.getSouth(p) != PieceUtils.getNorth(southP)) conflict = true;
                }

                if (conflict) {
                    int[] neighbors = {idx, idx + 1, idx + 16, idx - 1, idx - 16};
                    for (int n : neighbors) {
                        if (n >= 0 && n < 256) {
                            if (n != 135 && n != 221 && n != 45 && n != 210 && n != 34) rescuedBoard[n] = -1;
                        }
                    }
                }
            }
        }
        int validPieces = 0;
        for (int i = 0; i < 256; i++) {
            if (rescuedBoard[i] != -1 && rescuedBoard[i] != -2) validPieces++;
        }
        System.arraycopy(rescuedBoard, 0, corruptedBoard, 0, 256);
        return validPieces;
    }

    private void generateSpiralOrder() {
        int top = 0, bottom = 15, left = 0, right = 15, idx = 0;
        while (top <= bottom && left <= right) {
            for (int j = left; j <= right; j++) buildOrder[idx++] = top * 16 + j;
            top++;
            for (int i = top; i <= bottom; i++) buildOrder[idx++] = i * 16 + right;
            right--;
            if (top <= bottom) {
                for (int j = right; j >= left; j--) buildOrder[idx++] = bottom * 16 + j;
                bottom--;
            }
            if (left <= right) {
                for (int i = bottom; i >= top; i--) buildOrder[idx++] = i * 16 + left;
                left++;
            }
        }
    }

    public enum BuildStrategy { TYPEWRITER, SPIRAL }

    class TopBoardRegistry {
        private final List<int[]> registry = new ArrayList<>();
        private int currentIndex = 0;
        private static final int MAX_CAPACITY = 20;

        public synchronized void offer(int[] board, int score) {
            int boardHash = Arrays.hashCode(board);
            for (int[] existing : registry) if (Arrays.hashCode(existing) == boardHash) return;
            if (registry.size() < MAX_CAPACITY) registry.add(Arrays.copyOf(board, 256));
        }

        public synchronized int[] nextForRepair() {
            if (registry.isEmpty()) return null;
            int[] board = registry.get(currentIndex);
            currentIndex = (currentIndex + 1) % registry.size();
            return board;
        }

        public synchronized void clear() {
            registry.clear();
            currentIndex = 0;
        }

        public List<int[]> getRawRegistry() { return registry; }
    }

    private class CpuSearchWorker implements Callable<Boolean> {
        private final int[] localBoard = new int[256];
        private final int[] localResumeBoard = new int[256];
        private final boolean[] localUsed = new boolean[256];
        private final Random rnd = new Random();
        private final int activeBatch;
        private long localTrialCount = 0;

        public CpuSearchWorker(int activeBatch) {
            this.activeBatch = activeBatch;
            System.arraycopy(flatResumeBoard, 0, localBoard, 0, 256);
            System.arraycopy(flatResumeBoard, 0, localResumeBoard, 0, 256);
            System.arraycopy(usedPhysicalPieces, 0, localUsed, 0, 256);

            for (int i = 0; i < 256; i++) {
                int p = localBoard[i];
                if (p != -1) {
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
                if (centerPhysicalIdx != -1) localUsed[centerPhysicalIdx] = true;
            }
            for (int h = 0; h < 4; h++) {
                int packed = hintPackedValues[h];
                if (packed != -1) {
                    localBoard[HINT_POSITIONS[h]] = packed;
                    localUsed[hintPhysicalIndices[h]] = true;
                }
            }
        }

        @Override
        public Boolean call() throws Exception {
            boolean result = solve(0);
            globalCpuTrialCount.addAndGet(localTrialCount);
            return result;
        }

        private boolean solve(int step) {
            if (manualOverrideRequested || (useGpu && currentBatchSize.get() >= activeBatch)) return false;

            if (step == 256) {
                synchronized (displayLock) {
                    if (step > absoluteHighScore) {
                        absoluteHighScore = step;
                        System.arraycopy(localBoard, 0, globalBestBoard, 0, 256);
                        RecordManager.saveRecord(buildDisplayBoard(globalBestBoard), absoluteHighScore, saveProfile);
                        saveAndUploadBucasLink(globalBestBoard, absoluteHighScore);
                        logger.info(">>> [CPU MODE] VICTORY! ETERNITY II SOLVED! <<<");
                    }
                }
                return true;
            }

            localTrialCount++;
            if (localTrialCount >= 1000) {
                globalCpuTrialCount.addAndGet(localTrialCount);
                localTrialCount = 0;
            }

            if (step > deepestStep) {
                synchronized (displayLock) {
                    if (step > deepestStep) {
                        deepestStep = step;
                        updateDisplay(deepestStep, buildDisplayBoard(localBoard));
                        if (!useGpu && deepestStep > absoluteHighScore) {
                            absoluteHighScore = deepestStep;
                            System.arraycopy(localBoard, 0, globalBestBoard, 0, 256);
                            RecordManager.saveRecord(buildDisplayBoard(globalBestBoard), absoluteHighScore, saveProfile);
                            consecutiveExtinctions = 0;
                            saveAndUploadBucasLink(globalBestBoard, absoluteHighScore);
                        }
                    }
                }
            }

            if (useGpu && step == currentSeedDepth) {
                int[] seed = new int[256];
                System.arraycopy(localBoard, 0, seed, 0, 256);
                seedPool.add(seed);
                currentBatchSize.incrementAndGet();
                return false;
            }

            int boardIdx = buildOrder[step];
            if (localBoard[boardIdx] != -1 || (lockCenter && boardIdx == 135)) return solve(step + 1);

            int row = boardIdx / 16;
            int col = boardIdx % 16;
            int northReq = (row == 0) ? 0 : (localBoard[boardIdx - 16] != -1 ? PieceUtils.getSouth(localBoard[boardIdx - 16]) : CompatibilityIndex.WILDCARD);
            int southReq = (row == 15) ? 0 : (localBoard[boardIdx + 16] != -1 ? PieceUtils.getNorth(localBoard[boardIdx + 16]) : CompatibilityIndex.WILDCARD);
            int westReq = (col == 0) ? 0 : (localBoard[boardIdx - 1] != -1 ? PieceUtils.getEast(localBoard[boardIdx - 1]) : CompatibilityIndex.WILDCARD);
            int eastReq = (col == 15) ? 0 : (localBoard[boardIdx + 1] != -1 ? PieceUtils.getWest(localBoard[boardIdx + 1]) : CompatibilityIndex.WILDCARD);

            java.util.BitSet candidates = compatIndex.candidatesFor(northReq, eastReq, southReq, westReq);
            compatIndex.andNotUsed(candidates, localUsed);

            if (localResumeBoard[boardIdx] != -1) {
                int resumeP = localResumeBoard[boardIdx];
                for (int oi = candidates.nextSetBit(0); oi >= 0; oi = candidates.nextSetBit(oi + 1)) {
                    if (inventory.allOrientations[oi] != resumeP) candidates.clear(oi);
                }
            }

            int candidateCount = candidates.cardinality();
            if (candidateCount == 0) return false;

            int[] orientIdxs = new int[candidateCount];
            int k = 0;
            for (int oi = candidates.nextSetBit(0); oi >= 0; oi = candidates.nextSetBit(oi + 1)) orientIdxs[k++] = oi;
            int offset = rnd.nextInt(candidateCount);

            for (int i = 0; i < candidateCount; i++) {
                if (useGpu && seedPool.size() >= activeBatch) return false;

                int orientationIdx = orientIdxs[(i + offset) % candidateCount];
                int p = inventory.allOrientations[orientationIdx];
                int physicalIdx = inventory.physicalMapping[orientationIdx];

                if (eastReq != CompatibilityIndex.WILDCARD && PieceUtils.getEast(p) != eastReq) continue;
                if (southReq != CompatibilityIndex.WILDCARD && PieceUtils.getSouth(p) != southReq) continue;
                if (boardIdx == poisonedIndex && p == poisonedPiece) continue;

                if (passesLookahead(p, step, row, col, boardIdx)) {
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

        private boolean passesLookahead(int p, int step, int row, int col, int idx) {
            if (row < 15 && localBoard[idx + 16] == -1) {
                int reqN = PieceUtils.getSouth(p);
                int reqE = (col == 15) ? 0 : (localBoard[idx + 17] != -1 ? PieceUtils.getWest(localBoard[idx + 17]) : CompatibilityIndex.WILDCARD);
                int reqW = (col == 0) ? 0 : (localBoard[idx + 15] != -1 ? PieceUtils.getEast(localBoard[idx + 15]) : CompatibilityIndex.WILDCARD);
                int reqS = (row == 14) ? 0 : (localBoard[idx + 32] != -1 ? PieceUtils.getNorth(localBoard[idx + 32]) : CompatibilityIndex.WILDCARD);

                java.util.BitSet southCandidates = compatIndex.candidatesFor(reqN, reqE, reqS, reqW);
                compatIndex.andNotUsed(southCandidates, localUsed);

                if (southCandidates.isEmpty()) return false;
                if (step > 40 && row < 14 && localBoard[idx + 32] == -1) {
                    boolean secondaryFound = false;
                    for (int i = southCandidates.nextSetBit(0); i >= 0; i = southCandidates.nextSetBit(i + 1)) {
                        int testPiece = inventory.allOrientations[i];
                        int nextReqN = PieceUtils.getSouth(testPiece);
                        if (compatIndex.hasAnyCandidate(nextReqN, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, localUsed)) {
                            secondaryFound = true;
                            break;
                        }
                    }
                    if (!secondaryFound) return false;
                }
            }

            if (col < 15 && localBoard[idx + 1] == -1) {
                int reqW = PieceUtils.getEast(p);
                int reqN = (row == 0) ? 0 : (localBoard[idx - 15] != -1 ? PieceUtils.getSouth(localBoard[idx - 15]) : CompatibilityIndex.WILDCARD);
                int reqE = (col == 14) ? 0 : (localBoard[idx + 2] != -1 ? PieceUtils.getWest(localBoard[idx + 2]) : CompatibilityIndex.WILDCARD);
                int reqS = (row == 15) ? 0 : (localBoard[idx + 17] != -1 ? PieceUtils.getNorth(localBoard[idx + 17]) : CompatibilityIndex.WILDCARD);
                return compatIndex.hasAnyCandidate(reqN, reqE, reqS, reqW, localUsed);
            }
            return true;
        }
    }
}
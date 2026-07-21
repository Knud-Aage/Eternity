package dk.puzzle.core;

import dk.puzzle.ai.ConflictReducer;
import dk.puzzle.ai.SeedSelector;
import dk.puzzle.ai.SurgeonHeuristics;
import dk.puzzle.gpu.GpuEngine;
import dk.puzzle.io.BucasExporter;
import dk.puzzle.io.CheckpointManager;
import dk.puzzle.io.RecordManager;
import dk.puzzle.model.CompatibilityIndex;
import dk.puzzle.model.PieceInventory;
import dk.puzzle.tools.HoleSolver;
import dk.puzzle.util.PieceUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The central orchestrator for the Eternity II puzzle solver, managing a multiphase
 * hybrid search strategy involving CPU-based seed generation and GPU-accelerated
 * deep search and repair.
 */
public class EternitySolver implements Runnable {

    private long cumulativeTrials = 0;
    private final Set<Integer> savedVariantHashes = ConcurrentHashMap.newKeySet();
    private final java.util.concurrent.atomic.AtomicInteger variantSaveThreshold = new java.util.concurrent.atomic.AtomicInteger(210);
    private final java.util.concurrent.atomic.AtomicInteger conflictSaveThreshold = new java.util.concurrent.atomic.AtomicInteger(60);
    public static final int SEED_DEPTH = 110;
    public static final int LNS_THRESHOLD = 200;
    private static final Logger logger = LogManager.getFormatterLogger(EternitySolver.class);
    // HINT STRATEGY FIELDS
    private static final int[] HINT_POSITIONS = {221, 45, 210, 34};
    // GUI configurable parameters
    public static volatile int userCpuHandoffDepth = 8;
    public static volatile int userSurgeonHoles = 20;
    final SurgeonHeuristics surgeon;
    final ConflictReducer conflictReducer;
    //    final int[] tabuTenure = new int[256];
    final int[] buildOrder = new int[256];
    final int[] bestBoard = new int[256];
    final int[] globalBestBoard = new int[256];
    final TopBoardRegistry topBoards = new TopBoardRegistry();
    private final ConcurrentHashMap<Integer, Integer> hashStrikeCount = new ConcurrentHashMap<>();
    private final Set<Integer> poisonedHashes = ConcurrentHashMap.newKeySet();
    // Core Solver Components
    private final PieceInventory inventory;
    private final CompatibilityIndex compatIndex;
    private final boolean lockCenter;
    private long lastVanguardLogTime = 0;
    // Threading and Concurrency
    private final ExecutorService executor;
    private final int numCores;
    // Full-board Monte Carlo scans are queued far faster than a single thread can
    // drain them (13k+ [HIGH DEPTH VARIANT] triggers/day observed vs. one board at
    // a time), so almost all candidate boards were silently dropped via the
    // pendingAnalyses cap before ever being evaluated. MIN_PRIORITY keeps these
    // threads from stealing cycles from the primary CPU search.
    private static final int ANALYSIS_THREADS = 4;
    // A small buffer beyond ANALYSIS_THREADS so submissions aren't rejected the
    // instant every worker is busy, without letting a long backlog of stale
    // board snapshots build up.
    private static final int ANALYSIS_QUEUE_CAP = ANALYSIS_THREADS + 2;
    private final java.util.concurrent.atomic.AtomicInteger analysisThreadCounter = new java.util.concurrent.atomic.AtomicInteger(0);
    private final ExecutorService backgroundAnalysisExecutor = Executors.newFixedThreadPool(ANALYSIS_THREADS, r -> {
        Thread t = new Thread(r, "monte-carlo-analysis-" + analysisThreadCounter.incrementAndGet());
        t.setDaemon(true);
        t.setPriority(Thread.MIN_PRIORITY);
        return t;
    });
    private final java.util.concurrent.atomic.AtomicInteger pendingAnalyses = new java.util.concurrent.atomic.AtomicInteger(0);
    private final java.util.concurrent.LinkedBlockingQueue<int[]> seedPool =
            new java.util.concurrent.LinkedBlockingQueue<>();
    private final Object displayLock = new Object();
    // Board State
    private final int[] flatResumeBoard = new int[256];
    private final boolean[] usedPhysicalPieces = new boolean[256];
    // Puzzle Specifics
    private final int targetPiece;
    // Solver Profile
    private final String saveProfile;
    private final BuildStrategy currentStrategy;
    // Metrics and Progress Tracking
    private final AtomicLong globalCpuTrialCount = new AtomicLong(0);
    private final AtomicLong globalGpuTrialCount = new AtomicLong(0);
    private final java.util.concurrent.atomic.AtomicInteger eliteWins =
            new java.util.concurrent.atomic.AtomicInteger(0);
    private final java.util.concurrent.atomic.AtomicInteger diverseWins =
            new java.util.concurrent.atomic.AtomicInteger(0);
    private final java.util.concurrent.atomic.AtomicInteger restartWins =
            new java.util.concurrent.atomic.AtomicInteger(0);
    private final Set<Integer> uniqueMaxScoreHashes = ConcurrentHashMap.newKeySet();
    private final int[] hintPackedValues = new int[]{-1, -1, -1, -1};
    private final int[] hintPhysicalIndices = new int[]{-1, -1, -1, -1};
    private final long lastDisplayUpdateTime = 0;
    volatile int userBatchSizeOverride = -1;
    volatile int absoluteHighScore = 0;
    volatile int deepestStep = 0;
    volatile boolean manualOverrideRequested = false;
    volatile int manualBaseCampTarget = 0;
    volatile double extinctionThreshold = 0.98;
    volatile int stagnationLimitMinutes = 20;
    private long lastVisualUpdate = System.currentTimeMillis();
    private GpuEngine gpuEngine;
    // Configuration Flags
    private boolean useGpu;
    private long methodStartTime = 0;
    private ExecutorService gpuExecutor; // Dedicated GPU thread for Context safety
    //    private final AtomicInteger currentBatchSize = new AtomicInteger(0);
    private int fatalGpuBugCount = 0;
    private int centerPhysicalIdx = -1;
    private volatile int highestP2DepthThisCycle = 0;
    private volatile int currentSeedDepth = SEED_DEPTH;
    private int currentRepairIteration = 0;
    private int consecutiveExtinctions = 0;
    private volatile int lastReportedDepth = 0;
    private int consecutiveGpuStagnation = 0;
    private int phaseCounter = 0; // For GPU Scheduling
    private long lastThroughputReportTime = System.currentTimeMillis();
    private long totalRepairVariationsTested = 0;
    private long repairLoopsCounter = 0;
    private volatile boolean isGpuBusy = false;
    private volatile int poisonedIndex = -1;
    private volatile int poisonedPiece = -1;
    private volatile int absolutePeakDepth = 0;
    private volatile int trueStagnationCounter = 0;
    private long lastForcedGuiUpdate = 0;
    private volatile int banStartStep = 0;
    private volatile int banEndStep = 0;

    // --- STAGNATION TRACKING VARIABLES ---
    private volatile int cpuStagnationCounter = 0;
    private volatile int lastPeakDepth = 0;

    /**
     * The value of {@code consecutiveExtinctions} at which the current poison expires.
     * The ban is lifted automatically when the extinction counter exceeds this value,
     * ensuring the piece/position combination is reconsidered in future scrap cycles
     * rather than being banned indefinitely.
     */
    private volatile int poisonExpiryExtinction = -1;
    private ScheduledExecutorService repairReporterScheduler;

    public EternitySolver(PieceInventory inventory, int trueCenterPiece, boolean useGpu, BuildStrategy strategy,
                          boolean lockCenter) {
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

        if (this.lockCenter) {
            List<Integer> newOrder = new ArrayList<>();
            newOrder.add(135); // Add Center
            for (int hPos : HINT_POSITIONS) {
                newOrder.add(hPos); // Add Hints
            }

            // Add the rest of the board
            for (int i = 0; i < 256; i++) {
                int pos = buildOrder[i];
                boolean isStatic = (pos == 135);
                for (int hPos : HINT_POSITIONS) {
                    if (pos == hPos) {
                        isStatic = true;
                        break;
                    }
                }
                if (!isStatic) {
                    newOrder.add(pos);
                }
            }
            // Overwrite the build order
            for (int i = 0; i < 256; i++) {
                buildOrder[i] = newOrder.get(i);
            }
            logger.info(">>> [ARCHITECTURE] Build Order optimized: Static locks shielded from GPU.");
        }

        this.compatIndex = new CompatibilityIndex(inventory.allOrientations, inventory.physicalMapping);
        this.surgeon = new SurgeonHeuristics(lockCenter, 0.70);
        this.conflictReducer = new ConflictReducer(inventory, lockCenter);

        loadCheckpointAndHints();
    }

    private void loadCheckpointAndHints() {
        SolverState loadedState = CheckpointManager.loadSmartState(saveProfile);

        if (loadedState != null) {
            restoreBoardState(loadedState.bestBoard);
            this.uniqueMaxScoreHashes.addAll(loadedState.uniqueMaxScoreHashes);
            this.cumulativeTrials = loadedState.cumulativeTrials;
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

        if (this.lockCenter) {
            // Mapping the exact dataset configuration provided:
            // HINT_POSITIONS array is {221, 45, 210, 34}
            // Pos 221 (Row 13, Col 13) -> Piece 249, Rot 1
            // Pos 45  (Row 2,  Col 13) -> Piece 255, Rot 0
            // Pos 210 (Row 13, Col 2)  -> Piece 181, Rot 0
            // Pos 34  (Row 2,  Col 2)  -> Piece 208, Rot 0

            int[] exactHintIds = {249, 255, 181, 208};
            int[] exactHintRots = {1, 0, 0, 0};

            for (int h = 0; h < 4; h++) {
                int physId = exactHintIds[h] - 1; // -1 to convert 1-indexed Piece ID to 0-indexed physical array
                int targetRot = exactHintRots[h];

                int foundPacked = -1;
                for (int oi = 0; oi < 1024; oi++) {
                    if (inventory.physicalMapping[oi] == physId && (oi % 4) == targetRot) {
                        foundPacked = inventory.allOrientations[oi];
                        break;
                    }
                }

                if (foundPacked != -1) {
                    hintPackedValues[h] = foundPacked;
                    hintPhysicalIndices[h] = physId;

                    bestBoard[HINT_POSITIONS[h]] = foundPacked;
                    globalBestBoard[HINT_POSITIONS[h]] = foundPacked;
                    flatResumeBoard[HINT_POSITIONS[h]] = foundPacked;

                    logger.info(">>> [HINT LOCKED] Successfully locked Hint " + (h + 1) + " (Piece " + exactHintIds[h] + ") at position " + HINT_POSITIONS[h]);
                } else {
                    logger.error(">>> [FATAL CONFIG] Could not find Piece " + exactHintIds[h] + " in inventory!");
                }
            }

            // --- Override and lock the Center Piece exactly as specified ---
            // Pos 135 (Row 8, Col 7) -> Center Piece 139, Rot 3
            int centerPhysId = 139 - 1;
            int centerRot = 3;
            int centerPacked = -1;

            for (int oi = 0; oi < 1024; oi++) {
                if (inventory.physicalMapping[oi] == centerPhysId && (oi % 4) == centerRot) {
                    centerPacked = inventory.allOrientations[oi];
                    break;
                }
            }

            if (centerPacked != -1) {
                flatResumeBoard[135] = centerPacked;
                bestBoard[135] = centerPacked;
                globalBestBoard[135] = centerPacked;
                this.centerPhysicalIdx = centerPhysId; // Ensure CPU worker marks it as used!
                logger.info(">>> [CENTER LOCKED] Successfully locked Center (Piece 139) at position 135");
            } else {
                logger.error(">>> [FATAL CONFIG] Could not find Center Piece 139 in inventory!");
            }

        } else {
            Arrays.fill(hintPackedValues, -1);
            Arrays.fill(hintPhysicalIndices, -1);
            logger.info(">>> [UNCONSTRAINED] Checkbox is off. Running completely without Center Piece or Hints!");
        }
    }

    private void restoreBoardState(int[][] loaded) {
        if (loaded == null) {
            return;
        }
        int loadedCount = 0;
        for (int r = 0; r < 16; r++) {
            if (loaded[r] == null) {
                continue;
            }
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

    public void setStagnationLimit(int minutes) {
        this.stagnationLimitMinutes = minutes;
    }

    public void setBatchSizeOverride(int size) {
        this.userBatchSizeOverride = size;
    }

    public void setExtinctionThreshold(double threshold) {
        this.extinctionThreshold = threshold;
    }

    public void setTargetedHolesPercentage(double percentage) {
        if (this.surgeon != null) {
            this.surgeon.setTargetedHolesPercentage(percentage);
        }
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
        if (deepestStep >= 200) {
            return 250;
        }
        if (deepestStep >= 180) {
            return 1000;
        }
        if (deepestStep >= 100) {
            return 5000;
        }
        if (deepestStep < 50) {
            return 5000;
        }
        return 15000;
    }

    private void initializeGpuEngine() {
        if (this.useGpu) {
            this.gpuExecutor = Executors.newSingleThreadExecutor();
            try {
                this.gpuEngine = this.gpuExecutor.submit(() -> new GpuEngine(inventory, lockCenter, buildOrder)).get();
                logger.info(">>> [HARDWARE] NVIDIA CUDA GPU detected and initialized successfully on dedicated thread.");
            } catch (Throwable t) {
                this.useGpu = false;
                this.gpuEngine = null;
                if (this.gpuExecutor != null) {
                    this.gpuExecutor.shutdownNow();
                }
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
        } catch (InterruptedException ignored) {
        }

        // 4. Force Java to delete the old GpuEngine object from RAM
        System.gc();

        // 5. Spin up a fresh, clean GPU Engine
        initializeGpuEngine();
    }

    private void setupShutdownHook() {
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            logger.info(">>> Shutdown hook: Saving final checkpoint...");
            synchronized (displayLock) {
                if (repairReporterScheduler != null) {
                    repairReporterScheduler.shutdownNow();
                }
                cumulativeTrials += globalCpuTrialCount.getAndSet(0) + globalGpuTrialCount.getAndSet(0);
                SolverState memoryToSave = new SolverState(
                        buildDisplayBoard(globalBestBoard),
                        absoluteHighScore,
                        this.uniqueMaxScoreHashes,
                        this.topBoards.getRawRegistry(),
                        cumulativeTrials
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

        // --- PROGRESS TRACKER ---
        // Reports real solving progress over a rolling window (not instantaneous
        // rate, which is too noisy/bursty to compare configs by — see the SPEED
        // line discussion, and not the all-time record, which is rare enough
        // (currently 220) that it tells you nothing on a minute-by-minute basis.
        // windowMaxDepth tracks the highest depth actually touched in the last
        // window, whether or not it ties/beats the session or all-time peak —
        // that's the number that's actually comparable between two configs run
        // for a few minutes each.
        long lastProgressLogTime = System.currentTimeMillis();
        int windowMaxDepth = deepestStep;
        long lastProgressTrials = cumulativeTrials;

        long lastSeedGrowthTime = System.currentTimeMillis();
        int lastSeedCount = 0;

        while (true) {
            try {
                windowMaxDepth = Math.max(windowMaxDepth, deepestStep);

                if (manualOverrideRequested) {
                    manualOverrideRequested = false;
                    retreat(manualBaseCampTarget, ">>> User Override...");
                    try {
                        Thread.sleep(1000);
                    } catch (InterruptedException ignored) {
                    }
                }

                int activeBatch = getDynamicBatchSize();
                if (userBatchSizeOverride > 0) {
                    activeBatch = userBatchSizeOverride;
                }

                if (this.useGpu) {
                    int currentSeeds = seedPool.size();
                    if (currentSeeds > lastSeedCount) {
                        // The queue is actively growing. Reset the timer!
                        lastSeedGrowthTime = System.currentTimeMillis();
                        lastSeedCount = currentSeeds;
                    } else if (System.currentTimeMillis() - lastSeedGrowthTime > 5000) {

                        logger.warn(">>> Watchdog: Endgame Starvation Triggered! Executing immediate retreat.");

                        // 1. Clear the queue instantly so we don't waste time processing a dead branch
                        if (seedPool != null) {
                            seedPool.clear();
                        }

                        // 2. EXECUTE THE SMART RETREAT!
                        int currentDeadEndHash = Arrays.hashCode(bestBoard);
                        boolean wasPoisoned = checkPoisonAndRetreat(currentDeadEndHash, deepestStep);

                        if (!wasPoisoned) {
                            // Top flatResumeBoard back up from bestBoard first, so the
                            // prefix we're about to "keep" below is real piece data and
                            // not holes left over from an earlier drain (see
                            // refreshAndCountBaseCamp for why this is needed).

                            int deadEndDepth = deepestStep;

                            // Calculate which row the engine died on (16 pieces per row)
                            int failedRow = deadEndDepth / 16;

                            // Roll back 2 full rows.
                            int rollbackRow = Math.max(1, failedRow - 4);
                            int newSeedDepth = rollbackRow * 16;

                            logger.warn(String.format(">>> [SMART RETREAT] Starved at depth %d. Rolling Base Camp to Row %d (Depth %d)",
                                    deadEndDepth, rollbackRow, newSeedDepth));

                            // Sync the depth so the engine knows where it is
                            deepestStep = newSeedDepth;

                            // Erase the CPU start board down to the rollback row
                            for (int i = newSeedDepth; i < 256; i++) {
                                if (i < buildOrder.length) {
                                    int pos = buildOrder[i];

                                    // Properly calculate if this specific position is a static lock
                                    boolean isStatic = (lockCenter && pos == 135);
                                    if (lockCenter) {
                                        for (int hPos : HINT_POSITIONS) {
                                            if (pos == hPos) {
                                                isStatic = true;
                                                break;
                                            }
                                        }
                                    }

                                    // Only erase the piece if it is NOT the center or a hint!
                                    if (!isStatic) {
                                        flatResumeBoard[pos] = -1;
                                    }
                                }
                            }
                            consecutiveExtinctions++;
                        }

                        lastSeedCount = 0;
                        lastSeedGrowthTime = System.currentTimeMillis();
                    }
                }

                if (this.useGpu && this.gpuEngine != null) {
                    if (deepestStep >= LNS_THRESHOLD) {
                        boolean seedsReady = seedPool.size() >= activeBatch;

                        // phaseCounter counts only Phase 3 completions.
                        // Phase 2 refresh fires every refreshInterval Phase 3 rounds.
                        // When stuck (>10 extinctions) Phase 2 runs every 2nd round.
                        int refreshInterval = (consecutiveExtinctions > 10) ? 2 : 4;
                        if (phaseCounter % refreshInterval == 0 && seedsReady) {
                            // Scheduled Phase 2 exploration round
                            runPhase2_GpuDeepDfs();
                            if (seedPool.size() >= activeBatch) {
                                logger.warn(">>> Phase 2 traffic jam (LNS Mode)! Force-clearing the pool.");
                                seedPool.clear();
                                triggerBranchScrap();
                            }
                            // Do NOT increment here — only Phase 3 runs count
                        } else if (!seedsReady) {
                            // Seed pool dry — refill before next Phase 2 round
                            runPhase1_CpuSeedGen();
                            // Do NOT increment here — seed-gen rounds don't count
                        } else {
                            // Normal Phase 3 repair round
                            runPhase3_GpuSurgeon();
                            phaseCounter++; // Only Phase 3 runs advance the counter
                        }
                    } else if (seedPool.size() >= activeBatch) {
                        runPhase2_GpuDeepDfs();
                        if (seedPool.size() >= activeBatch) {
                            logger.warn(">>> Phase 2 traffic jam (Standard Mode)! Force-clearing the pool.");
                            seedPool.clear();
                            triggerBranchScrap();
                        }
                    } else {
                        runPhase1_CpuSeedGen();
                    }
                } else {
                    runPhase1_CpuSeedGen();
                }

                if (System.currentTimeMillis() - lastPeriodicSave > 300_000) {
                    synchronized (displayLock) {
                        cumulativeTrials += globalCpuTrialCount.getAndSet(0) + globalGpuTrialCount.getAndSet(0);
                        SolverState memoryToSave = new SolverState(
                                buildDisplayBoard(globalBestBoard),
                                absoluteHighScore,
                                this.uniqueMaxScoreHashes,
                                this.topBoards.getRawRegistry(),
                                cumulativeTrials
                        );
                        CheckpointManager.saveSmartState(memoryToSave, saveProfile);
                    }
                    lastPeriodicSave = System.currentTimeMillis();
                }

                long sinceLastProgressLog = System.currentTimeMillis() - lastProgressLogTime;
                if (sinceLastProgressLog > 60_000) {
                    long trialsGain = cumulativeTrials - lastProgressTrials;
                    double trialsPerSec = trialsGain / (sinceLastProgressLog / 1000.0);

                    logger.info(">>> [PROGRESS] Last %.0fs: Highest depth touched %d/256 (all-time record: %d/256) | Trials: %,d (%,.0f/s avg)",
                            sinceLastProgressLog / 1000.0,
                            windowMaxDepth, absoluteHighScore,
                            trialsGain, trialsPerSec);

                    lastProgressLogTime = System.currentTimeMillis();
                    windowMaxDepth = deepestStep;
                    lastProgressTrials = cumulativeTrials;
                }
            } catch (Exception e) {
                logger.error(">>> [FATAL ERROR] PIPELINE CRASHED: ", e);
                try {
                    Thread.sleep(5000);
                } catch (InterruptedException ignored) {
                }
            }
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                logger.warn(">>> Orchestrator interrupted during breather.");
                break;
            }
        }
    }

    /**
     * Executes Phase 1 of the solving process, which runs a CPU-based search to generate initial board seeds.
     * <p>
     * This method performs the following steps:
     * <ul>
     * <li>Resets the tracking of used physical pieces.</li>
     * <li>Counts the number of locked (already placed) pieces on the board.</li>
     * <li>Determines the CPU search depth dynamically (handling fast GPU handoff logic).</li>
     * <li>Spawns parallel {@link CpuSearchWorker} instances across all currentSeedDepthCPU cores to search for and populate the seed
     * pool.</li>
     * <li>Handles deadlocks or branch exhaustions by checking if the seed pool is empty or if the CPU search has
     * finished
     * without finding a solution, triggering a branch scrap when appropriate.</li>
     * </ul>
     */
    private void runPhase1_CpuSeedGen() {
        Arrays.fill(usedPhysicalPieces, false);

        if (!useGpu) {
            currentSeedDepth = 256;
        }
        int lockedPieces = 0;
        for (int p : flatResumeBoard) {
            if (p != -1 && p != -2) {
                lockedPieces++;
            }
        }

        // Only sync deepestStep DOWN to lockedPieces if it was explicitly
        // reset by triggerBranchScrap. Never overwrite genuine GPU progress.
        if (deepestStep < lockedPieces) {
            deepestStep = lockedPieces;
        }

        // --- FAST GPU HANDOFF LOGIC ---
        int dynamicOffset;
        if (lockedPieces >= 190) {
            dynamicOffset = 2; // Endgame: CPU barely touches it, let the GPU brute-force!
        } else if (lockedPieces >= 150) {
            dynamicOffset = 4; // Late game
        } else {
            dynamicOffset = userCpuHandoffDepth; // Default (usually 8)
        }

        if (lockedPieces <= 5) {
            currentSeedDepth = 16;
            logger.info(">>> [PHASE 1] Empty board. Setting fast GPU handoff depth to: " + currentSeedDepth);
        } else if (this.lockCenter && lockedPieces < SEED_DEPTH) {
            currentSeedDepth = lockedPieces + dynamicOffset;
        } else if (lockedPieces < SEED_DEPTH) {
            // Unconstrained: The board is wide open, CPU can easily reach 110.
            currentSeedDepth = Math.max(SEED_DEPTH, lockedPieces + dynamicOffset);
        } else {
            currentSeedDepth = lockedPieces + dynamicOffset;
        }

        Set<Integer> dominantFrontierPieces = new java.util.HashSet<>();

        // We must ban a MASSIVE chunk of the poisoned tree, not just the first step!
        // This forces the CPU to build a radically alien foundation.
        int banLength = 15;
        int frontierEnd = Math.min(256, lockedPieces + banLength);

        for (int i = lockedPieces; i < frontierEnd; i++) {
            int idx = buildOrder[i];
            int p = bestBoard[idx];
            if (p != -1 && p != -2) {
                for (int oi = 0; oi < 1024; oi++) {
                    if (inventory.allOrientations[oi] == p) {
                        dominantFrontierPieces.add(inventory.physicalMapping[oi]);
                        break;
                    }
                }
            }
        }
        try {
            int activeBatch = getDynamicBatchSize();
            if (userBatchSizeOverride > 0) {
                activeBatch = userBatchSizeOverride;
            }

            List<CpuSearchWorker> workers = new ArrayList<>();

            for (int i = 0; i < numCores; i++) {
                if (!dominantFrontierPieces.isEmpty()) {
                    // Vanguard workers must survive for 'banLength' steps without touching ANY of the poisoned pieces!
                    workers.add(new CpuSearchWorker(activeBatch, dominantFrontierPieces, lockedPieces, lockedPieces + banLength));
                } else {
                    workers.add(new CpuSearchWorker(activeBatch, null, 0, 0));
                }
            }

            if (!dominantFrontierPieces.isEmpty()) {
                long now = System.currentTimeMillis();
                if (now - lastVanguardLogTime > 10000) {
                    logger.info(">>> [GENETIC EXPLORATION] 100%% Vanguard Threads. Hard-banning the next %d poisoned pieces at depth %d.",
                            dominantFrontierPieces.size(), lockedPieces);
                    lastVanguardLogTime = now;
                }
            }

            List<Future<Boolean>> futures = executor.invokeAll(workers);

            for (Future<Boolean> f : futures) {
                try {
                    f.get();
                } catch (ExecutionException e) {
                }
            }

            if (this.useGpu && seedPool.isEmpty()) {
                logger.warn(">>> [PHASE 1 DEADLOCK] CPU proved that NO alternative seeds exist for this Base Camp!");
                consecutiveExtinctions++;
                triggerBranchScrap();
            } else if (!this.useGpu) {
                logger.warn(">>> [CPU DEADLOCK] Exhausted branch without finding a solution. Tearing down...");
                consecutiveExtinctions++;
                triggerBranchScrap();
            }
        } catch (InterruptedException e) {
        }
    }

    private void runPhase2_GpuDeepDfs() {
        methodStartTime = System.currentTimeMillis();

        int activeBatch = getDynamicBatchSize();
        if (userBatchSizeOverride > 0) {
            activeBatch = userBatchSizeOverride;
        }

        List<int[]> seeds = new ArrayList<>();
        for (int i = 0; i < activeBatch; i++) {
            int[] s = seedPool.poll();
            if (s != null) {
                seeds.add(s);
            }
        }
        if (seeds.isEmpty()) {
            return;
        }

        int[] bestBoardOut = new int[256];
        isGpuBusy = true;

        GpuEngine.GpuResult result = null;

        try {
            Future<GpuEngine.GpuResult> future = gpuExecutor.submit(() ->
                    gpuEngine.runDeepDfs(seeds, currentSeedDepth, deepestStep, bestBoardOut)
            );
            result = future.get(10, java.util.concurrent.TimeUnit.MINUTES);
        } catch (Exception e) {
            isGpuBusy = false;
            rebootGpuEngine();
            triggerBranchScrap();
            return;
        }

        applyStaticLocks(bestBoardOut);
        // --- ANALYZE BATCH SEED PERFORMANCE ---
        int maxDepthInBatch = 0;
        int bestSeedIndex = 0;
        int[] threadDepths = result.threadDepths();

        for (int i = 0; i < threadDepths.length; i++) {
            if (threadDepths[i] > maxDepthInBatch) {
                maxDepthInBatch = threadDepths[i];
                bestSeedIndex = i;
            }
        }

        if (maxDepthInBatch > highestP2DepthThisCycle) {
            highestP2DepthThisCycle = maxDepthInBatch;
        }

        // Classify the winning seed based on its position in the stratified tiers
        // Elite = first 20%, Diverse = next 50%, Random-Restart = remaining 30%
        int totalSeeds = seeds.size();
        if (totalSeeds > 0) {
            double relativePosition = (double) bestSeedIndex / totalSeeds;
            if (relativePosition <= 0.20) {
                eliteWins.incrementAndGet();
            } else if (relativePosition <= 0.70) {
                diverseWins.incrementAndGet();
            } else {
                restartWins.incrementAndGet();
            }
        }
        globalGpuTrialCount.addAndGet(result.stepsTaken());
        isGpuBusy = false;

        long elapsed = System.currentTimeMillis() - methodStartTime;
//        logger.info(">>> GPU Phase 2 complete. Steps taken per second: %,d",
//                Math.round((double) result.stepsTaken() * 1000) / Math.max(1, elapsed));

        if (result.solved()) {
            handleVictory(bestBoardOut);
        }

        if (result.newHighScore() > deepestStep) {

            // 1. Accept the GPU's local progress!
            deepestStep = result.newHighScore();
            consecutiveGpuStagnation = 0;

            if (deepestStep >= absoluteHighScore - 5) {
                consecutiveExtinctions = 0;
            }

            System.arraycopy(bestBoardOut, 0, bestBoard, 0, 256);
            updateDisplay(deepestStep, buildDisplayBoard(bestBoard));

            if (deepestStep > variantSaveThreshold.get()) {
                int hash = Arrays.hashCode(bestBoard);
                if (savedVariantHashes.add(hash)) {
                    logger.info(">>> [HIGH DEPTH VARIANT] Unique %d-piece board detected! Generating Full Board Variant...", deepestStep);
                    logger.info(">>> Total Trials to reach this milestone: " + String.format("%,d", cumulativeTrials));
                    analyzeFullBoardPotential(bestBoard, deepestStep);
                }
            }

            // Log the local climb so you can watch it rise
            if (deepestStep > lastReportedDepth) {
                logger.info(">>> [CLIMBING] Depth: %d / 256 | Board Hash: %08X", deepestStep,
                        Arrays.hashCode(bestBoard));
                lastReportedDepth = deepestStep;
            }

            // 2. ONLY if it beats the Global Record do we save to disk
            if (deepestStep > absoluteHighScore) {
                absoluteHighScore = deepestStep;
                System.arraycopy(bestBoardOut, 0, globalBestBoard, 0, 256);

                int hash = Arrays.hashCode(globalBestBoard);
                logger.info(">>> [NEW GLOBAL RECORD] Depth: %d / 256 | Board Hash: %08X", absoluteHighScore, hash);
                logger.info(">>> Total Trials to reach this milestone: " + String.format("%,d", cumulativeTrials));
                uniqueMaxScoreHashes.clear();
                uniqueMaxScoreHashes.add(hash);

                int[] displayBoard = extendRecordGreedily(globalBestBoard, absoluteHighScore);
                int displayScore = countPieces(displayBoard);

                RecordManager.saveRecord(buildDisplayBoard(displayBoard), displayScore, saveProfile);
                try {
                    saveAndUploadBucasLink(displayBoard, displayScore);
                } catch (Exception e) {
                    logger.warn(">>> Skipping Google Drive upload: Not connected or unavailable.");
                }
                analyzeFullBoardPotential(globalBestBoard, absoluteHighScore);
            }

        } else {
            // The GPU completely failed to beat the current local depth.
            consecutiveGpuStagnation++;
//            analyzeFullBoardPotential(bestBoardOut, deepestStep);
        }

        // --- Seed generation for the next batch ---
        if (consecutiveGpuStagnation < 4) {
            seedPool.addAll(SeedSelector.selectBest(
                    seeds,
                    result.threadDepths(),
                    getDynamicBatchSize(),
                    buildOrder,
                    bestBoard,
                    new Random()
            ));
        } else {
            logger.warn(">>> [!!!] Phase 2 GPU stagnated! Activating Teardown...");
            consecutiveExtinctions++;
            triggerBranchScrap();
        }
    }

    private void runPhase3_GpuSurgeon() {
        long start = System.currentTimeMillis();
        int numClones = 50000;
        int holesToPunch = userSurgeonHoles;

        currentRepairIteration++;
        repairLoopsCounter++;
        totalRepairVariationsTested += numClones;

        int[] sourceBoard = topBoards.nextForRepair();
        if (sourceBoard == null) {
            sourceBoard = bestBoard;
        }

        // More holes when we're stuck deep — 5 holes at 209 pieces is not enough
        // to escape a structural dead end. Scale up with depth and stagnation.
        int actualHoles;
        if (consecutiveExtinctions > 20) {
            // Heavy stagnation: aggressive excavation to break out
            actualHoles = Math.min(holesToPunch * 2, deepestStep / 5);
        } else if (deepestStep > 240) {
            // Very close to solved: tiny surgical holes only
            actualHoles = Math.max(3, holesToPunch / 4);
        } else if (deepestStep > 200) {
            // Normal high-depth operation: use configured hole count
            actualHoles = holesToPunch;
        } else {
            // Below 200: use configured hole count
            actualHoles = holesToPunch;
        }
        actualHoles = Math.clamp(actualHoles, 1, deepestStep - 5); // safety clamp
        List<int[]> variations = surgeon.excavateFrontier(
                sourceBoard, numClones, actualHoles, currentRepairIteration, deepestStep, buildOrder);

        int scoreBefore = deepestStep;
        int[] bestBoardOut = new int[256];
        GpuEngine.GpuResult result = null;

        try {
            Future<GpuEngine.GpuResult> future = gpuExecutor.submit(() ->
                    gpuEngine.runRepairMode(variations, deepestStep, bestBoardOut)
            );
            // Phase 3 repair batches take 4-8 minutes per run with 50k clones.
            // 10 seconds killed every batch before results could be collected.
            result = future.get(10, java.util.concurrent.TimeUnit.MINUTES);
        } catch (java.util.concurrent.TimeoutException e) {
            logger.warn(">>> [GPU TIMEOUT] Phase 3 Surgeon locked up after 10 minutes! Triggering teardown and rebooting GPU...");
            rebootGpuEngine();
            triggerBranchScrap();
            return;
        } catch (Exception e) {
            logger.error(">>> [GPU ERROR] Pipeline crashed during Phase 3: " + e.getMessage());
            rebootGpuEngine();
            triggerBranchScrap();
            return;
        }

        applyStaticLocks(bestBoardOut);

// --- 1. METRICS & VICTORY CHECK ---
        globalGpuTrialCount.addAndGet(result.stepsTaken());
        isGpuBusy = false;

        // We declare the timer here, immediately before the logging,
        // using a unique name and a direct System call.
        long calculationElapsed = System.currentTimeMillis() - start;

        logger.info(">>> GPU Phase 3 (Surgeon) complete. Steps taken per second: %,d",
                Math.round((double) result.stepsTaken() * 1000) / Math.max(1, calculationElapsed));

        if (result.solved()) {
            handleVictory(bestBoardOut);
        }
        // --- 2. ACCEPT ANY LOCAL PROGRESS ---
        // Phase 3 must act on ANY improvement over deepestStep, not just
        // absolute records. Without this it cannot build momentum through
        // 213 -> 214 -> 215 -> 216 incrementally.
        if (result.newHighScore() > deepestStep) {

            if (!verifyBoardStrict(bestBoardOut)) {
                logger.error(">>> [FATAL GPU BUG] Phase 3 returned a board with an illegal edge conflict!");
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

            // --- 3. APPLY LOCAL PROGRESS ---
            deepestStep = result.newHighScore();
            consecutiveGpuStagnation = 0;
            consecutiveExtinctions = 0;

            System.arraycopy(bestBoardOut, 0, bestBoard, 0, 256);
            topBoards.offer(bestBoardOut, deepestStep);
            updateDisplay(deepestStep, buildDisplayBoard(bestBoard));

            if (deepestStep > variantSaveThreshold.get()) {
                int hash = Arrays.hashCode(bestBoard);
                if (savedVariantHashes.add(hash)) {
                    logger.info(">>> [HIGH DEPTH VARIANT] Unique %d-piece board detected! Generating Full Board Variant...", deepestStep);
                    analyzeFullBoardPotential(bestBoard, deepestStep);
                }
            }

            logger.info(">>> [PHASE 3 PROGRESS] Depth: %d / 256 | Hash: %08X",
                    deepestStep, Arrays.hashCode(bestBoard));

            // Only save to disk when beating the absolute global record
            if (deepestStep > absoluteHighScore) {
                absoluteHighScore = deepestStep;
                lastReportedDepth = absoluteHighScore;
                System.arraycopy(bestBoardOut, 0, globalBestBoard, 0, 256);

                int hash = Arrays.hashCode(globalBestBoard);
                uniqueMaxScoreHashes.clear();
                uniqueMaxScoreHashes.add(hash);

                logger.info(">>> [NEW GLOBAL RECORD] Depth: %d / 256 | Board Hash: %08X",
                        absoluteHighScore, hash);
                logger.info(">>> Total Trials to reach this milestone: " + String.format("%,d", cumulativeTrials));

                int[] displayBoard = extendRecordGreedily(globalBestBoard, absoluteHighScore);
                int displayScore = countPieces(displayBoard);

                RecordManager.saveRecord(buildDisplayBoard(displayBoard), displayScore, saveProfile);
                saveAndUploadBucasLink(displayBoard, displayScore);
                analyzeFullBoardPotential(globalBestBoard, absoluteHighScore);
            }

        } else {
            consecutiveGpuStagnation++;
        }
        if (consecutiveGpuStagnation >= 4) {
            logger.warn(">>> [!!!] Phase 3 stagnated (4 rounds). Activating Teardown...");
            consecutiveExtinctions++;
            triggerBranchScrap();
        }
    }

    private boolean checkPoisonAndRetreat(int currentBoardHash, int currentDepth) {
        // Only bother poisoning deep endgame boards (e.g., 200+)
        if (currentDepth < 200) {
            return false;
        }

        // If this board is already poisoned, instantly reject it!
        if (poisonedHashes.contains(currentBoardHash)) {
            logger.warn(">>> POISONED BOARD DETECTED! (" + currentBoardHash + "). Executing Nuclear Retreat!");
            executeNuclearRetreat(currentDepth);
            return true; // Tell the Watchdog we handled it
        }

        // Add a "strike" to this board hash
        int strikes = hashStrikeCount.getOrDefault(currentBoardHash, 0) + 1;
        hashStrikeCount.put(currentBoardHash, strikes);

        // 3 Strikes and it's permanently poisoned!
        if (strikes >= 3) {
            logger.error(">>> GRAVITY WELL DETECTED! Poisoning hash: " + currentBoardHash);
            poisonedHashes.add(currentBoardHash);
            executeNuclearRetreat(currentDepth);
            return true; // Tell the Watchdog we handled it
        }

        // It wasn't poisoned yet, just added a strike.
        return false;
    }

    private String generateHashForBoard(int[] boardArray) {
        if (boardArray == null) {
            return "EMPTY_BOARD";
        }

        // Converts the entire array into a single 32-bit integer hash,
        // then formats it as a nice, readable Hex String (like "8A90D2FC")
        return Integer.toHexString(java.util.Arrays.hashCode(boardArray)).toUpperCase();
    }

    private void executeNuclearRetreat(int currentDepth) {
        // Calculate which row the engine died on (16 pieces per row)
        int failedRow = currentDepth / 16;

        // Roll back 2 full rows (Minimum row 1 to protect the highly constrained grey border!)
        int rollbackRow = Math.max(1, failedRow - 4);
        int newSeedDepth = rollbackRow * 16;

        logger.info(">>> [GENETIC EXPLORATION] 100%% Vanguard Threads. Hard-banning the next 16 poisoned pieces at depth %d.", newSeedDepth);

        // 1. Sync the depth so the engine knows where it is
        deepestStep = newSeedDepth;

        // 2. Erase the CPU start board down to the new deep base camp
        for (int i = newSeedDepth; i < 256; i++) {
            if (i < buildOrder.length) {
                int pos = buildOrder[i];
                // Preserve static hint locks (-2)
                if (flatResumeBoard[pos] != -2) {
                    flatResumeBoard[pos] = -1;
                }
            }
        }

        // 3. Clear the GPU queue and trigger the Vanguard threads to rebuild the branch
        if (seedPool != null) {
            seedPool.clear();
        }
        consecutiveExtinctions++;
        triggerBranchScrap();
    }

    private void injectEntropy(int currentDepth) {
        java.util.Random rand = new java.util.Random();
        int piecesToRip = rand.nextInt(3) + 1; // Rip 1 to 3 random pieces

        for (int i = 0; i < piecesToRip; i++) {
            int randomDepthToRip = currentDepth - rand.nextInt(15); // Pick a piece near the edge
            if (randomDepthToRip > 0 && randomDepthToRip < 256) {
                int boardPos = buildOrder[randomDepthToRip];

                // Protect static locks (Center + Hints) from being ripped out
                boolean isStatic = (lockCenter && boardPos == 135);
                for (int hPos : HINT_POSITIONS) {
                    if (boardPos == hPos) {
                        isStatic = true;
                        break;
                    }
                }

                if (!isStatic) {
                    flatResumeBoard[boardPos] = -1; // Punch a hole in the CPU start board
                    bestBoard[boardPos] = -1;       // Erase it from local memory
                    consecutiveExtinctions++;
                }
            }
        }
        logger.info(">>> Injected Entropy: Randomly removed " + piecesToRip + " internal pieces to scramble the pathfinder.");
    }

    private void analyzeFullBoardPotential(int[] recordBoard) {
        analyzeFullBoardPotential(recordBoard, absoluteHighScore);
    }

    private void analyzeFullBoardPotential(int[] recordBoard, int baseScore) {
        int[] boardCopy = Arrays.copyOf(recordBoard, 256);
        int numEmpty = 0;
        for (int p : boardCopy) if (p == -1 || p == -2) numEmpty++;
        final int iterations = 200_000;
        logger.info(">>> [FULL BOARD SCAN] Queuing Monte Carlo fill: %d empty spots, %,d iterations (background)", numEmpty, iterations);

        if (pendingAnalyses.get() >= ANALYSIS_QUEUE_CAP) {
            logger.info(">>> [FULL BOARD SCAN] Skipped — %d analyses already queued.", pendingAnalyses.get());
            return;
        }
        pendingAnalyses.incrementAndGet();
        backgroundAnalysisExecutor.submit(() -> {
            try {
                // MCV (most-constrained-variable) ordering: always fill whichever
                // remaining hole has the most already-determined neighbour edges
                // next, instead of a fixed/random visiting order. See
                // ConflictReducer.mcvRestartFill's javadoc for why this beats a
                // fixed back-to-front or row-order fill.
                int[] bestBoard = conflictReducer.mcvRestartFill(boardCopy, iterations);
                int totalConflicts = conflictReducer.countConflicts(bestBoard);

                logger.info(">>> [FULL BOARD SCAN] Best result: %d / 480 edge conflicts after %,d iterations", totalConflicts, iterations);
                if (totalConflicts < 25) {
                    logger.warn(">>> [!!!] WOW! You are mathematically incredibly close to a full solution!");
                }
                if ((baseScore >= absoluteHighScore - 1) || (totalConflicts < conflictSaveThreshold.get() && baseScore > variantSaveThreshold.get())) {
                    // Board already cleared the save bar — worth the exact per-region
                    // search's extra cost now (rare event, off the hot GPU-batch path)
                    // to see if HoleSolver can beat the MCV/polish result before we save it.
                    int[] holeSolved = HoleSolver.solveConflicts(bestBoard, inventory, false).bestBoard();
                    int holeSolvedConflicts = conflictReducer.countConflicts(holeSolved);
                    if (holeSolvedConflicts < totalConflicts) {
                        logger.info(">>> [HOLE SOLVER] Improved %d -> %d conflicts before saving.",
                                totalConflicts, holeSolvedConflicts);
                        bestBoard = holeSolved;
                        totalConflicts = holeSolvedConflicts;
                    }
                    saveFullBoardVariant(bestBoard, baseScore, totalConflicts);
                } else {
                    logger.info(">>> [FULL BOARD SCAN] Not saved: %d conflicts >= threshold %d and base score %d <= variant threshold %d",
                            totalConflicts, conflictSaveThreshold.get(), baseScore, variantSaveThreshold.get());
                }
            } catch (Exception e) {
                logger.error(">>> [FULL BOARD SCAN] Background analysis failed: " + e.getMessage(), e);
            } finally {
                pendingAnalyses.decrementAndGet();
            }
        });
    }

    private void triggerBranchScrap() {
        analyzeFullBoardPotential(bestBoard, deepestStep);
        consecutiveExtinctions++;

        // --- TRUE STAGNATION TRACKER (GPU) ---
        // Climbing back to an old trap is NOT progress!
        if (deepestStep > absolutePeakDepth) {
            absolutePeakDepth = deepestStep;
            trueStagnationCounter = 0; // We broke the ALL-TIME record! Reset the bomb.
        } else {
            trueStagnationCounter++;   // Stuck in a Yo-Yo loop.
        }

        // --- STAGNATION DETECTION (CPU ONLY) ---
        if (!this.useGpu) {
            if (Math.abs(deepestStep - lastPeakDepth) <= 15) {
                cpuStagnationCounter++;
            } else {
                cpuStagnationCounter = 0;
                lastPeakDepth = deepestStep;
            }
        }

        // --- CALCULATE THE CURRENT FOUNDATION ---
        int currentBaseCamp = 0;
        for (int p : flatResumeBoard) {
            if (p != -1 && p != -2) {
                currentBaseCamp++;
            }
        }
        if (currentBaseCamp == 0) {
            currentBaseCamp = deepestStep;
        }

        int lockedPieces;

        if (this.useGpu) {
            // --- ELASTIC GPU TEARDOWN ---
            if (trueStagnationCounter > 4) {
                // [GENETIC RESET]: The foundation is poisoned!
                // Subtract from the BASE CAMP, not the peak, so we dig deeper every time!
                int dropAmount = 40;
                lockedPieces = Math.max(0, currentBaseCamp - dropAmount);
                trueStagnationCounter = 0;
                logger.warn(">>> [GENETIC RESET] Tunnel collapse! Purging deep foundation to depth " + lockedPieces);
            } else if (trueStagnationCounter > 2) {
                // Medium retreat (Subtract from the Peak)
                lockedPieces = Math.max(0, deepestStep - 40);
            } else {
                // Fast backtrack (Subtract from the Peak)
                lockedPieces = Math.max(0, deepestStep - (15 + (trueStagnationCounter * 2)));
            }
        } else {
            // --- ELASTIC CPU TEARDOWN ---
            int dropAmount;
            if (cpuStagnationCounter > 20) {
                dropAmount = 15; // Massive retreat
                cpuStagnationCounter = 0; // Reset after explosion
            } else if (cpuStagnationCounter > 10) {
                dropAmount = 8;  // Medium retreat
            } else if (cpuStagnationCounter > 5) {
                dropAmount = 4;  // Small retreat
            } else {
                dropAmount = 2;  // Normal efficient backtracking
            }
            lockedPieces = Math.max(0, deepestStep - dropAmount);
        }

// --- EXECUTE THE TEARDOWN ---
        int newTargetDepth = Math.max(0, lockedPieces - 16);
        int piecesDropped = deepestStep - newTargetDepth;

        logger.info(">>> [TEARDOWN] Deadlock trapped. Dropping " + piecesDropped + " pieces to depth " + newTargetDepth + ".");

        // CRITICAL: Update the ghost tracker
        deepestStep = newTargetDepth;

// --- THE UNIVERSAL TEARDOWN OVERRIDE ---
        for (int i = newTargetDepth; i < 256; i++) {
            if (i < buildOrder.length) {
                int pos = buildOrder[i];

                // Properly calculate if this specific position is a static lock
                boolean isStatic = (lockCenter && pos == 135);
                if (lockCenter) {
                    for (int hPos : HINT_POSITIONS) {
                        if (pos == hPos) {
                            isStatic = true;
                            break;
                        }
                    }
                }

                // Only erase the piece if it is NOT the center or a hint!
                if (!isStatic) {
                    flatResumeBoard[pos] = -1;
                }
            }
        }
    }

    private int[] extendRecordGreedily(int[] recordBoard, int currentDepth) {
        int[] extendedBoard = Arrays.copyOf(recordBoard, 256);
        boolean[] usedPhysical = new boolean[256];

        // Find out which pieces are already on the board
        for (int i = 0; i < 256; i++) {
            int p = extendedBoard[i];
            if (p != -1 && p != -2) {
                for (int oi = 0; oi < 1024; oi++) {
                    if (inventory.allOrientations[oi] == p) {
                        usedPhysical[inventory.physicalMapping[oi]] = true;
                        break;
                    }
                }
            }
        }

        int extraPieces = 0;

        // Try to build further from currentDepth without Lookahead
        for (int step = currentDepth; step < 256; step++) {
            int boardIdx = buildOrder[step];
            if (extendedBoard[boardIdx] != -1) {
                continue;
            }

            int row = boardIdx / 16;
            int col = boardIdx % 16;

            int northReq = (row == 0) ? 0 : (extendedBoard[boardIdx - 16] != -1 ? PieceUtils.getSouth(extendedBoard[boardIdx - 16]) : -1);
            int southReq = (row == 15) ? 0 : (extendedBoard[boardIdx + 16] != -1 ? PieceUtils.getNorth(extendedBoard[boardIdx + 16]) : -1);
            int westReq = (col == 0) ? 0 : (extendedBoard[boardIdx - 1] != -1 ? PieceUtils.getEast(extendedBoard[boardIdx - 1]) : -1);
            int eastReq = (col == 15) ? 0 : (extendedBoard[boardIdx + 1] != -1 ? PieceUtils.getWest(extendedBoard[boardIdx + 1]) : -1);

            boolean pieceFound = false;

            // Look for a piece that fits the immediate edges
            for (int physId = 0; physId < 256; physId++) {
                if (usedPhysical[physId]) {
                    continue;
                }

                for (int oi = 0; oi < 1024; oi++) {
                    if (inventory.physicalMapping[oi] == physId) {
                        int p = inventory.allOrientations[oi];

                        boolean fits = true;
                        if (northReq != -1 && PieceUtils.getNorth(p) != northReq) {
                            fits = false;
                        }
                        if (southReq != -1 && PieceUtils.getSouth(p) != southReq) {
                            fits = false;
                        }
                        if (westReq != -1 && PieceUtils.getWest(p) != westReq) {
                            fits = false;
                        }
                        if (eastReq != -1 && PieceUtils.getEast(p) != eastReq) {
                            fits = false;
                        }

                        if (fits) {
                            extendedBoard[boardIdx] = p;
                            usedPhysical[physId] = true;
                            extraPieces++;
                            pieceFound = true;
                            break;
                        }
                    }
                }
                if (pieceFound) {
                    break;
                }
            }

            // If we couldn't find a piece that fits, stop the decoration
            if (!pieceFound) {
                break;
            }
        }

        if (extraPieces > 0) {
            logger.info(">>> [GREEDY EXTENSION] Decorated the record with " + extraPieces + " extra piece(s) without using lookahead!");
        }

        return extendedBoard;
    }

    private void handleVictory(int[] winningBoard) {
        logger.info(">>> ETERNITY II SOLVED BY GPU PIPELINE!!! <<<");
        logger.info(">>> Total Trials to reach this milestone: " + String.format("%,d", cumulativeTrials));
        updateDisplay(256, buildDisplayBoard(winningBoard));
        RecordManager.saveRecord(buildDisplayBoard(winningBoard), 256, saveProfile);
        consecutiveExtinctions = 0;
        saveAndUploadBucasLink(winningBoard, 256);
        System.exit(0);
    }

    /**
     * Computes a position-weighted board fingerprint.
     * Printed in the speed log so repeated boards are immediately visible.
     * Two boards with the same pieces in different positions produce different hashes.
     */
    private int boardHash(int[] board) {
        int h = 0;
        for (int i = 0; i < 256; i++) {
            if (board[i] != -1) {
                h = h * 31 + (i * 1000003 ^ board[i]);
            }
        }
        return h & 0x7FFFFFFF;
    }

    private void reportSpeed() {
        long now = System.currentTimeMillis();
        long elapsed = now - lastThroughputReportTime;
        if (elapsed < 2000) {
            return;
        }

        long cpuTrials = globalCpuTrialCount.getAndSet(0);
        long gpuTrials = globalGpuTrialCount.getAndSet(0);

        // --- DIAGNOSTIC HEARTBEAT ---
        if (cpuTrials == 0 && gpuTrials == 0) {
            lastThroughputReportTime = now;
            return;
        }
        cumulativeTrials += gpuTrials + cpuTrials;
        double cpuTps = cpuTrials / (elapsed / 1000.0);
        double gpuTps = gpuTrials / (elapsed / 1000.0);

        int hash = Arrays.hashCode(bestBoard);

        // Print the primary throughput speed
        logger.info("[SPEED] CPU: %,.0f/s  |  GPU: %,.0f/s  |  Pieces: %d  |  Hash: %08X",
                cpuTps, gpuTps, deepestStep, hash);
        System.out.printf("[SPEED] CPU: %,.0f/s  |  GPU: %,.0f/s  |  Pieces: %d  |  Hash: %08X%n",
                cpuTps, gpuTps, deepestStep, hash);

        // --- PRINT POPULATION PERFORMANCE STATS ---
        int totalWins = eliteWins.get() + diverseWins.get() + restartWins.get();
        if (totalWins > 0) {
            logger.info("[EVOLUTION] Batch Winner Distribution -> Elite: %d%% | Diverse: %d%% | Restarts: %d%% | Peak P2 Depth: %d",
                    (eliteWins.get() * 100) / totalWins,
                    (diverseWins.get() * 100) / totalWins,
                    (restartWins.get() * 100) / totalWins,
                    highestP2DepthThisCycle);
            System.out.printf("[EVOLUTION] Batch Winner Distribution -> Elite: %d%% | Diverse: %d%% | Restarts: %d%% | Peak P2 Depth: %d%n",
                    (eliteWins.get() * 100) / totalWins,
                    (diverseWins.get() * 100) / totalWins,
                    (restartWins.get() * 100) / totalWins,
                    highestP2DepthThisCycle);
            // --- LIVE GPU ACTION FEED ---
            // Grab the GPU's peak mutation board before the Watchdog touches it
            int[] liveBoardCopy = new int[256];
            System.arraycopy(bestBoard, 0, liveBoardCopy, 0, 256);
            // Force the UI to draw the GPU's highest depth from this specific batch
//            Eternity.updateDisplay(highestP2DepthThisCycle, this.absoluteHighScore, buildDisplayBoard(liveBoardCopy));

        }

        // Reset the peak tracker for the next report window
        highestP2DepthThisCycle = deepestStep;

        lastThroughputReportTime = now;

        // --- LIVE FEED UI OVERRIDE ---
        if (now - lastForcedGuiUpdate > 60000) { // 60,000 milliseconds = 1 minute
            lastForcedGuiUpdate = now;

            // 1. Create a safe copy of the active GPU peak (NOT the base camp!)
            int[] liveBoardCopy = new int[256];
            System.arraycopy(bestBoard, 0, liveBoardCopy, 0, 256);
        }
    }

    void retreat(int targetStep, String logMessage) {
        deepestStep = Math.max(0, targetStep);
        for (int s = deepestStep; s < 256; s++) {
            bestBoard[buildOrder[s]] = -1;
        }
        if (lockCenter) {
            bestBoard[135] = targetPiece;
            for (int h = 0; h < 4; h++) {
                if (hintPackedValues[h] != -1) {
                    bestBoard[HINT_POSITIONS[h]] = hintPackedValues[h];
                }
            }
        }
        updateDisplay(countPieces(bestBoard), buildDisplayBoard(bestBoard));
        if (logMessage != null) {
            logger.info(logMessage);
        }
    }

    int countPieces(int[] board) {
        int count = 0;
        for (int p : board) {
            if (p != -1 && p != -2) {
                count++;
            }
        }
        return count;
    }

    int[][] buildDisplayBoard(int[] sourceArray) {
        int[][] displayBoard = new int[16][16];
        for (int i = 0; i < 16; i++) {
            Arrays.fill(displayBoard[i], -1);
        }
        for (int i = 0; i < 256; i++) {
            if (sourceArray[i] == -1) {
                continue;
            }
            displayBoard[i / 16][i % 16] = sourceArray[i];
        }
        return displayBoard;
    }

    private void updateDisplay(int depth, int[][] displayBoard) {
        long now = System.currentTimeMillis();

        boolean isNewRecord = depth >= absoluteHighScore;
        boolean timeToRefresh = (now - lastVisualUpdate) >= 300_000;

        if (!isNewRecord && !timeToRefresh) {
            return; // Gate is closed!
        }

        // If we passed the gate, update the timer
        lastVisualUpdate = now;

        Eternity.updateDisplay(depth + 1, this.absoluteHighScore + 1, displayBoard);
    }

    private void saveAndUploadBucasLink(int[] board, int score) {
        try {
            String bucasLink = BucasExporter.exportBoard(board);
            java.io.File folder = new java.io.File(saveProfile);
            if (!folder.exists()) {
                folder.mkdirs();
            }

            java.io.File linkFile = new java.io.File(folder, "bucas_link_" + score + ".txt");
            try (java.io.FileWriter writer = new java.io.FileWriter(linkFile)) {
                writer.write("Eternity II Record: " + score + " pieces\n");
                writer.write("Time: " + LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss")) + "\n");
                writer.write("Strategy: " + saveProfile + "\n\n");

                writer.write("--- VISUALIZER LINK (Colors Only) ---\n");
                writer.write(bucasLink + "\n\n");

                writer.write("--- VERIFIABLE MATHEMATICAL PROOF (1-Based-ID/Rotation) ---\n");
                for (int row = 0; row < 16; row++) {
                    for (int col = 0; col < 16; col++) {
                        int p = board[row * 16 + col];
                        if (p == -1 || p == -2) {
                            writer.write("---/- ");
                        } else {
                            int physId = -1;
                            int rotation = 0;
                            for (int oi = 0; oi < 1024; oi++) {
                                if (inventory.allOrientations[oi] == p) {
                                    physId = inventory.physicalMapping[oi];
                                    rotation = oi % 4;
                                    break;
                                }
                            }
                            writer.write((physId + 1) + "/" + rotation + " ");
                        }
                    }
                    writer.write("\n");
                }
            }
        } catch (Exception e) {
            logger.warn(String.format(">>> Error writing verifiable bucas text file: %s", e.getMessage()));
        }
    }

    private void saveFullBoardVariant(int[] simulatedBoard, int baseScore, int conflicts) {
        String fullProfileFolder = saveProfile + "_FULL";
        java.io.File folder = new java.io.File(fullProfileFolder);
        if (!folder.exists()) {
            folder.mkdirs();
        }

        String timeId = java.time.LocalTime.now().format(java.time.format.DateTimeFormatter.ofPattern("HHmmss_SSS"));
        int displayScore = baseScore + 1; // radar holds one piece back; show the true count
        String baseFilename = "Errors" + conflicts + "_Base" + displayScore + "_" + timeId;

        try (java.io.PrintWriter writer = new java.io.PrintWriter(new java.io.File(folder, "Raw_Board_Output_" + displayScore + ".txt"))) {
            for (int row = 0; row < 16; row++) {
                StringBuilder line = new StringBuilder();
                for (int col = 0; col < 16; col++) {
                    int p = simulatedBoard[row * 16 + col];
                    if (p == -1 || p == -2) {
                        line.append("0/0");
                    } else {
                        int physId = -1;
                        int rotation = 0;
                        for (int oi = 0; oi < 1024; oi++) {
                            if (inventory.allOrientations[oi] == p) {
                                physId = inventory.physicalMapping[oi];
                                rotation = oi % 4;
                                break;
                            }
                        }
                        line.append((physId + 1)).append("/").append(rotation);
                    }
                    if (col < 15) {
                        line.append(" ");
                    }
                }
                writer.println(line);
            }
            logger.info(String.format(">>> [FULL BOARD SCAN] Saved raw board text for BoardImporter: Raw_Board_Output_%d.txt", baseScore));
        } catch (Exception e) {
            logger.error(String.format(">>> Error saving Raw Board Text: %s", e.getMessage()));
        }

        try (java.io.PrintWriter writer = new java.io.PrintWriter(new java.io.File(folder, baseFilename + ".csv"))) {
            for (int row = 0; row < 16; row++) {
                StringBuilder line = new StringBuilder();
                for (int col = 0; col < 16; col++) {
                    int p = simulatedBoard[row * 16 + col];
                    if (p == -1 || p == -2) {
                        line.append("---/-");
                    } else {
                        int physId = -1;
                        int rotation = 0;
                        for (int oi = 0; oi < 1024; oi++) {
                            if (inventory.allOrientations[oi] == p) {
                                physId = inventory.physicalMapping[oi];
                                rotation = oi % 4;
                                break;
                            }
                        }
                        line.append((physId + 1)).append("/").append(rotation);
                    }
                    if (col < 15) {
                        line.append(",");
                    }
                }
                writer.println(line);
            }
            logger.info(String.format(">>> Saved official verification file: %s.csv", baseFilename));
        } catch (Exception e) {
            logger.error(String.format(">>> Error saving pieces.csv: %s", e.getMessage()));
        }

        try (java.io.PrintWriter writer = new java.io.PrintWriter(new java.io.File(folder,
                baseFilename + "_link.txt"))) {
            long totalTrials = cumulativeTrials + globalCpuTrialCount.get() + globalGpuTrialCount.get();
            writer.println("Base Score: " + displayScore);
            writer.println("Edge Conflicts: " + conflicts);
            writer.println("Total Trials: " + String.format("%,d", totalTrials));
            writer.println(BucasExporter.exportBoard(simulatedBoard));
            logger.info(String.format(">>> Saved full board Bucas link: %s_link.txt", baseFilename));
        } catch (Exception e) {
            logger.error(String.format(">>> Error saving Full Board Bucas Link: %s", e.getMessage()));
        }

        try {
            RecordManager.saveImage(buildDisplayBoard(simulatedBoard), new java.io.File(folder,
                    baseFilename + ".png").getAbsolutePath());
            logger.info(String.format(">>> Saved full board PNG image: %s.png", baseFilename));
        } catch (Exception e) {
            logger.error(String.format(">>> Error saving PNG image: %s", e.getMessage()));
        }
    }

    boolean verifyBoardStrict(int[] board) {
        for (int row = 0; row < 16; row++) {
            for (int col = 0; col < 16; col++) {
                int idx = row * 16 + col;
                int p = board[idx];
                if (p == -1 || p == -2) {
                    continue;
                }
                if (col < 15) {
                    int eastP = board[idx + 1];
                    if (eastP != -1 && eastP != -2) {
                        if (PieceUtils.getEast(p) != PieceUtils.getWest(eastP)) {
                            return false;
                        }
                    }
                }
                if (row < 15) {
                    int southP = board[idx + 16];
                    if (southP != -1 && southP != -2) {
                        if (PieceUtils.getSouth(p) != PieceUtils.getNorth(southP)) {
                            return false;
                        }
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
                if (p == -1 || p == -2) {
                    continue;
                }

                boolean conflict = false;
                if (col < 15) {
                    int eastP = rescuedBoard[idx + 1];
                    if (eastP != -1 && eastP != -2 && PieceUtils.getEast(p) != PieceUtils.getWest(eastP)) {
                        conflict = true;
                    }
                }
                if (row < 15) {
                    int southP = rescuedBoard[idx + 16];
                    if (southP != -1 && southP != -2 && PieceUtils.getSouth(p) != PieceUtils.getNorth(southP)) {
                        conflict = true;
                    }
                }

                if (conflict) {
                    int[] neighbors = {idx, idx + 1, idx + 16, idx - 1, idx - 16};
                    for (int n : neighbors) {
                        if (n >= 0 && n < 256) {
                            if (this.lockCenter && (n == 135 || n == 221 || n == 45 || n == 210 || n == 34)) {
                                continue;
                            }
                            rescuedBoard[n] = -1;
                        }
                    }
                }
            }
        }
        int validPieces = 0;
        for (int i = 0; i < 256; i++) {
            if (rescuedBoard[i] != -1 && rescuedBoard[i] != -2) {
                validPieces++;
            }
        }
        System.arraycopy(rescuedBoard, 0, corruptedBoard, 0, 256);
        return validPieces;
    }

    /**
     * Allows the GUI to dynamically change the depth at which full-board variants are saved.
     */
    public void setVariantSaveThreshold(int newThreshold) {
        this.variantSaveThreshold.set(newThreshold);
        logger.info(">>> [CONFIG] Variant Save Threshold updated to: " + newThreshold);
    }

    public void setConflictSaveThreshold(int newThreshold) {
        this.conflictSaveThreshold.set(newThreshold);
        logger.info(">>> [CONFIG] Conflict Save Threshold updated to: " + newThreshold);
    }

    private void applyStaticLocks(int[] board) {
        if (this.lockCenter) {
            board[135] = targetPiece;
            for (int h = 0; h < 4; h++) {
                if (hintPackedValues[h] != -1) {
                    board[HINT_POSITIONS[h]] = hintPackedValues[h];
                }
            }
        }
    }

    private void generateSpiralOrder() {
        int top = 0, bottom = 15, left = 0, right = 15, idx = 0;
        while (top <= bottom && left <= right) {
            for (int j = left; j <= right; j++) {
                buildOrder[idx++] = top * 16 + j;
            }
            top++;
            for (int i = top; i <= bottom; i++) {
                buildOrder[idx++] = i * 16 + right;
            }
            right--;
            if (top <= bottom) {
                for (int j = right; j >= left; j--) {
                    buildOrder[idx++] = bottom * 16 + j;
                }
                bottom--;
            }
            if (left <= right) {
                for (int i = bottom; i >= top; i--) {
                    buildOrder[idx++] = i * 16 + left;
                }
                left++;
            }
        }
    }

    public enum BuildStrategy {TYPEWRITER, SPIRAL}

    class TopBoardRegistry {
        private static final int MAX_CAPACITY = 20;
        private final List<int[]> registry = new ArrayList<>();
        private int currentIndex = 0;

        public synchronized void offer(int[] board, int score) {
            int boardHash = Arrays.hashCode(board);
            for (int[] existing : registry) {
                if (Arrays.hashCode(existing) == boardHash) {
                    return;
                }
            }
            if (registry.size() < MAX_CAPACITY) {
                registry.add(Arrays.copyOf(board, 256));
            }
        }

        public synchronized int[] nextForRepair() {
            if (registry.isEmpty()) {
                return null;
            }
            int[] board = registry.get(currentIndex);
            currentIndex = (currentIndex + 1) % registry.size();
            return board;
        }

        public synchronized void clear() {
            registry.clear();
            currentIndex = 0;
        }

        public List<int[]> getRawRegistry() {
            return registry;
        }
    }

    private class CpuSearchWorker implements Callable<Boolean> {
        private final int[] localBoard = new int[256];
        private final int[] localResumeBoard = new int[256];
        private final boolean[] localUsed = new boolean[256];
        // Eternity II has colors from 0 (grey edge) up to 22. We allocate 24 for safety.
        private int[] inventoryColorsLeft = new int[24];
        private int[] openEdgesOnBoard = new int[24];
        private final Random rnd = new Random();
        private final int activeBatch;
        private long localTrialCount = 0;

        // --- NEW VANGUARD EXPLORATION FIELDS ---
        private final Set<Integer> bannedPhysicalPieces;
        private final int banStartStep;
        private final int banEndStep;

        public CpuSearchWorker(int activeBatch, Set<Integer> bannedPieces, int banStart, int banEnd) {
            this.activeBatch = activeBatch;
            this.bannedPhysicalPieces = bannedPieces;
            this.banStartStep = banStart;
            this.banEndStep = banEnd;

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
                if (centerPhysicalIdx != -1) {
                    localUsed[centerPhysicalIdx] = true;
                }
                for (int h = 0; h < 4; h++) {
                    int packed = hintPackedValues[h];
                    if (packed != -1) {
                        localBoard[HINT_POSITIONS[h]] = packed;
                        if (hintPhysicalIndices[h] != -1) {
                            localUsed[hintPhysicalIndices[h]] = true;
                        }
                    }
                }
            }
            initColorBudget();
        }

        @Override
        public Boolean call() {
            boolean result = solve(0);
            globalCpuTrialCount.addAndGet(localTrialCount);
            return result;
        }

        private boolean solve(int step) {
            if (manualOverrideRequested || (useGpu && seedPool.size() >= activeBatch)) {
                return false;
            }

            if (step == 256) {
                synchronized (displayLock) {
                    if (step > absoluteHighScore) {
                        absoluteHighScore = step;
                        System.arraycopy(localBoard, 0, globalBestBoard, 0, 256);
                        RecordManager.saveRecord(buildDisplayBoard(globalBestBoard), absoluteHighScore, saveProfile);
                        saveAndUploadBucasLink(globalBestBoard, absoluteHighScore);
                        logger.info(">>> [CPU MODE] VICTORY! ETERNITY II SOLVED! <<<");
                        logger.info(">>> Total Trials to reach this milestone: " + String.format("%,d", cumulativeTrials));
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
                        System.arraycopy(localBoard, 0, bestBoard, 0, 256);
//                        updateDisplay(deepestStep, buildDisplayBoard(localBoard));
                        if (!useGpu && deepestStep > variantSaveThreshold.get()) {
                            int variantHash = Arrays.hashCode(bestBoard);
                            if (savedVariantHashes.add(variantHash)) {
                                logger.info(">>> [HIGH DEPTH VARIANT] Unique %d-piece board detected! Generating Full Board Variant...", deepestStep);
                                analyzeFullBoardPotential(bestBoard, deepestStep);
                            }
                        }
                        if (!useGpu && deepestStep > absoluteHighScore) {
                            absoluteHighScore = deepestStep;
                            System.arraycopy(localBoard, 0, globalBestBoard, 0, 256);
                            logger.info(">>> Total Trials to reach this milestone: " + String.format("%,d", cumulativeTrials));

                            int[] displayBoard = extendRecordGreedily(globalBestBoard, absoluteHighScore);
                            int displayScore = countPieces(displayBoard);

                            RecordManager.saveRecord(buildDisplayBoard(displayBoard), displayScore, saveProfile);
                            analyzeFullBoardPotential(globalBestBoard, absoluteHighScore);
                            try {
                                saveAndUploadBucasLink(displayBoard, displayScore);
                            } catch (Exception e) {
                                logger.warn(">>> Skipping Google Drive upload: Not connected or unavailable.");
                            }
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
//                currentBatchSize.incrementAndGet();
                return false;
            }

            int boardIdx = buildOrder[step];
            if (localBoard[boardIdx] != -1 || (lockCenter && boardIdx == 135)) {
                return solve(step + 1);
            }

            int row = boardIdx / 16;
            int col = boardIdx % 16;
            int northReq = (row == 0) ? 0 : (localBoard[boardIdx - 16] != -1 ?
                    PieceUtils.getSouth(localBoard[boardIdx - 16]) : CompatibilityIndex.WILDCARD);
            int southReq = (row == 15) ? 0 : (localBoard[boardIdx + 16] != -1 ?
                    PieceUtils.getNorth(localBoard[boardIdx + 16]) : CompatibilityIndex.WILDCARD);
            int westReq = (col == 0) ? 0 : (localBoard[boardIdx - 1] != -1 ?
                    PieceUtils.getEast(localBoard[boardIdx - 1]) : CompatibilityIndex.WILDCARD);
            int eastReq = (col == 15) ? 0 : (localBoard[boardIdx + 1] != -1 ?
                    PieceUtils.getWest(localBoard[boardIdx + 1]) : CompatibilityIndex.WILDCARD);

            java.util.BitSet candidates = compatIndex.candidatesFor(northReq, eastReq, southReq, westReq);
            compatIndex.andNotUsed(candidates, localUsed);

            if (localResumeBoard[boardIdx] != -1) {
                int resumeP = localResumeBoard[boardIdx];
                for (int oi = candidates.nextSetBit(0); oi >= 0; oi = candidates.nextSetBit(oi + 1)) {
                    if (inventory.allOrientations[oi] != resumeP) {
                        candidates.clear(oi);
                    }
                }
            }

            int candidateCount = candidates.cardinality();
            if (candidateCount == 0) {
                return false;
            }

            int[] orientIdxs = new int[candidateCount];
            int k = 0;
            for (int oi = candidates.nextSetBit(0); oi >= 0; oi = candidates.nextSetBit(oi + 1)) {
                orientIdxs[k++] = oi;
            }
            int offset = rnd.nextInt(candidateCount);

            for (int i = 0; i < candidateCount; i++) {
                if (useGpu && seedPool.size() >= activeBatch) {
                    return false;
                }

                int orientationIdx = orientIdxs[(i + offset) % candidateCount];
                int p = inventory.allOrientations[orientationIdx];
                int physicalIdx = inventory.physicalMapping[orientationIdx];

                // --- VANGUARD EXPLORATION LOGIC ---
                // If this is a Vanguard Worker in the active frontier, mathematically forbid
                // placing the pieces that make up the current local trap!
                if (bannedPhysicalPieces != null && step >= banStartStep && step < banEndStep) {
                    if (bannedPhysicalPieces.contains(physicalIdx)) {
                        continue;
                    }
                }

                if (eastReq != CompatibilityIndex.WILDCARD && PieceUtils.getEast(p) != eastReq) {
                    continue;
                }

                // >>> NEW: PARITY CHECK (COLOR BUDGET) LOGIC <<<

                // Extract the 4 colors of the candidate piece
                int nC = PieceUtils.getNorth(p);
                int eC = PieceUtils.getEast(p);
                int sC = PieceUtils.getSouth(p);
                int wC = PieceUtils.getWest(p);

                // 1. DECREASE SUPPLY: The piece leaves the inventory
                inventoryColorsLeft[nC]--;
                inventoryColorsLeft[eC]--;
                inventoryColorsLeft[sC]--;
                inventoryColorsLeft[wC]--;

                // 2. UPDATE DEMAND (Open Edges):
                // If a neighbor is empty (-1), we CREATE an open edge (demand goes up).
                // If a neighbor is present, we CLOSE an open edge (demand goes down).
                if (row > 0 && localBoard[boardIdx - 16] == -1) {
                    openEdgesOnBoard[nC]++;
                } else if (row > 0) {
                    openEdgesOnBoard[nC]--;
                }

                if (col < 15 && localBoard[boardIdx + 1] == -1) {
                    openEdgesOnBoard[eC]++;
                } else if (col < 15) {
                    openEdgesOnBoard[eC]--;
                }

                if (row < 15 && localBoard[boardIdx + 16] == -1) {
                    openEdgesOnBoard[sC]++;
                } else if (row < 15) {
                    openEdgesOnBoard[sC]--;
                }

                if (col > 0 && localBoard[boardIdx - 1] == -1) {
                    openEdgesOnBoard[wC]++;
                } else if (col > 0) {
                    openEdgesOnBoard[wC]--;
                }

                // 3. PARITY CHECK (The Early Pruning Gate)
                boolean isColorBudgetSafe = true;
                // We only check the colors we just modified to save CPU cycles
                if (openEdgesOnBoard[nC] > inventoryColorsLeft[nC] ||
                        openEdgesOnBoard[eC] > inventoryColorsLeft[eC] ||
                        openEdgesOnBoard[sC] > inventoryColorsLeft[sC] ||
                        openEdgesOnBoard[wC] > inventoryColorsLeft[wC]) {
                    isColorBudgetSafe = false; // PATH IS MATHEMATICALLY DEAD!
                }

                // Proceed ONLY if the color budget is safe AND the local lookahead passes
                if (isColorBudgetSafe && passesLookahead(p, step, row, col, boardIdx)) {

                    // 1. Temporarily place the piece to change the board state
                    localBoard[boardIdx] = p;
                    localUsed[physicalIdx] = true;
                    int ghost = localResumeBoard[boardIdx];
                    localResumeBoard[boardIdx] = -1;

                    // 2. >>> THE RADAR SWEEP <<<
                    if (step < 200 || isTopologicallyViable(localBoard, localUsed)) {

                        // 3. The geometry is safe! Take the next step.
                        if (solve(step + 1)) {
                            return true;
                        }
                    }

                    // 4. Backtrack geometry
                    localBoard[boardIdx] = -1;
                    localUsed[physicalIdx] = false;
                    localResumeBoard[boardIdx] = ghost;
                }
                // 5. BACKTRACK PARITY: Revert the Supply and Demand arrays
                inventoryColorsLeft[nC]++;
                inventoryColorsLeft[eC]++;
                inventoryColorsLeft[sC]++;
                inventoryColorsLeft[wC]++;

                if (row > 0 && localBoard[boardIdx - 16] == -1) {
                    openEdgesOnBoard[nC]--;
                } else if (row > 0) {
                    openEdgesOnBoard[nC]++;
                }

                if (col < 15 && localBoard[boardIdx + 1] == -1) {
                    openEdgesOnBoard[eC]--;
                } else if (col < 15) {
                    openEdgesOnBoard[eC]++;
                }

                if (row < 15 && localBoard[boardIdx + 16] == -1) {
                    openEdgesOnBoard[sC]--;
                } else if (row < 15) {
                    openEdgesOnBoard[sC]++;
                }

                if (col > 0 && localBoard[boardIdx - 1] == -1) {
                    openEdgesOnBoard[wC]--;
                } else if (col > 0) {
                    openEdgesOnBoard[wC]++;
                }
            }
            return false;
        }

        /**
         * Initializes the color budget for this specific thread.
         * Calculates total supply in the inventory and adjusts for any
         * pieces already locked on the board (Center/Hints).
         */
        private void initColorBudget() {
            java.util.Arrays.fill(inventoryColorsLeft, 0);
            java.util.Arrays.fill(openEdgesOnBoard, 0);

            // 1. Calculate total global supply
            for (int i = 0; i < 256; i++) {
                int p = inventory.allOrientations[inventory.physicalMapping[i]];
                inventoryColorsLeft[PieceUtils.getNorth(p)]++;
                inventoryColorsLeft[PieceUtils.getEast(p)]++;
                inventoryColorsLeft[PieceUtils.getSouth(p)]++;
                inventoryColorsLeft[PieceUtils.getWest(p)]++;
            }

            // 2. Adjust supply and calculate DEMAND based on the current Base Camp
            for (int i = 0; i < 256; i++) {
                int p = localBoard[i];
                if (p != -1) {
                    int r = i / 16;
                    int c = i % 16;
                    int nC = PieceUtils.getNorth(p);
                    int eC = PieceUtils.getEast(p);
                    int sC = PieceUtils.getSouth(p);
                    int wC = PieceUtils.getWest(p);

                    // Deduct from inventory (Piece is already on the board)
                    inventoryColorsLeft[nC]--;
                    inventoryColorsLeft[eC]--;
                    inventoryColorsLeft[sC]--;
                    inventoryColorsLeft[wC]--;

                    // Calculate demand: If the neighbor is an empty space (-1), we need a color!
                    if (r > 0 && localBoard[i - 16] == -1) {
                        openEdgesOnBoard[nC]++;
                    }
                    if (c < 15 && localBoard[i + 1] == -1) {
                        openEdgesOnBoard[eC]++;
                    }
                    if (r < 15 && localBoard[i + 16] == -1) {
                        openEdgesOnBoard[sC]++;
                    }
                    if (c > 0 && localBoard[i - 1] == -1) {
                        openEdgesOnBoard[wC]++;
                    }
                }
            }
        }

        /**
         * DYNAMIC LOOKAHEAD HEURISTIC
         * Scans the unplaced board. If any empty cell has 0 valid candidates left in the
         * remaining inventory, the current branch is topologically dead.
         */
        private boolean isTopologicallyViable(int[] localBoard, boolean[] localUsed) {
            // Sweep the entire board looking for empty spaces
            for (int i = 0; i < 256; i++) {
                if (localBoard[i] == -1) {
                    int r = i / 16;
                    int c = i % 16;

                    // Identify the mandatory edges surrounding this empty hole
                    int nReq = (r == 0) ? 0 : (localBoard[i - 16] != -1 ? PieceUtils.getSouth(localBoard[i - 16]) : CompatibilityIndex.WILDCARD);
                    int sReq = (r == 15) ? 0 : (localBoard[i + 16] != -1 ? PieceUtils.getNorth(localBoard[i + 16]) : CompatibilityIndex.WILDCARD);
                    int wReq = (c == 0) ? 0 : (localBoard[i - 1] != -1 ? PieceUtils.getEast(localBoard[i - 1]) : CompatibilityIndex.WILDCARD);
                    int eReq = (c == 15) ? 0 : (localBoard[i + 1] != -1 ? PieceUtils.getWest(localBoard[i + 1]) : CompatibilityIndex.WILDCARD);

                    // Optimization: If the hole is sitting in wide-open space, skip it to save CPU cycles
                    if (nReq == CompatibilityIndex.WILDCARD && sReq == CompatibilityIndex.WILDCARD &&
                            eReq == CompatibilityIndex.WILDCARD && wReq == CompatibilityIndex.WILDCARD) {
                        continue;
                    }

                    // Query the index using your existing mechanics
                    java.util.BitSet futureCandidates = compatIndex.candidatesFor(nReq, eReq, sReq, wReq);
                    compatIndex.andNotUsed(futureCandidates, localUsed);

                    // If this specific hole has ZERO pieces left that can fit, the branch is dead!
                    if (futureCandidates.isEmpty()) {
                        return false;
                    }
                }
            }
            return true; // The board geometry is mathematically safe
        }

        private boolean passesLookahead(int p, int step, int row, int col, int idx) {
            if (row < 15 && localBoard[idx + 16] == -1) {
                int reqN = PieceUtils.getSouth(p);
                int reqE = (col == 15) ? 0 : (localBoard[idx + 17] != -1 ? PieceUtils.getWest(localBoard[idx + 17]) :
                                              CompatibilityIndex.WILDCARD);
                int reqW = (col == 0) ? 0 : (localBoard[idx + 15] != -1 ? PieceUtils.getEast(localBoard[idx + 15]) :
                                             CompatibilityIndex.WILDCARD);
                int reqS = (row == 14) ? 0 : (localBoard[idx + 32] != -1 ? PieceUtils.getNorth(localBoard[idx + 32])
                                              : CompatibilityIndex.WILDCARD);

                java.util.BitSet southCandidates = compatIndex.candidatesFor(reqN, reqE, reqS, reqW);
                compatIndex.andNotUsed(southCandidates, localUsed);

                if (southCandidates.isEmpty()) {
                    return false;
                }
                if (step > 40 && row < 14 && localBoard[idx + 32] == -1) {
                    boolean secondaryFound = false;
                    for (int i = southCandidates.nextSetBit(0); i >= 0; i = southCandidates.nextSetBit(i + 1)) {
                        int testPiece = inventory.allOrientations[i];
                        int nextReqN = PieceUtils.getSouth(testPiece);
                        if (compatIndex.hasAnyCandidate(nextReqN, CompatibilityIndex.WILDCARD,
                                CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, localUsed)) {
                            secondaryFound = true;
                            break;
                        }
                    }
                    if (!secondaryFound) {
                        return false;
                    }
                }
            }

            if (col < 15 && localBoard[idx + 1] == -1) {
                int reqW = PieceUtils.getEast(p);
                int reqN = (row == 0) ? 0 : (localBoard[idx - 15] != -1 ? PieceUtils.getSouth(localBoard[idx - 15]) :
                                             CompatibilityIndex.WILDCARD);
                int reqE = (col == 14) ? 0 : (localBoard[idx + 2] != -1 ? PieceUtils.getWest(localBoard[idx + 2]) :
                                              CompatibilityIndex.WILDCARD);
                int reqS = (row == 15) ? 0 : (localBoard[idx + 17] != -1 ? PieceUtils.getNorth(localBoard[idx + 17])
                                              : CompatibilityIndex.WILDCARD);
                return compatIndex.hasAnyCandidate(reqN, reqE, reqS, reqW, localUsed);
            }
            return true;
        }
    }
}
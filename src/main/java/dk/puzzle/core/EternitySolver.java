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
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The central orchestrator for the Eternity II puzzle solver, managing a multiphase
 * hybrid search strategy involving CPU-based seed generation and GPU-accelerated
 * deep search and repair.
 */
public class EternitySolver implements Runnable {

    private final java.util.concurrent.atomic.AtomicInteger variantSaveThreshold = new java.util.concurrent.atomic.AtomicInteger(198);
    private long cumulativeTrials = 0;
    private final Set<Integer> savedHashes = ConcurrentHashMap.newKeySet();
    private static final Logger logger = LogManager.getLogger(EternitySolver.class);

    public static final int SEED_DEPTH = 110;

    public final int lnsThreshold;
    public final int hyperDiveThreshold;
    private volatile int localWallDepth = 0;

    public static final int GOLDMINE_THRESHOLD = 208;

    public static final long HYPER_DIVE_BUDGET = 4_000_000L;

    public static final long BASE_GPU_BUDGET = 25_000L;
    public static final long HYPER_GPU_BUDGET = 500_000L;

    private static final int[] HINT_POSITIONS = {221, 45, 210, 34};
    public static volatile int userCpuHandoffDepth = 8;
    public static volatile int userSurgeonHoles = 20;
    final SurgeonHeuristics surgeon;
    final int[] buildOrder = new int[256];
    final int[] bestBoard = new int[256];
    final int[] globalBestBoard = new int[256];
    final TopBoardRegistry topBoards = new TopBoardRegistry();
    private final ConcurrentHashMap<Integer, Integer> hashStrikeCount = new ConcurrentHashMap<>();
    private final Set<Integer> poisonedHashes = ConcurrentHashMap.newKeySet();

    private final PieceInventory inventory;
    private final CompatibilityIndex compatIndex;
    private final boolean lockCenter;

    private final ExecutorService executor;
    private final int numCores;
    private final java.util.concurrent.LinkedBlockingQueue<int[]> seedPool =
            new java.util.concurrent.LinkedBlockingQueue<>();
    private final Object displayLock = new Object();

    private final int[] flatResumeBoard = new int[256];
    private final boolean[] usedPhysicalPieces = new boolean[256];

    private int targetPiece;

    private final String saveProfile;
    private final BuildStrategy currentStrategy;

    private final AtomicLong globalCpuTrialCount = new AtomicLong(0);
    private final AtomicLong globalGpuTrialCount = new AtomicLong(0);
    private final java.util.concurrent.atomic.AtomicInteger eliteWins = new java.util.concurrent.atomic.AtomicInteger(0);
    private final java.util.concurrent.atomic.AtomicInteger diverseWins = new java.util.concurrent.atomic.AtomicInteger(0);
    private final java.util.concurrent.atomic.AtomicInteger restartWins = new java.util.concurrent.atomic.AtomicInteger(0);
    private final Set<Integer> uniqueMaxScoreHashes = ConcurrentHashMap.newKeySet();
    private final int[] hintPackedValues = new int[]{-1, -1, -1, -1};
    private final int[] hintPhysicalIndices = new int[]{-1, -1, -1, -1};
    private final long lastDisplayUpdateTime = 0;

    // >>> WATCHDOG & GUI THRESHOLDS <<<
    private final AtomicLong lastActivityTimestamp = new AtomicLong(System.currentTimeMillis());

    volatile int userBatchSizeOverride = -1;
    volatile int absoluteHighScore = 0;
    volatile int deepestStep = 0;
    volatile boolean manualOverrideRequested = false;
    volatile int manualBaseCampTarget = 0;
    volatile double extinctionThreshold = 0.98;
    volatile int stagnationLimitMinutes = 20;
    private long lastVisualUpdate = System.currentTimeMillis();
    private GpuEngine gpuEngine;

    private boolean useGpu;
    private long methodStartTime = 0;
    private ExecutorService gpuExecutor;
    private int fatalGpuBugCount = 0;
    private int centerPhysicalIdx = -1;
    private volatile int highestP2DepthThisCycle = 0;
    private volatile int currentSeedDepth = SEED_DEPTH;
    private int currentRepairIteration = 0;
    private int consecutiveExtinctions = 0;
    private volatile int lastReportedDepth = 0;
    private int consecutiveGpuStagnation = 0;
    private int phaseCounter = 0;
    private long lastThroughputReportTime = System.currentTimeMillis();
    private long totalRepairVariationsTested = 0;
    private long repairLoopsCounter = 0;
    private volatile boolean isGpuBusy = false;
    private volatile int poisonedIndex = -1;
    private volatile int poisonedPiece = -1;
    private volatile int poisonExpiryExtinction = -1;
    private ScheduledExecutorService repairReporterScheduler;
    private int watchdogRecoveryAttempts = 0;

    private volatile long lastSeedGrowthTime = System.currentTimeMillis();
    private volatile int lastSeedCount = 0;

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

        if (this.lockCenter) {
            this.lnsThreshold = 140;
            this.hyperDiveThreshold = 175;
            logger.info(String.format(">>> [DIFFICULTY] Constrained Board. LNS @ %d, Hyper-Dive @ %d", lnsThreshold, hyperDiveThreshold));
        } else {
            this.lnsThreshold = 200;
            this.hyperDiveThreshold = 200;
            logger.info(String.format(">>> [DIFFICULTY] Unconstrained Board. LNS @ %d, Hyper-Dive @ %d", lnsThreshold, hyperDiveThreshold));
        }

        this.saveProfile = strategy.name() + (lockCenter ? "_LOCKED" : "_UNLOCKED");
        this.numCores = Runtime.getRuntime().availableProcessors();
        this.executor = Executors.newFixedThreadPool(numCores);
        logger.info(String.format(">>> Multithreading active with %d cores.", numCores));

        if (strategy == BuildStrategy.SPIRAL) {
            generateSpiralOrder();
        } else {
            for (int i = 0; i < 256; i++) {
                buildOrder[i] = i;
            }
        }

        if (this.lockCenter) {
            List<Integer> newOrder = new ArrayList<>();
            newOrder.add(135);
            for (int hPos : HINT_POSITIONS) {
                newOrder.add(hPos);
            }

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
            for (int i = 0; i < 256; i++) {
                buildOrder[i] = newOrder.get(i);
            }
            logger.info(">>> [ARCHITECTURE] Build Order optimized: Static locks shielded from GPU.");
        }

        this.compatIndex = new CompatibilityIndex(inventory.allOrientations, inventory.physicalMapping);
        this.surgeon = new SurgeonHeuristics(lockCenter, 0.70);

        loadCheckpointAndHints();
    }

    public void setVariantSaveThreshold(int threshold) {
        this.variantSaveThreshold.set(threshold);
    }

    private void loadCheckpointAndHints() {
        SolverState loadedState = CheckpointManager.loadSmartState(saveProfile);

        if (loadedState != null) {
            restoreBoardState(loadedState.bestBoard);
            this.uniqueMaxScoreHashes.addAll(loadedState.uniqueMaxScoreHashes);
            for (int[] historicBoard : loadedState.topBoardsRegistry) {
                this.topBoards.offer(historicBoard, loadedState.score);
            }
            this.cumulativeTrials = loadedState.cumulativeTrials;
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
                this.targetPiece = centerPacked;
                flatResumeBoard[135] = centerPacked;
                bestBoard[135] = centerPacked;
                globalBestBoard[135] = centerPacked;
                this.centerPhysicalIdx = centerPhysId;
                logger.info(String.format(">>> [CENTER LOCKED] Successfully locked Center (Piece 139) at position 135"));
            } else {
                logger.error(">>> [FATAL CONFIG] Could not find Center Piece 139 in inventory!");
            }

            int[][] officialHints = {
                    {57 - 1, 3, 221}, // Piece 57, Rot 3, Pos 221
                    {50 - 1, 3, 45},  // Piece 50, Rot 3, Pos 45
                    {22 - 1, 2, 210}, // Piece 22, Rot 2, Pos 210
                    {255 - 1, 1, 34}  // Piece 255, Rot 1, Pos 34
            };

            for (int h = 0; h < 4; h++) {
                int physId = officialHints[h][0];
                int rot = officialHints[h][1];
                int boardPos = officialHints[h][2];
                int foundPacked = -1;

                for (int oi = 0; oi < 1024; oi++) {
                    if (inventory.physicalMapping[oi] == physId && (oi % 4) == rot) {
                        foundPacked = inventory.allOrientations[oi];
                        hintPhysicalIndices[h] = physId;
                        break;
                    }
                }

                if (foundPacked != -1) {
                    hintPackedValues[h] = foundPacked;
                    bestBoard[boardPos] = foundPacked;
                    globalBestBoard[boardPos] = foundPacked;
                    flatResumeBoard[boardPos] = foundPacked;
                    logger.info(String.format(">>> [HINT LOCKED] Locked Hint %d (Piece %d) at pos %d", (h + 1), (physId + 1), boardPos));
                } else {
                    logger.error(String.format(">>> [FATAL CONFIG] Could not find Hint Piece %d in inventory!", (physId + 1)));
                }
            }
        } else {
            Arrays.fill(hintPackedValues, -1);
            Arrays.fill(hintPhysicalIndices, -1);
            logger.info(">>> [UNCONSTRAINED] Checkbox is off. Running completely without Center Piece or Hints!");
        }
        updateDisplay(absoluteHighScore, buildDisplayBoard(globalBestBoard));
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
            this.localWallDepth = loadedCount;
            System.arraycopy(bestBoard, 0, flatResumeBoard, 0, 256);
            updateDisplay(absoluteHighScore, buildDisplayBoard(globalBestBoard));
        }
    }

    public void setStagnationLimit(int minutes) { this.stagnationLimitMinutes = minutes; }
    public void setBatchSizeOverride(int size) { this.userBatchSizeOverride = size; }
    public void setExtinctionThreshold(double threshold) { this.extinctionThreshold = threshold; }
    public void setTargetedHolesPercentage(double percentage) {
        if (this.surgeon != null) { this.surgeon.setTargetedHolesPercentage(percentage); }
    }

    public void triggerManualOverride(int targetBaseCamp) {
        this.manualBaseCampTarget = targetBaseCamp;
        this.manualOverrideRequested = true;
    }

    @Override
    public void run() {
        lastActivityTimestamp.set(System.currentTimeMillis());
        startWatchdogThread();
        startReporterThread();
        setupShutdownHook();

        logger.info("Starting Solver Orchestrator...");

        initializeGpuEngine();

        updateDisplay(countPieces(bestBoard), buildDisplayBoard(bestBoard));

        if (this.absoluteHighScore > 0) {
            logger.info(">>> [BOOT] Checkpoint detected! Setting up Base Camp to resume search...");
            triggerBranchScrap();
        }

        run3PhasePipeline();
    }

    private <T> T waitForGpu(Future<T> future, long timeoutMinutes) throws Exception {
        long deadline = System.currentTimeMillis() + timeoutMinutes * 60_000L;
        while (true) {
            long remaining = deadline - System.currentTimeMillis();
            if (remaining <= 0) throw new java.util.concurrent.TimeoutException("GPU timed out after " + timeoutMinutes + " minutes");
            try {
                T result = future.get(Math.min(30_000L, remaining), java.util.concurrent.TimeUnit.MILLISECONDS);
                lastActivityTimestamp.set(System.currentTimeMillis());
                return result;
            } catch (java.util.concurrent.TimeoutException e) {
                lastActivityTimestamp.set(System.currentTimeMillis()); // heartbeat: GPU is still running
            }
        }
    }

    private void startWatchdogThread() {
        Thread watchdog = new Thread(() -> {
            while (!Thread.currentThread().isInterrupted()) {
                try {
                    Thread.sleep(30000);
                    long inactiveDuration = System.currentTimeMillis() - lastActivityTimestamp.get();
                    if (inactiveDuration > 180000) {
                        watchdogRecoveryAttempts++;
                        logger.error(String.format(">>> [WATCHDOG] Solver unresponsive for %ds. Recovery attempt #%d.", inactiveDuration / 1000, watchdogRecoveryAttempts));

                        if (watchdogRecoveryAttempts > 2) {
                            logger.error(">>> [WATCHDOG] Recovery failed after " + watchdogRecoveryAttempts + " attempts. Forcing exit.");
                            System.exit(1);
                        }

                        try {
                            isGpuBusy = false;
                            seedPool.clear();
                            if (useGpu) {
                                rebootGpuEngine();
                            }
                            triggerBranchScrap();
                        } catch (Exception recoveryEx) {
                            logger.error(">>> [WATCHDOG] Recovery threw exception: " + recoveryEx.getMessage());
                        }

                        lastActivityTimestamp.set(System.currentTimeMillis());
                        logger.warn(">>> [WATCHDOG] Recovery complete. Resuming search from cleared base camp.");
                    } else {
                        watchdogRecoveryAttempts = 0;
                    }
                } catch (InterruptedException ignored) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        });
        watchdog.setDaemon(true);
        watchdog.start();
    }

    private int getDynamicBatchSize() {
        // Reduceret let ved ekstreme dybder, da udvalget er ekstremt snævert.
        if (deepestStep >= hyperDiveThreshold) return 2500; // Nedsat fra 8192
        if (deepestStep >= hyperDiveThreshold - 20) return 10000; // Nedsat fra 16384
        if (deepestStep >= lnsThreshold / 2) return 25000;
        return 50000;
    }

    private void initializeGpuEngine() {
        if (this.useGpu) {
            this.gpuExecutor = Executors.newSingleThreadExecutor();
            try {
                this.gpuEngine = this.gpuExecutor.submit(() -> new GpuEngine(inventory, lockCenter))
                        .get(30, java.util.concurrent.TimeUnit.SECONDS);
                logger.info(">>> [HARDWARE] NVIDIA CUDA GPU detected and initialized successfully on dedicated thread.");
            } catch (java.util.concurrent.TimeoutException te) {
                logger.error(">>> [CRITICAL] GPU driver unresponsive during initialization! Falling back to CPU Mode.");
                this.useGpu = false;
                this.gpuEngine = null;
                if (this.gpuExecutor != null) {
                    this.gpuExecutor.shutdownNow();
                }
            } catch (Throwable t) {
                this.useGpu = false;
                this.gpuEngine = null;
                if (this.gpuExecutor != null) {
                    this.gpuExecutor.shutdownNow();
                }
                logger.warn(String.format(">>> [HARDWARE] GPU initialization failed: %s", t.getMessage()));
            }
        }
    }

    private void rebootGpuEngine() {
        logger.warn(">>> [HARDWARE] Rebooting GPU Engine to escape kernel deadlock...");
        if (this.gpuExecutor != null) {
            this.gpuExecutor.shutdownNow();
        }

        try {
            Thread.sleep(500);
        } catch (InterruptedException ignored) {
        }

        System.gc();
        initializeGpuEngine();
    }

    private void setupShutdownHook() {
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            logger.info(">>> Shutdown hook: Saving final checkpoint...");
            synchronized (displayLock) {
                if (repairReporterScheduler != null) {
                    repairReporterScheduler.shutdownNow();
                }
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

        while (true) {
            lastActivityTimestamp.set(System.currentTimeMillis()); // Watchdog Heartbeat
            try {
                if (manualOverrideRequested) {
                    manualOverrideRequested = false;
                    retreat(manualBaseCampTarget, ">>> User Override...");
                    try { Thread.sleep(1000); } catch (InterruptedException ignored) {}
                }

                int activeBatch = getDynamicBatchSize();
                if (userBatchSizeOverride > 0) {
                    activeBatch = userBatchSizeOverride;
                }

                if (this.useGpu && this.gpuEngine != null) {
                    if (deepestStep >= lnsThreshold) {
                        phaseCounter++;
                        boolean seedsReady = seedPool.size() >= activeBatch;

                        int refreshInterval = (consecutiveExtinctions > 10) ? 2 : 4;
                        if (phaseCounter % refreshInterval == 0 && seedsReady) {
                            runPhase2_GpuDeepDfs();
                            if (seedPool.size() > activeBatch * 2) {
                                logger.info(">>> Phase 2 traffic jam (LNS Mode)! Clearing excess seeds.");
                                seedPool.clear();
                            }
                        } else if (!seedsReady) {
                            runPhase1_CpuSeedGen();
                        } else if (deepestStep >= absoluteHighScore - 10) {
                            runPhase3_GpuSurgeon();
                            phaseCounter++;
                        } else {
                            phaseCounter = 0;
                        }
                    } else if (seedPool.size() >= activeBatch) {
                        runPhase2_GpuDeepDfs();
                        if (seedPool.size() > activeBatch * 2) {
                            logger.info(">>> Phase 2 traffic jam (Standard Mode)! Clearing excess seeds.");
                            seedPool.clear();
                        }
                    } else {
                        runPhase1_CpuSeedGen();
                    }
                } else {
                    runPhase1_CpuSeedGen();
                }

                if (System.currentTimeMillis() - lastPeriodicSave > 300_000) {
                    synchronized (displayLock) {
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
            } catch (Exception e) {
                logger.error(">>> [FATAL ERROR] PIPELINE CRASHED: ", e);
                try { Thread.sleep(5000); } catch (InterruptedException ignored) {}
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

        // Cap currentSeedDepth so CPU workers only need to place a few pieces
        // beyond the base camp. At depth 197, lockedPieces+2=199 is nearly
        // impossible for CPU workers to reach. Cap at lockedPieces+2 but never
        // more than lockedPieces+2 — and rely on seed padding for GPU saturation.
        if (lockedPieces == 0) {
            currentSeedDepth = 14;
            logger.info(String.format(">>> [PHASE 1] Empty board. Setting fast GPU handoff depth to: %d", currentSeedDepth));
        } else {
            // Always just 2 steps beyond base camp — CPU places 2 pieces then hands to GPU.
            // Seed padding (MIN_GPU_THREADS=8192) fills remaining GPU capacity.
            currentSeedDepth = lockedPieces + 2;
        }

        try {
            int activeBatch = getDynamicBatchSize();
            if (userBatchSizeOverride > 0) {
                activeBatch = userBatchSizeOverride;
            }

            int numVanguard = 0;
            if (consecutiveExtinctions > 5) {
                numVanguard = Math.max(1, numCores / 4);
            } else if (consecutiveExtinctions > 2) {
                numVanguard = Math.max(1, numCores / 8);
            }

//            if (numVanguard > 0) {
//                logger.info(String.format(">>> [GENETIC EXPLORATION] %d%% Vanguard Threads. Hard-banning recently used pieces (Extinction level: %d | Local Wall: %d).",
//                        (numVanguard*100/numCores), consecutiveExtinctions, localWallDepth));
//            }

//            int initialPoolSize = seedPool.size();

            List<CpuSearchWorker> workers = new ArrayList<>();
            for (int i = 0; i < numCores; i++) {
                boolean isVanguard = i < numVanguard;
                workers.add(new CpuSearchWorker(activeBatch, isVanguard));
            }

            // >>> TIMEOUT TILFØJET: 2.5 Sekunder <<<
            // CPU'en får nu maksimalt 2,5 sekunder til at finde sine brikker. Hvis den ikke kan
            // opfylde den massive GPU kvote (fordi brættet er for svært), afbrydes trådene
            // og vi sender simpelthen bare de seeds vi fandt til GPU'en i stedet for at fryse.
            long timeoutMillis = this.useGpu ? 2500 : Long.MAX_VALUE;
            List<Future<Boolean>> futures = executor.invokeAll(workers, timeoutMillis, java.util.concurrent.TimeUnit.MILLISECONDS);

            if (this.useGpu) {
                if (seedPool.isEmpty()) {
                    logger.warn(">>> [PHASE 1 DEADLOCK] CPU ledte i 2.5 sekunder og fandt 0 seeds. Base Camp er ufrugtbar!");
                    consecutiveExtinctions++;
                    triggerBranchScrap();
                } else if (seedPool.size() < activeBatch) {
                    // Timeout blev ramt, men vi fik nogle få seeds. Sender dem videre!
                    // Valgfri info log fjernet for at undgå spam.
                }
            } else {
                boolean solved = false;
                for (Future<Boolean> f : futures) {
                    try {
                        if (!f.isCancelled() && f.get()) solved = true;
                    } catch (Exception ignored) {}
                }
                if (!solved) {
                    logger.warn(">>> [CPU DEADLOCK] Udtømte grenen uden at finde en løsning. Tearing down...");
                    consecutiveExtinctions++;
                    triggerBranchScrap();
                }
            }

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private int getGoldmineThreshold() {
        return Math.max(lnsThreshold, absoluteHighScore - 3);
    }

    private void runPhase2_GpuDeepDfs() {
        methodStartTime = System.currentTimeMillis();

        int activeBatch = getDynamicBatchSize();
        if (userBatchSizeOverride > 0) {
            activeBatch = userBatchSizeOverride;
        }

        List<int[]> seeds = new ArrayList<>();
        // Trækker op til activeBatch seeds ud. Hvis timeouten fra Phase 1 ramte,
        // kan der sagtens være færre, hvilket er helt fint.
        for (int i = 0; i < activeBatch; i++) {
            int[] s = seedPool.poll();
            if (s != null) {
                seeds.add(s);
            }
        }
        if (seeds.isEmpty()) {
            return;
        }

        // Pad to minimum GPU thread count by duplicating seeds.
        // At deep depths CPU generates few seeds — duplication fills the GPU.
        // Duplicate seeds diverge naturally because DFS explores different piece
        // orderings from the same position.
        final int MIN_GPU_THREADS = 8192;
        if (seeds.size() < MIN_GPU_THREADS) {
            int originalSize = seeds.size();
            while (seeds.size() < MIN_GPU_THREADS) {
                seeds.add(seeds.get(seeds.size() % originalSize).clone());
            }
        }

        logger.info(String.format(">>> [PHASE 2] Overdrager %,d seeds til GPU'en. Beregner... (Dette tager lidt tid)", seeds.size()));

        int[] bestBoardOut = new int[256];
        Arrays.fill(bestBoardOut, -1);
        isGpuBusy = true;

        GpuEngine.GpuResult result = null;

        final int goldmineThreshold = Math.max(lnsThreshold, absoluteHighScore - 3);
        final long currentGpuBudget;
        if (deepestStep >= hyperDiveThreshold) {
            currentGpuBudget = HYPER_GPU_BUDGET;
            logger.info(String.format(">>> [HYPER-DIVE ACTIVATED] Base Camp is at depth %d. Granting GPU maximum budget (%,d steps)!", deepestStep, currentGpuBudget));
        } else if (deepestStep >= goldmineThreshold) {
            currentGpuBudget = 200_000L;
        } else if (deepestStep >= lnsThreshold) {
            currentGpuBudget = 75_000L;
        } else {
            currentGpuBudget = 35_000L;
        }

        try {
            Future<GpuEngine.GpuResult> future = gpuExecutor.submit(() ->
                    gpuEngine.runDeepDfs(seeds, currentSeedDepth, deepestStep, bestBoardOut, buildOrder)
            );
            result = waitForGpu(future, 10);
        } catch (Exception e) {
            isGpuBusy = false;
            rebootGpuEngine();
            triggerBranchScrap();
            return;
        }

        if (result == null) {
            isGpuBusy = false;
            return;
        }

        applyStaticLocks(bestBoardOut);

        int maxDepthInBatch = 0;
        int bestSeedIndex = 0;
        int[] threadDepths = result.threadDepths(); // >>> FIX: Rettet til field access for at matche GpuEngine

        List<Integer> bestIndices = new ArrayList<>();
        for (int i = 0; i < threadDepths.length; i++) {
            if (threadDepths[i] > maxDepthInBatch) {
                maxDepthInBatch = threadDepths[i];
                bestIndices.clear();
                bestIndices.add(i);
            } else if (threadDepths[i] == maxDepthInBatch) {
                bestIndices.add(i);
            }
        }
        if (!bestIndices.isEmpty()) {
            bestSeedIndex = bestIndices.get(new Random().nextInt(bestIndices.size()));
        }

        if (maxDepthInBatch > highestP2DepthThisCycle) {
            highestP2DepthThisCycle = maxDepthInBatch;
        }

        if (maxDepthInBatch <= currentSeedDepth) {
            logger.warn(String.format(">>> [GPU DEADLOCK] GPU'en testede %,d seeds og kom 0 brikker frem (Dybde %d). Tearing down øjeblikkeligt...", seeds.size(), currentSeedDepth));
            consecutiveExtinctions++;
            triggerBranchScrap();
            isGpuBusy = false;
            return;
        }

        if (maxDepthInBatch > currentSeedDepth && maxDepthInBatch >= getGoldmineThreshold() && bestSeedIndex < seeds.size()) {
            int[] goldmineSeed = seeds.get(bestSeedIndex);
            logger.info(String.format(">>> [GOLDMINE] Tråd %d nåede dybde %d (startede ved %d)! Starter hyper-dyk med %,d steps...",
                    bestSeedIndex, maxDepthInBatch, currentSeedDepth, HYPER_DIVE_BUDGET));

            int[] hyperBestOut = new int[256];
            Arrays.fill(hyperBestOut, -1);
            try {
                Future<GpuEngine.GpuResult> hyperFuture = gpuExecutor.submit(() ->
                        gpuEngine.runDeepDfs(Collections.singletonList(goldmineSeed), currentSeedDepth, deepestStep, hyperBestOut, buildOrder)
                );
                GpuEngine.GpuResult hyperResult = waitForGpu(hyperFuture, 15);

                if (hyperResult != null) {
                    globalGpuTrialCount.addAndGet(hyperResult.stepsTaken()); // Rettet til field access

                    if (hyperResult.solved()) { // Rettet til field access
                        handleVictory(hyperBestOut, inventory);
                    }

                    if (hyperResult.newHighScore() > deepestStep) { // Rettet til field access
                        int javaCountedScore = countPieces(hyperBestOut);
                        int validScore = Math.min(javaCountedScore, hyperResult.newHighScore());

                        if (validScore > deepestStep) {
                            applyStaticLocks(hyperBestOut);
                            if (verifyBoardStrict(hyperBestOut)) {
                                deepestStep = validScore;
                                System.arraycopy(hyperBestOut, 0, bestBoard, 0, 256);
                                topBoards.offer(hyperBestOut, deepestStep);
                                updateDisplay(deepestStep, buildDisplayBoard(bestBoard));

                                if (deepestStep > localWallDepth) {
                                    logger.info(String.format(">>> [WALL BROKEN] Gammel mur: %d. Ny mur etableret ved: %d", localWallDepth, deepestStep));
                                    localWallDepth = deepestStep;
                                    consecutiveExtinctions = 0;
                                    consecutiveGpuStagnation = 0;
                                } else {
                                    consecutiveGpuStagnation = 0;
                                }

                                logger.info(String.format(">>> [HYPER-DYK SUCCES] Nåede %d brikker! Hash: %08X",
                                        deepestStep, Arrays.hashCode(bestBoard)));

                                if (deepestStep > absoluteHighScore) {
                                    absoluteHighScore = deepestStep;
                                    System.arraycopy(hyperBestOut, 0, globalBestBoard, 0, 256);
                                    int hash = Arrays.hashCode(globalBestBoard);
                                    logger.info(String.format(">>> [NEW GLOBAL RECORD VIA HYPER-DIVE] %d / 256 | Hash: %08X",
                                            absoluteHighScore, hash));
                                    logger.info(String.format(">>> Total Trials to reach this milestone: %,d", cumulativeTrials));
                                    RecordManager.saveRecord(buildDisplayBoard(globalBestBoard), absoluteHighScore, saveProfile);
                                    try {
                                        saveAndUploadBucasLink(globalBestBoard, absoluteHighScore);
                                    } catch (Exception e) {
                                        logger.warn(">>> Skipping Google Drive upload.");
                                    }
                                    analyzeFullBoardPotential(globalBestBoard);
                                }
                            } else {
                                logger.warn(">>> [HYPER-DYK] Brættet fejlede verifikation — ignorerer.");
                            }
                        }
                    } else {
                        logger.info(String.format(">>> [HYPER-DYK] Ingen fremgang fra dybde %d. Goldmine udtømt.",
                                maxDepthInBatch));
                    }
                }
            } catch (Exception e) {
                logger.warn(String.format(">>> [HYPER-DYK FEJL] %s", e.getMessage()));
            }
        }

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
        globalGpuTrialCount.addAndGet(result.stepsTaken()); // Rettet til field access
        isGpuBusy = false;

        long elapsed = System.currentTimeMillis() - methodStartTime;
        logger.info(String.format(">>> GPU Phase 2 complete. Steps taken per second: %,d",
                Math.round((double) result.stepsTaken() * 1000) / Math.max(1, elapsed))); // Rettet til field access

        if (result.solved()) { // Rettet til field access
            handleVictory(bestBoardOut, inventory);
        }

        // >>> FIX: Check med korrekt wrapper-logik, så Java ikke tæller affald
        if (result.newHighScore() > deepestStep) { // Rettet til field access
            int javaCountedScore = countPieces(bestBoardOut);
            if (javaCountedScore != result.newHighScore()) {
                logger.warn(">>> [ADVARSEL] Fase 2 GPU rapporterede " + result.newHighScore() + " men Java talte " + javaCountedScore + "! Retter score...");
            }
            int validScore = Math.min(javaCountedScore, result.newHighScore());

            if (validScore > deepestStep) {
                deepestStep = validScore;

                if (deepestStep > localWallDepth) {
                    logger.info(String.format(">>> [WALL BROKEN] Ny mur sat ved: %d", deepestStep));
                    localWallDepth = deepestStep;
                    consecutiveExtinctions = 0;
                    consecutiveGpuStagnation = 0;
                } else {
                    consecutiveGpuStagnation = 0;
                }

                System.arraycopy(bestBoardOut, 0, bestBoard, 0, 256);
                updateDisplay(deepestStep, buildDisplayBoard(bestBoard));

//                if (deepestStep > variantSaveThreshold.get()) {
//                    int hash = Arrays.hashCode(bestBoard);
//                    if (savedVariantHashes.add(hash)) {
//                        logger.info(">>> [HIGH DEPTH VARIANT] Unique %d-piece board detected! Generating Full Board Variant...", deepestStep);
//                        logger.info(">>> Total Trials to reach this milestone: " + String.format("%,d", cumulativeTrials));
//                        analyzeFullBoardPotential(bestBoard, deepestStep);
//                    }
//                }

                if (deepestStep > lastReportedDepth) {
                    logger.info(String.format(">>> [CLIMBING] Depth: %d / 256 | Board Hash: %08X", deepestStep,
                            Arrays.hashCode(bestBoard)));
                    lastReportedDepth = deepestStep;
                }

                // Check if this board is a gravity well — seen too many times without progress.
                // If so, poison it and force a nuclear retreat to break out of the loop.
                int boardHash = Arrays.hashCode(bestBoard);
                if (checkPoisonAndRetreat(boardHash, deepestStep)) {
                    return; // retreat already triggered inside checkPoisonAndRetreat
                }

                if (deepestStep > absoluteHighScore) {
                    absoluteHighScore = deepestStep;
                    System.arraycopy(bestBoardOut, 0, globalBestBoard, 0, 256);

                    int hash = Arrays.hashCode(globalBestBoard);
                    logger.info(String.format(">>> [NEW GLOBAL RECORD] Depth: %d / 256 | Board Hash: %08X", absoluteHighScore, hash));
                    logger.info(String.format(">>> Total Trials to reach this milestone: %,d", cumulativeTrials));

                    uniqueMaxScoreHashes.clear();
                    uniqueMaxScoreHashes.add(hash);

                    RecordManager.saveRecord(buildDisplayBoard(globalBestBoard), absoluteHighScore, saveProfile);
                    try {
                        saveAndUploadBucasLink(globalBestBoard, absoluteHighScore);
                    } catch (Exception e) {
                        logger.warn(">>> Skipping Google Drive upload: Not connected or unavailable.");
                    }
                    analyzeFullBoardPotential(globalBestBoard);
                }
            } else {
                consecutiveGpuStagnation++;
            }
        } else {
            consecutiveGpuStagnation++;
        }

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

        int numClones = (deepestStep >= hyperDiveThreshold) ? 100000 : 250000;
        int holesToPunch = userSurgeonHoles;

        currentRepairIteration++;
        repairLoopsCounter++;
        totalRepairVariationsTested += numClones;

        int[] sourceBoard = topBoards.nextForRepair();
        if (sourceBoard == null) {
            sourceBoard = bestBoard;
        }

        int actualHoles;
        if (consecutiveExtinctions > 20) {
            actualHoles = Math.min(holesToPunch * 2, deepestStep / 5);
        } else if (deepestStep > 240) {
            actualHoles = Math.max(3, holesToPunch / 4);
        } else if (deepestStep > lnsThreshold) {
            actualHoles = holesToPunch;
        } else {
            actualHoles = holesToPunch;
        }
        actualHoles = Math.clamp(actualHoles, 1, deepestStep - 5);
        List<int[]> variations = surgeon.excavateFrontier(
                sourceBoard, numClones, actualHoles, currentRepairIteration, deepestStep, buildOrder);

        int[] bestBoardOut = new int[256];
        Arrays.fill(bestBoardOut, -1);

        GpuEngine.GpuResult result = null;

        long currentGpuBudget = (deepestStep >= hyperDiveThreshold) ? HYPER_GPU_BUDGET : BASE_GPU_BUDGET;

        if (deepestStep >= hyperDiveThreshold) {
            logger.info(String.format(">>> [HYPER-DIVE ACTIVATED] Surgeon Base Camp is at depth %d. Granting GPU maximum budget (%,d steps)!", deepestStep, currentGpuBudget));
        }

        try {
            Future<GpuEngine.GpuResult> future = gpuExecutor.submit(() ->
                    gpuEngine.runRepairMode(variations, deepestStep, bestBoardOut)
            );
            result = waitForGpu(future, 10);
        } catch (Exception e) {
            logger.error(">>> Phase 3 Error: " + e.getMessage());
        }

        isGpuBusy = false;
        long calculationElapsed = System.currentTimeMillis() - start;

        if (result != null) {
            logger.info(String.format(">>> GPU Phase 3 complete. Steps taken per second: %,d",
                    Math.round((double) result.stepsTaken() * 1000) / Math.max(1, calculationElapsed))); // Rettet til field access

            if (result.solved()) { // Rettet til field access
                handleVictory(bestBoardOut, inventory);
            }

            // >>> FIX: Tjek og tæl KUN brættet, hvis GPU'en rent faktisk har slået rekorden <<<
            if (result.newHighScore() > deepestStep) { // Rettet til field access
                int javaCountedScore = countPieces(bestBoardOut);
                if (javaCountedScore != result.newHighScore()) {
                    logger.warn(">>> [ADVARSEL] Fase 3 GPU rapporterede " + result.newHighScore() + " men Java talte " + javaCountedScore + "! Retter score...");
                }
                int validScore = Math.min(javaCountedScore, result.newHighScore());

                if (validScore > deepestStep) {
                    if (!verifyBoardStrict(bestBoardOut)) {
                        logger.error(">>> [FATAL GPU BUG] The GPU returned a board with an illegal edge conflict!");
                        int piecesPlaced = countPieces(bestBoardOut);

                        if (piecesPlaced >= 240) {
                            logger.info(String.format(">>> [RESCUE] High-score board (%d) detected. Executing emergency surgery...", piecesPlaced));
                            int validPieces = rescueBoard(bestBoardOut);
                            if (validPieces > deepestStep) {
                                deepestStep = validPieces;
                                System.arraycopy(bestBoardOut, 0, bestBoard, 0, 256);
                                topBoards.offer(bestBoardOut, deepestStep);
                                updateDisplay(deepestStep, buildDisplayBoard(bestBoard));

                                if (deepestStep > localWallDepth) {
                                    localWallDepth = deepestStep;
                                    consecutiveExtinctions = 0;
                                }
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

                    int hash = Arrays.hashCode(bestBoardOut);
                    logger.info(String.format(">>> [NEW GLOBAL RECORD] Depth: %d / 256 | Board Hash: %08X", validScore, hash));
                    logger.info(String.format(">>> Total Trials to reach this milestone: %,d", cumulativeTrials));

//                    if (deepestStep > variantSaveThreshold.get()) {
//                        int hash = Arrays.hashCode(bestBoard);
//                        if (savedVariantHashes.add(hash)) {
//                            logger.info(">>> [HIGH DEPTH VARIANT] Unique %d-piece board detected! Generating Full Board Variant...", deepestStep);
//                            analyzeFullBoardPotential(bestBoard, deepestStep);
//                        }
//                    }

                    deepestStep = validScore;

                    if (deepestStep > absoluteHighScore) {
                        absoluteHighScore = deepestStep;
                        lastReportedDepth = absoluteHighScore;
                        System.arraycopy(bestBoardOut, 0, globalBestBoard, 0, 256);
                        uniqueMaxScoreHashes.clear();
                        uniqueMaxScoreHashes.add(hash);
                        RecordManager.saveRecord(buildDisplayBoard(globalBestBoard), absoluteHighScore, saveProfile);
                        try {
                            saveAndUploadBucasLink(globalBestBoard, absoluteHighScore);
                        } catch (Exception e) {
                            logger.warn(">>> Skipping Google Drive upload.");
                        }
                        analyzeFullBoardPotential(globalBestBoard);
                    }

                    consecutiveGpuStagnation = 0;

                    if (deepestStep > localWallDepth) {
                        localWallDepth = deepestStep;
                        consecutiveExtinctions = 0;
                    }

                    System.arraycopy(bestBoardOut, 0, bestBoard, 0, 256);
                    topBoards.offer(bestBoardOut, deepestStep);
                    updateDisplay(deepestStep, buildDisplayBoard(bestBoard));
                } else {
                    consecutiveGpuStagnation++;
                }
            } else {
                consecutiveGpuStagnation++;
            }
        } else {
            consecutiveGpuStagnation++;
        }

        if (consecutiveGpuStagnation >= 4) {
            logger.warn(">>> [!!!] Phase 3 GPU stagnated! Activating Teardown...");
            consecutiveExtinctions++;
            triggerBranchScrap();
        }
    }

    private boolean checkPoisonAndRetreat(int currentBoardHash, int currentDepth) {
        if (currentDepth < hyperDiveThreshold) {
            return false;
        }

        if (poisonedHashes.contains(currentBoardHash)) {
            logger.warn(String.format(">>> POISONED BOARD DETECTED! (%08X). Executing Nuclear Retreat!", currentBoardHash));
            executeNuclearRetreat(currentDepth);
            return true;
        }

        int strikes = hashStrikeCount.getOrDefault(currentBoardHash, 0) + 1;
        hashStrikeCount.put(currentBoardHash, strikes);

        if (strikes >= 3) {
            logger.error(String.format(">>> GRAVITY WELL DETECTED! Poisoning hash: %08X", currentBoardHash));
            poisonedHashes.add(currentBoardHash);
            executeNuclearRetreat(currentDepth);
            return true;
        }

        return false;
    }

    private String generateHashForBoard(int[] boardArray) {
        if (boardArray == null) {
            return "EMPTY_BOARD";
        }
        return Integer.toHexString(java.util.Arrays.hashCode(boardArray)).toUpperCase();
    }

    private void executeNuclearRetreat(int currentDepth) {
        int nuclearTargetDepth = Math.max(150, currentDepth - 35);
        logger.warn(String.format(">>> NUCLEAR RETREAT: Tearing board down to depth %d", nuclearTargetDepth));

        deepestStep = nuclearTargetDepth;
        localWallDepth = deepestStep;
        consecutiveExtinctions = 0;

        for (int s = deepestStep; s < 256; s++) {
            bestBoard[buildOrder[s]] = -1;
        }

        Arrays.fill(flatResumeBoard, -1);
        for (int step = 0; step < deepestStep; step++) {
            int idx = buildOrder[step];
            boolean isStatic = (lockCenter && idx == 135);
            if (lockCenter) {
                for (int hPos : HINT_POSITIONS) {
                    if (idx == hPos) {
                        isStatic = true;
                        break;
                    }
                }
            }
            if (!isStatic) {
                flatResumeBoard[idx] = bestBoard[idx];
            }
        }
        applyStaticLocks(flatResumeBoard);

        injectEntropy(nuclearTargetDepth);

        seedPool.clear();
        updateDisplay(deepestStep, buildDisplayBoard(flatResumeBoard));
    }

    private void injectEntropy(int currentDepth) {
        java.util.Random rand = new java.util.Random();
        int piecesToRip = rand.nextInt(3) + 1;

        for (int i = 0; i < piecesToRip; i++) {
            int randomDepthToRip = currentDepth - rand.nextInt(15);
            if (randomDepthToRip > 0 && randomDepthToRip < 256) {
                int boardPos = buildOrder[randomDepthToRip];

                boolean isStatic = (lockCenter && boardPos == 135);
                for (int hPos : HINT_POSITIONS) {
                    if (boardPos == hPos) {
                        isStatic = true;
                        break;
                    }
                }

                if (!isStatic) {
                    flatResumeBoard[boardPos] = -1;
                    bestBoard[boardPos] = -1;
                    consecutiveExtinctions++;
                }
            }
        }
        logger.info(String.format(">>> Injected Entropy: Randomly removed %d internal pieces to scramble the pathfinder.", piecesToRip));
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
                if (physicalId != -1) {
                    usedPhysical[physicalId] = true;
                }
            } else {
                emptySpots.add(i);
            }
        }

        List<Integer> unusedPhysIds = new ArrayList<>();
        for (int physId = 0; physId < 256; physId++) {
            if (!usedPhysical[physId]) {
                unusedPhysIds.add(physId);
            }
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
                        boolean match = !atNorthEdge || PieceUtils.getNorth(p) == 0;

                        if (atSouthEdge && PieceUtils.getSouth(p) != 0) {
                            match = false;
                        }
                        if (atWestEdge && PieceUtils.getWest(p) != 0) {
                            match = false;
                        }
                        if (atEastEdge && PieceUtils.getEast(p) != 0) {
                            match = false;
                        }

                        if (!atNorthEdge && PieceUtils.getNorth(p) == 0) {
                            match = false;
                        }
                        if (!atSouthEdge && PieceUtils.getSouth(p) == 0) {
                            match = false;
                        }
                        if (!atWestEdge && PieceUtils.getWest(p) == 0) {
                            match = false;
                        }
                        if (!atEastEdge && PieceUtils.getEast(p) == 0) {
                            match = false;
                        }

                        if (match && row > 0 && simulatedBoard[spot - 16] != -1) {
                            if (PieceUtils.getNorth(p) != PieceUtils.getSouth(simulatedBoard[spot - 16])) {
                                match = false;
                            }
                        }
                        if (match && col > 0 && simulatedBoard[spot - 1] != -1) {
                            if (PieceUtils.getWest(p) != PieceUtils.getEast(simulatedBoard[spot - 1])) {
                                match = false;
                            }
                        }

                        if (match) {
                            bestBrikIndex = i;
                            bestOrientedPiece = p;
                            break;
                        }
                    }
                }
                if (bestBrikIndex != -1) {
                    break;
                }
            }

            if (bestBrikIndex == -1) {
                for (int i = 0; i < unusedPhysIds.size(); i++) {
                    int physId = unusedPhysIds.get(i);
                    for (int oi = 0; oi < 1024; oi++) {
                        if (inventory.physicalMapping[oi] == physId) {
                            int p = inventory.allOrientations[oi];
                            boolean match = !atNorthEdge || PieceUtils.getNorth(p) == 0;

                            if (atSouthEdge && PieceUtils.getSouth(p) != 0) {
                                match = false;
                            }
                            if (atWestEdge && PieceUtils.getWest(p) != 0) {
                                match = false;
                            }
                            if (atEastEdge && PieceUtils.getEast(p) != 0) {
                                match = false;
                            }

                            if (!atNorthEdge && PieceUtils.getNorth(p) == 0) {
                                match = false;
                            }
                            if (!atSouthEdge && PieceUtils.getSouth(p) == 0) {
                                match = false;
                            }
                            if (!atWestEdge && PieceUtils.getWest(p) == 0) {
                                match = false;
                            }
                            if (!atEastEdge && PieceUtils.getEast(p) == 0) {
                                match = false;
                            }

                            if (match) {
                                bestBrikIndex = i;
                                bestOrientedPiece = p;
                                break;
                            }
                        }
                    }
                    if (bestBrikIndex != -1) {
                        break;
                    }
                }
            }

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
                if (currentPiece == -1) {
                    continue;
                }
                if (col < 15) {
                    int eastNeighbor = simulatedBoard[idx + 1];
                    if (eastNeighbor != -1 && PieceUtils.getEast(currentPiece) != PieceUtils.getWest(eastNeighbor)) {
                        totalConflicts++;
                    }
                }
                if (row < 15) {
                    int southNeighbor = simulatedBoard[idx + 16];
                    if (southNeighbor != -1 && PieceUtils.getSouth(currentPiece) != PieceUtils.getNorth(southNeighbor)) {
                        totalConflicts++;
                    }
                }
            }
        }

        logger.info(String.format(">>> [FULL BOARD SCAN] Simulated an edge-aware fully laid board. Total internal edge conflicts: %d / 480", totalConflicts));
        if (totalConflicts < 30) {
            logger.warn(">>> [!!!] WOW! You are mathematically incredibly close to a full solution!");
        }

        saveFullBoardVariant(simulatedBoard, absoluteHighScore, totalConflicts);
    }

    private void triggerBranchScrap() {
        consecutiveExtinctions++;
        int dropAmount;
        if (this.useGpu) {
            dropAmount = 10 + (Math.min(10, consecutiveExtinctions) * 5);
        } else {
            dropAmount = 2;
        }
        int lockedPieces = Math.max(0, deepestStep - dropAmount);

        int staticCount = this.lockCenter ? (1 + HINT_POSITIONS.length) : 0;

        if (lockedPieces > staticCount) {
            poisonedIndex = buildOrder[lockedPieces];
            poisonedPiece = bestBoard[poisonedIndex];
            if (poisonedPiece == -1) {
                poisonedPiece = flatResumeBoard[poisonedIndex];
            }
            poisonExpiryExtinction = consecutiveExtinctions;

            logger.info(String.format(">>> [TEARDOWN] Deadlock trapped. Dropping %d pieces to depth %d. Poison expires at extinction %d.",
                    dropAmount, lockedPieces, poisonExpiryExtinction + 1));
        } else {
            lockedPieces = staticCount;
            poisonedIndex = -1;
            poisonedPiece = -1;
            poisonExpiryExtinction = -1;
            consecutiveExtinctions = 0;
            localWallDepth = 0;
            logger.info(">>> [TEARDOWN] Deadlock trapped. Reached Base Camp. Extinctions & Local Wall reset to 0.");
        }

        Arrays.fill(flatResumeBoard, -1);
        for (int step = 0; step < lockedPieces; step++) {
            int idx = buildOrder[step];
            boolean isStatic = (lockCenter && idx == 135);
            if (lockCenter) {
                for (int hPos : HINT_POSITIONS) {
                    if (idx == hPos) {
                        isStatic = true;
                        break;
                    }
                }
            }
            if (!isStatic) {
                flatResumeBoard[idx] = bestBoard[idx];
            }
        }
        applyStaticLocks(flatResumeBoard);

        for (int i = lockedPieces; i < 256; i++) {
            int boardPos = buildOrder[i];
            bestBoard[boardPos] = -1;
        }

        deepestStep = lockedPieces;
        seedPool.clear();
        consecutiveGpuStagnation = 0;
        phaseCounter = 0;

        topBoards.clear();
        updateDisplay(deepestStep, buildDisplayBoard(flatResumeBoard));

        syncInventoryWithResumeBoard();
    }

    private void syncInventoryWithResumeBoard() {
        java.util.Arrays.fill(usedPhysicalPieces, false);
        for (int i = 0; i < 256; i++) {
            int p = flatResumeBoard[i];
            if (p != -1) {
                for (int oi = 0; oi < 1024; oi++) {
                    if (inventory.allOrientations[oi] == p) {
                        usedPhysicalPieces[inventory.physicalMapping[oi]] = true;
                        break;
                    }
                }
            }
        }
    }

    private void handleVictory(int[] winningBoard, PieceInventory inventory) {
        logger.info(">>> ETERNITY II SOLVED BY GPU PIPELINE!!! <<<");
        updateDisplay(256, buildDisplayBoard(winningBoard));
        RecordManager.saveRecord(buildDisplayBoard(winningBoard), 256, saveProfile);
        consecutiveExtinctions = 0;
        saveAndUploadBucasLink(winningBoard, 256);

        // --- UPLOAD TO DRIVE KODE FJERNET FOR AT UNDGÅ CRASH ---

        System.exit(0);
    }

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

        cumulativeTrials += (cpuTrials + gpuTrials);

        if (cpuTrials == 0 && gpuTrials == 0) {
            lastThroughputReportTime = now;
            return;
        }

        double cpuTps = cpuTrials / (elapsed / 1000.0);
        double gpuTps = gpuTrials / (elapsed / 1000.0);

        int hash = Arrays.hashCode(bestBoard);

        logger.info(String.format("[SPEED] CPU: %,.0f/s  |  GPU: %,.0f/s  |  Pieces: %d  |  Hash: %08X",
                cpuTps, gpuTps, deepestStep, hash));
        System.out.printf("[SPEED] CPU: %,.0f/s  |  GPU: %,.0f/s  |  Pieces: %d  |  Hash: %08X%n",
                cpuTps, gpuTps, deepestStep, hash);

        int totalWins = eliteWins.get() + diverseWins.get() + restartWins.get();
        if (totalWins > 0) {
            logger.info(String.format("Distribution -> Elite: %d%% | Diverse: %d%% | Restarts: %d%% | Depth: %d",
                    (eliteWins.get() * 100) / totalWins,
                    (diverseWins.get() * 100) / totalWins,
                    (restartWins.get() * 100) / totalWins,
                    highestP2DepthThisCycle));
            System.out.printf("Distribution -> Elite: %d%% | Diverse: %d%% | Restarts: %d%% | Depth: %d%n",
                    (eliteWins.get() * 100) / totalWins,
                    (diverseWins.get() * 100) / totalWins,
                    (restartWins.get() * 100) / totalWins,
                    highestP2DepthThisCycle);
        }

        highestP2DepthThisCycle = deepestStep;
        lastThroughputReportTime = now;
    }

    void retreat(int targetStep, String logMessage) {
        deepestStep = Math.max(0, targetStep);
        localWallDepth = deepestStep;
        consecutiveExtinctions = 0;
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
            return;
        }

        lastVisualUpdate = now;
        Eternity.updateDisplay(depth, this.absoluteHighScore, displayBoard);
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
        String baseFilename = "Errors" + conflicts + "_Base" + baseScore + "_" + timeId;

        try (java.io.PrintWriter writer = new java.io.PrintWriter(new java.io.File(folder, "Raw_Board_Output_" + baseScore + ".txt"))) {
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
            logger.info(String.format(">>> Saved raw board text for BoardImporter: Raw_Board_Output_%d.txt", baseScore));
        } catch (Exception e) {
            logger.error(String.format(">>> Error saving Raw Board Text: %s", e.getMessage()));
        }

        try (java.io.PrintWriter writer = new java.io.PrintWriter(new java.io.File(folder, "pieces.csv"))) {
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
            logger.info(">>> Saved official verification file: pieces.csv");
        } catch (Exception e) {
            logger.error(String.format(">>> Error saving pieces.csv: %s", e.getMessage()));
        }

        try (java.io.PrintWriter writer = new java.io.PrintWriter(new java.io.File(folder,
                baseFilename + "_link.txt"))) {
            writer.println("Base Score: " + baseScore);
            writer.println("Edge Conflicts: " + conflicts);
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

//    public void setVariantSaveThreshold(int newThreshold) {
//        this.variantSaveThreshold.set(newThreshold);
//        logger.info(">>> [CONFIG] Variant Save Threshold updated to: " + newThreshold);
//    }

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
        private final Random rnd = new Random();
        private final int activeBatch;
        private long localTrialCount = 0;
        private final boolean isVanguard;

        public CpuSearchWorker(int activeBatch, boolean isVanguard) {
            this.activeBatch = activeBatch;
            this.isVanguard = isVanguard;
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
        }

        @Override
        public Boolean call() {
            boolean result = solve(0);
            globalCpuTrialCount.addAndGet(localTrialCount);
            return result;
        }

        private boolean solve(int step) {
            // >>> TIMEOUT CHECK <<< Lader tråden afbryde sig selv smukt
            if (manualOverrideRequested || Thread.currentThread().isInterrupted() || (useGpu && seedPool.size() >= activeBatch)) {
                return false;
            }

            if (step == 256) {
                synchronized (displayLock) {
                    if (step > absoluteHighScore) {
                        absoluteHighScore = step;
                        System.arraycopy(localBoard, 0, globalBestBoard, 0, 256);
                        RecordManager.saveRecord(buildDisplayBoard(globalBestBoard), absoluteHighScore, saveProfile);
                        logger.info(">>> [CPU MODE] VICTORY! ETERNITY II SOLVED! <<<");
                        logger.info(String.format(">>> Total Trials to reach this milestone: %,d", cumulativeTrials));
                    }
                }
                return true;
            }

            localTrialCount++;
            if (localTrialCount >= 1000) {
                globalCpuTrialCount.addAndGet(localTrialCount);
                localTrialCount = 0;
                lastActivityTimestamp.set(System.currentTimeMillis()); // Watchdog Heartbeat
            }

            if (step > deepestStep) {
                synchronized (displayLock) {
                    if (step > deepestStep) {
                        deepestStep = step;
                        System.arraycopy(localBoard, 0, bestBoard, 0, 256);
                        updateDisplay(deepestStep, buildDisplayBoard(localBoard));
                        if (!useGpu && deepestStep > absoluteHighScore) {
                            absoluteHighScore = deepestStep;
                            System.arraycopy(localBoard, 0, globalBestBoard, 0, 256);
                            logger.info(String.format(">>> Total Trials to reach this milestone: %,d", cumulativeTrials));
                            RecordManager.saveRecord(buildDisplayBoard(globalBestBoard), absoluteHighScore,
                                    saveProfile);
                            analyzeFullBoardPotential(globalBestBoard);
                            consecutiveExtinctions = 0;
                        }
                    }
                }
            }

            if (useGpu && step == currentSeedDepth) {
                int[] seed = new int[256];
                System.arraycopy(localBoard, 0, seed, 0, 256);
                seedPool.add(seed);
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

            if (localResumeBoard[boardIdx] != -1 && !isVanguard) {
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

                if (eastReq != CompatibilityIndex.WILDCARD && PieceUtils.getEast(p) != eastReq) {
                    continue;
                }
                if (southReq != CompatibilityIndex.WILDCARD && PieceUtils.getSouth(p) != southReq) {
                    continue;
                }

                if (isVanguard && boardIdx == poisonedIndex && p == poisonedPiece && poisonExpiryExtinction >= consecutiveExtinctions) {
                    continue;
                }

                if (passesLookahead(p, step, row, col, boardIdx)) {
                    localBoard[boardIdx] = p;
                    localUsed[physicalIdx] = true;
                    int ghost = localResumeBoard[boardIdx];
                    localResumeBoard[boardIdx] = -1;

                    if (solve(step + 1)) {
                        return true;
                    }

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
package dk.puzzle.core;

import dk.puzzle.ai.SeedSelector;
import dk.puzzle.ai.SurgeonHeuristics;
import dk.puzzle.gpu.GpuEngine;
import dk.puzzle.io.BucasExporter;
import dk.puzzle.io.CheckpointManager;
import dk.puzzle.io.RecordManager;
import dk.puzzle.model.CompatibilityIndex;
import dk.puzzle.model.PieceInventory;
import dk.puzzle.util.PieceUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static dk.puzzle.io.RecordManager.uploadToDrive;

/**
 * The central orchestrator for the Eternity II puzzle solver, managing a multiphase
 * hybrid search strategy involving CPU-based seed generation and GPU-accelerated
 * deep search and repair.
 *
 * <p>The solver operates in three distinct phases:
 * <ul>
 *     <li><b>Phase 1:</b> Parallel CPU workers generate "seeds" (partial boards)
 *     using a randomized depth-first search (DFS) with lookahead.</li>
 *     <li><b>Phase 2:</b> The GPU consumes these seeds and performs a massively
 *     parallel Deep DFS to reach higher placement depths.</li>
 *     <li><b>Phase 3:</b> Upon reaching a high-depth threshold, the solver enters
 *     "Surgeon" mode (Large Neighborhood Search), strategically removing pieces
 *     to resolve conflicts and push toward a complete 256-piece solution.</li>
 * </ul></p>
 *
 * <p>This class also handles state persistence via checkpoints, real-time metrics
 * reporting, and dynamic parameter updates from the user interface.</p>
 */
public class EternitySolver implements Runnable {

    /**
     * The target depth for Phase 1 CPU seed generation.
     */
    public static final int SEED_DEPTH = 80;
    volatile int currentSeedDepth = SEED_DEPTH;
    /**
     * The piece count threshold that triggers Phase 3 Large Neighborhood Search (LNS).
     */
    public static final int LNS_THRESHOLD = 200;

    ConcurrentLinkedQueue<int[]> seedPool = new ConcurrentLinkedQueue<>();

    private final PieceInventory inventory;
    private final CompatibilityIndex compatIndex;
    private final SurgeonHeuristics surgeon;
    private GpuEngine gpuEngine;
    private final boolean useGpu;

    // Threading
    private final ExecutorService executor;
    private final int numCores;
    private final int targetBatchSize = 50000;
    private volatile int userBatchSizeOverride = -1;
    AtomicInteger currentBatchSize = new AtomicInteger(0);

    // Board State
    private final int[] flatBoard = new int[256];
    int[] bestBoard = new int[256];
    private final int[] globalBestBoard = new int[256];
    private final TopBoardRegistry topBoards = new TopBoardRegistry();
    final int[] flatResumeBoard = new int[256];
    final boolean[] usedPhysicalPieces = new boolean[256];
    final int[] tabuTenure = new int[256];
    final int[] buildOrder = new int[256];
    private volatile int lastReportedDepth = 0;
    volatile int absoluteHighScore = 0;
    volatile int deepestStep = 0;
    int currentRepairIteration = 0;
    int consecutiveExtinctions = 0;
    private int consecutiveGpuStagnation = 0;

    private final boolean lockCenter;
    final int targetPiece;
    private int centerPhysicalIdx = -1;
    private final String saveProfile;
    private final BuildStrategy currentStrategy;
    private final Object displayLock = new Object();

    // Metrics
    private final AtomicLong globalCpuTrialCount = new AtomicLong(0);
    private final AtomicLong globalGpuTrialCount = new AtomicLong(0);
    private long lastThroughputReportTime = System.currentTimeMillis();
    private long totalRepairVariationsTested = 0;
    private long repairLoopsCounter = 0;
    private volatile boolean isGpuBusy = false;

    private volatile int stagnationLimitMinutes = 20;
    volatile boolean manualOverrideRequested = false;
    private volatile int manualBaseCampTarget = 0;
    private volatile double extinctionThreshold = 0.98;
    private volatile int poisonedIndex = -1;
    private volatile int poisonedPiece = -1;
    public static volatile int userCpuHandoffDepth = 8;
    public static volatile int userSurgeonHoles = 20;

    private final Set<Integer> uniqueMaxScoreHashes = ConcurrentHashMap.newKeySet();
    private ScheduledExecutorService repairReporterScheduler;
    // Using getFormatterLogger allows for printf-style formatting (%,d, %s, etc.)
    private static final Logger logger = LogManager.getFormatterLogger(EternitySolver.class);

    /**
     * Constructs a new {@code EternitySolver} and initializes the search environment.
     *
     * <p>Initializes the build order based on the selected strategy, configures
     * the compatibility index, and attempts to resume search from the latest
     * persistent checkpoint matching the strategy profile.</p>
     *
     * @param inventory       The inventory containing all physical pieces and their orientations.
     * @param trueCenterPiece The bit-packed representation of the mandatory centerpiece.
     * @param useGpu          Whether to enable OpenCL/GPU acceleration for Phase 2 and 3.
     * @param strategy        The board-filling pattern (e.g., SPIRAL or TYPEWRITER).
     * @param lockCenter      If true, ensures the centerpiece remains fixed at index 135.
     */
    public EternitySolver(PieceInventory inventory, int trueCenterPiece, boolean useGpu, BuildStrategy strategy, boolean lockCenter) {
        Arrays.fill(flatBoard, -1);
        Arrays.fill(bestBoard, -1);
        Arrays.fill(globalBestBoard, -1);
        Arrays.fill(flatResumeBoard, -1);

        this.inventory = inventory;
        this.targetPiece = trueCenterPiece;
        this.currentStrategy = strategy;
        this.lockCenter = lockCenter;

        this.saveProfile = strategy.name() + (lockCenter ? "_LOCKED" : "_UNLOCKED");
        this.useGpu = useGpu;

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

        // Load Checkpoint Robustly
        int[][] loaded = CheckpointManager.loadSmartCheckpoint(saveProfile);
        if (loaded != null) {
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
                System.arraycopy(bestBoard, 0, flatBoard, 0, 256);
                System.arraycopy(bestBoard, 0, flatResumeBoard, 0, 256);
                updateDisplay(absoluteHighScore, buildDisplayBoard(globalBestBoard));
                logger.info(">>> SUCCESS: Loaded checkpoint fully! Engine locked at " + absoluteHighScore + " pieces.");
            } else {
                logger.info(">>> [WARNING] Smart Load found the file, but it was EMPTY or CORRUPTED inside!");
            }
        } else {
            logger.info(">>> [WARNING] Smart Load failed and returned null. Starting from scratch.");
        }
    }

    // GUI BRIDGES

    /**
     * Sets the duration the solver is allowed to stagnate at a high score before
     * triggering automated recovery actions.
     *
     * @param minutes Stagnation limit in minutes.
     */
    public void setStagnationLimit(int minutes) {
        this.stagnationLimitMinutes = minutes;
    }

    /**
     * Manually overrides the target seed pool size.
     *
     * @param size Number of seeds to generate in Phase 1, or -1 for automatic scaling.
     */
    public void setBatchSizeOverride(int size) {
        this.userBatchSizeOverride = size;
    }

    /**
     * Configures the population "extinction" threshold used during search phases.
     *
     * @param threshold A ratio between 0.0 and 1.0.
     */
    public void setExtinctionThreshold(double threshold) {
        this.extinctionThreshold = threshold;
    }

    /**
     * Adjusts the percentage of holes targeted at identified conflict zones
     * during Phase 3 Surgeon operations.
     *
     * @param percentage Ratio of targeted vs. random holes (0.0 to 1.0).
     */
    public void setTargetedHolesPercentage(double percentage) {
        if (this.surgeon != null) {
            this.surgeon.setTargetedHolesPercentage(percentage);
        }
    }

    /**
     * Forces the solver to discard current progress and "retreat" to a
     * specific piece count (base camp).
     *
     * @param targetBaseCamp The piece count (depth) to roll back to.
     */
    public void triggerManualOverride(int targetBaseCamp) {
        this.manualBaseCampTarget = targetBaseCamp;
        this.manualOverrideRequested = true;
    }

    /**
     * The main execution loop of the solver.
     *
     * <p>Cycles through search phases, processes manual overrides, manages
     * periodic checkpoints, and monitors the health of the GPU pipelines.</p>
     */
    @Override
    public void run() {
        if (this.useGpu) {
            this.gpuEngine = new GpuEngine(inventory, lockCenter);
        }

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            logger.info(">>> Shutdown hook: Saving final checkpoint...");
            synchronized (displayLock) {
                if (repairReporterScheduler != null) {
                    repairReporterScheduler.shutdownNow();
                }
                CheckpointManager.saveRecordCheckpoint(buildDisplayBoard(globalBestBoard), absoluteHighScore, saveProfile);
            }
        }));

        Thread reporterThread = new Thread(() -> {
            while (true) {
                try {
                    Thread.sleep(5000);
                } catch (InterruptedException ignored) {
                }
                reportSpeed();
            }
        });
        reporterThread.setDaemon(true);
        reporterThread.start();

        logger.info("Starting 3-Phase Pipeline Orchestrator...");
        long lastPeriodicSave = System.currentTimeMillis();

        while (true) {
            try {
                if (manualOverrideRequested) {
                    manualOverrideRequested = false;
                    retreat(manualBaseCampTarget, ">>> User Override...");
                    try {
                        Thread.sleep(1000);
                    } catch (InterruptedException ignored) {
                    }
                }

                if (gpuEngine != null && deepestStep >= LNS_THRESHOLD) {
                    runPhase3_GpuSurgeon();
                } else if (gpuEngine != null && seedPool.size() >= targetBatchSize) {
                    runPhase2_GpuDeepDfs();
                } else {
                    runPhase1_CpuSeedGen();
                }

                if (System.currentTimeMillis() - lastPeriodicSave > 300_000) {
                    synchronized (displayLock) {
                        CheckpointManager.saveRecordCheckpoint(buildDisplayBoard(globalBestBoard), absoluteHighScore, saveProfile);
                    }
                    lastPeriodicSave = System.currentTimeMillis();
                }
            } catch (Exception e) {
                logger.error(">>> [FATAL ERROR] PIPELINE CRASHED: ",e);
                if (repairReporterScheduler != null) {
                    repairReporterScheduler.shutdownNow();
                }
                try {
                    Thread.sleep(5000);
                } catch (InterruptedException ignored) {
                }
            }
        }
    }

    void runPhase1_CpuSeedGen() {
        Arrays.fill(usedPhysicalPieces, false);

        // Count how many pieces are currently locked in our Base Camp
        int lockedPieces = 0;
        for (int p : flatResumeBoard) {
            if (p != -1 && p != -2) {
                lockedPieces++;
            }
        }

        // Dynamic Handoff: The CPU should always generate seeds 8 pieces deeper than the Base Camp
        if (lockedPieces > 0) {
            currentSeedDepth = Math.max(SEED_DEPTH, lockedPieces + userCpuHandoffDepth);
        } else {
            currentSeedDepth = SEED_DEPTH;
        }

        currentBatchSize.set(seedPool.size());

        try {
            int activeBatch = (userBatchSizeOverride > 0) ? userBatchSizeOverride : targetBatchSize;
            List<CpuSearchWorker> workers = new ArrayList<>();
            for (int i = 0; i < numCores; i++) {
                workers.add(new CpuSearchWorker(activeBatch));
            }

            executor.invokeAll(workers);

            // DEADLOCK DETECTOR
            if (seedPool.isEmpty()) {
                logger.warn(">>> [PHASE 1 DEADLOCK] CPU proved that NO alternative seeds exist for this Base Camp!");
                consecutiveExtinctions++;
                triggerBranchScrap();
            }else {
                if (poisonedIndex != -1) {
                    logger.info(">>> [TABU] CPU successfully bypassed the dead end. Global Tabu lifted.");
                    poisonedIndex = -1;
                    poisonedPiece = -1;
                }
            }

        } catch (InterruptedException e) {
            return;
        }
    }

    void runPhase2_GpuDeepDfs() {
        List<int[]> seeds = new ArrayList<>();
        for (int i = 0; i < targetBatchSize; i++) {
            int[] s = seedPool.poll();
            if (s != null) {
                seeds.add(s);
            }
        }
        if (seeds.isEmpty()) {
            return;
        }

        int scoreBefore = deepestStep;
        int[] bestBoardOut = new int[256];

        isGpuBusy = true;

        long start = System.currentTimeMillis();
        GpuEngine.GpuResult result = gpuEngine.runDeepDfs(
                seeds, currentSeedDepth, deepestStep, bestBoardOut, buildOrder);
        // CLIMBING TRACKER: Print to the log ONLY when the GPU reaches a new depth for this branch
        if (result.newHighScore() > lastReportedDepth) {
            logger.info(">>> [CLIMBING] Current pieces placed: %d / 256", result.newHighScore());
            lastReportedDepth = result.newHighScore();
        }
        globalGpuTrialCount.addAndGet(result.stepsTaken());
        isGpuBusy = false;

        long elapsed = System.currentTimeMillis() - start;
        logger.info(">>> GPU Phase 2 complete. Steps taken per second: %,d",
                Math.round((double) result.stepsTaken() * 1000) / Math.max(1, elapsed));

        if (result.solved()) {
            handleVictory(bestBoardOut);
        }

        if (result.newHighScore() > scoreBefore) {
            if (!verifyBoardStrict(bestBoardOut)) {
                logger.error(">>> [FATAL GPU BUG] The GPU returned a board with an illegal edge conflict! Rejecting this fake record.");
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

                logger.info(">>> PHASE 3 (SURGEON) BROKE RECORD! NEW HIGH SCORE: " + absoluteHighScore + " <<<");
                analyzeFullBoardPotential(globalBestBoard);

            } else if (deepestStep == absoluteHighScore) {
                int boardHash = Arrays.hashCode(bestBoardOut);
                if (uniqueMaxScoreHashes.add(boardHash)) {
                    analyzeFullBoardPotential(bestBoardOut);
                    logger.info(">>> [PROGRESS] Surgeon got a new unique variant of %d-pieced board! (Uniquely found %d times so far",
                            absoluteHighScore, uniqueMaxScoreHashes.size());
                }
            }
        } else {
            consecutiveGpuStagnation++;
//            System.out.printf(">>> PHASE 2 (GPU) No progress. Stagnation counter: %d/10%n", consecutiveGpuStagnation);
        }

        if (consecutiveGpuStagnation < 10) {
            List<int[]> nextSeeds = SeedSelector.selectBest(
                    seeds,
                    result.threadDepths(),
                    targetBatchSize,
                    new Random()
            );

            logger.info(">>> Phase 2 Evolution: Bred %d new seeds (Elite + Mutated) for next round!", nextSeeds.size());

            seedPool.addAll(nextSeeds);
            currentBatchSize.set(seedPool.size());
        } else {
            logger.warn(">>> [!!!] Phase 2 GPU stagnated! Trapped below LNS threshold. Activating Teardown...");
            triggerBranchScrap();
        }
    }

    /**
     * Executes Phase 3 of the solver pipeline: Large Neighborhood Search (LNS)
     * using the GPU in repair mode.
     *
     * <p>Instead of always operating on a single best board, this method cycles
     * through the top-20 registry in round-robin order, giving the GPU varied
     * structural starting points each iteration. This reduces the risk of getting
     * permanently stuck in a single local optimum.</p>
     *
     * <p>For each iteration, a source board is selected from the registry, a batch
     * of "swiss cheese" variants is generated by punching {@code holesToPunch} holes,
     * and the GPU attempts to fill them. If a new high score is found, it is added
     * back into the top-20 registry. If no progress is made for 10 consecutive
     * iterations, {@link #triggerBranchScrap()} is called to refresh the search.</p>
     */
    void runPhase3_GpuSurgeon() {
        int numClones = 50000;

        int holesToPunch = userSurgeonHoles;

        currentRepairIteration++;
        repairLoopsCounter++;
        totalRepairVariationsTested += numClones;

        int[] sourceBoard = topBoards.nextForRepair();
        if (sourceBoard == null) {
            sourceBoard = bestBoard; // fallback if registry is empty
        }

        List<int[]> swissCheeseBoards = surgeon.punchHoles(
                sourceBoard, numClones, holesToPunch,
                tabuTenure, currentRepairIteration, deepestStep, buildOrder);

        int scoreBefore = deepestStep;
        int[] bestBoardOut = new int[256];
        GpuEngine.GpuResult result = gpuEngine.runRepairMode(swissCheeseBoards, deepestStep, bestBoardOut);
        // CLIMBING TRACKER: Print to the log ONLY when the GPU reaches a new depth for this branch
        if (result.newHighScore() > lastReportedDepth) {
            logger.info(">>> [CLIMBING] Current pieces placed: %d / 256", result.newHighScore());
            lastReportedDepth = result.newHighScore();
        }
        globalGpuTrialCount.addAndGet(result.stepsTaken());

        if (result.solved()) {
            handleVictory(bestBoardOut);
        }

        if (result.newHighScore() > scoreBefore) {
            if (!verifyBoardStrict(bestBoardOut)) {
                logger.error(">>> [FATAL GPU BUG] The GPU returned a board with an illegal edge conflict! Rejecting this fake record.");
                return;
            }
            deepestStep = result.newHighScore();
            updateTabuList(bestBoardOut);
            System.arraycopy(bestBoardOut, 0, bestBoard, 0, 256);

            // Register the improved board so future iterations can build on it
            topBoards.offer(bestBoardOut, deepestStep);
            updateDisplay(deepestStep, buildDisplayBoard(bestBoard));

            if (deepestStep > absoluteHighScore) {
                absoluteHighScore = deepestStep;
                System.arraycopy(bestBoardOut, 0, globalBestBoard, 0, 256);
                RecordManager.saveRecord(buildDisplayBoard(globalBestBoard), absoluteHighScore, saveProfile);
                consecutiveExtinctions = 0;

                saveAndUploadBucasLink(globalBestBoard, absoluteHighScore);

                logger.info(">>> PHASE 3 (SURGEON) BROKE RECORD! NEW HIGH SCORE: " + absoluteHighScore + " <<<");
                analyzeFullBoardPotential(globalBestBoard);
            } else if (deepestStep == absoluteHighScore) {
                int boardHash = Arrays.hashCode(bestBoardOut);
                if (uniqueMaxScoreHashes.add(boardHash)) {
                    analyzeFullBoardPotential(bestBoardOut);
                    logger.info(">>> [PROGRESS] Surgeon got a new unique variant of %d-pieced board! (Uniquely found %d times so far)",
                            absoluteHighScore, uniqueMaxScoreHashes.size());
                }
            }
            consecutiveExtinctions = 0;
        } else {
            consecutiveExtinctions++;
            if (consecutiveExtinctions >= 10) {
                triggerBranchScrap();
            }
        }
    }

    void updateTabuList(int[] newBoard) {
        // RELAXED TENURE: Shorter memory allows the Surgeon to re-optimize areas much faster.
        // Base of 5 rounds, plus 1 extra round for every 25 pieces placed.
        // E.g., at 200 pieces: 5 + (200 / 25) = 13 rounds (much more forgiving than 25!)
        int tenureLength = 5 + (absoluteHighScore / 25);

        for (int i = 0; i < 256; i++) {
            if (newBoard[i] != bestBoard[i] && newBoard[i] != -1) {
                tabuTenure[i] = currentRepairIteration + tenureLength;
            }
        }
    }

    void clearTabu() {
        Arrays.fill(tabuTenure, 0);
        logger.info(">>> [TABU] All tenures cleared for fresh Phase 3 cycle.");
    }

    /**
     * Handles a "Dead End" in Phase 3 (Surgeon).
     * Employs an Adaptive Large Neighborhood Search (ALNS) strategy combined with
     * Simulated Annealing principles. Stagnation progressively increases the teardown
     * depth. Extreme stagnation triggers a massive structural rollback to escape
     * deep local optima, especially crucial for the Typewriter build strategy.
     */
    void triggerBranchScrap() {
        // 1. Increment the frustration/stagnation counter
        consecutiveExtinctions++;

        logger.info(">>> [!!!] DEAD END AT PHASE 3 (Stagnation level: " + consecutiveExtinctions + ") [!!!]");

        // 2. Base frustration: calculate dynamic retreat bounds (lapping the bottom rows)
        int minRetreat = 10 + (consecutiveExtinctions * 5);
        int maxRetreat = minRetreat + 15;

        // 3. Critical Stagnation: if we are stuck for a long time (e.g., trapped in upper rows)
        if (consecutiveExtinctions > 25) {
            maxRetreat = absoluteHighScore;
        }

        Random rand = new Random();
        int retreatAmount = minRetreat + rand.nextInt(maxRetreat - minRetreat + 1);

        int lockedPieces = Math.max(0, absoluteHighScore - retreatAmount);

        if (consecutiveExtinctions > 25) {
            lockedPieces = 0;
            retreatAmount = absoluteHighScore;
        }

        // 4. Execute the teardown
        // Reset the climbing tracker so it starts reporting from the new Base Camp
        lastReportedDepth = lockedPieces;

        if (consecutiveExtinctions > 25) {
//            logger.warn("================================================================");
//            logger.warn(">>> [CRITICAL TEARDOWN] MASSIVE STAGNATION!");
//            logger.warn(">>> Removed: %d pieces", retreatAmount);
//            logger.warn(">>> New Base Camp: 0 pieces remaining");
//            logger.warn("================================================================");
            consecutiveExtinctions = 0;
//        } else {
//            logger.info("----------------------------------------------------------------");
//            logger.info(">>> [TEARDOWN] Branch scrapped due to stagnation.");
//            logger.info(">>> Removed: %d pieces", retreatAmount);
//            logger.info(">>> New Base Camp: %d pieces remaining", lockedPieces);
//            logger.info("----------------------------------------------------------------");
        }

        if (lockedPieces == 0) {
            poisonedIndex = buildOrder[0];
            poisonedPiece = globalBestBoard[poisonedIndex];
            logger.info(">>> [GLOBAL TABU] Board reset! Old start piece %d is strictly banned at index %d.", poisonedPiece, poisonedIndex);
        } else if (absoluteHighScore > lockedPieces + 5) {
            poisonedIndex = buildOrder[lockedPieces + 2];
            poisonedPiece = globalBestBoard[poisonedIndex];
            logger.info(">>> [GLOBAL TABU] Poisoned Square Active! Piece %d is banned at index %d.", poisonedPiece, poisonedIndex);
        } else {
            poisonedIndex = -1;
            poisonedPiece = -1;
        }

        // Clear the board and lock the Base Camp securely...
        Arrays.fill(flatResumeBoard, -1);
        for (int step = 0; step < lockedPieces; step++) {
            int idx = buildOrder[step];
            if (lockCenter && idx == 135) {
                continue;
            }
            flatResumeBoard[idx] = globalBestBoard[idx];
        }

        // Drop deepestStep below the LNS threshold to force the main loop back into Phase 1
        deepestStep = Math.max(0, lockedPieces);

        // Clear the seed pool so the GPU doesn't process outdated, dead-end paths
        seedPool.clear();
        currentBatchSize.set(0);
        consecutiveGpuStagnation = 0;
        clearTabu();
    }

    /**
     * Simulates filling the rest of the board with the remaining unused pieces.
     * Uses an "Intelligent Edge-Aware Fill" algorithm that ensures corner pieces
     * and border pieces are placed at the edges and rotated so their gray sides (0) face outwards.
     * * @param recordBoard The current best board
     */
    private void analyzeFullBoardPotential(int[] recordBoard) {
        int[] simulatedBoard = Arrays.copyOf(recordBoard, 256);
        List<Integer> emptySpots = new ArrayList<>();
        boolean[] usedPhysical = new boolean[256];

        // 1. Identify empty spots and record which physical pieces are already used
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

        // 2. Gather all remaining unused physical pieces
        List<Integer> unusedPhysIds = new ArrayList<>();
        for (int physId = 0; physId < 256; physId++) {
            if (!usedPhysical[physId]) {
                unusedPhysIds.add(physId);
            }
        }

        // Shuffle to ensure a different random variant each time
        Collections.shuffle(unusedPhysIds);

        // 3. INTELLIGENT FILL: Place pieces based on board geography
        for (int spot : emptySpots) {
            int row = spot / 16;
            int col = spot % 16;

            // Define exactly which sides MUST be gray (0) based on the spot's position
            boolean reqN = (row == 0);
            boolean reqS = (row == 15);
            boolean reqW = (col == 0);
            boolean reqE = (col == 15);

            int selectedListIndex = -1;
            int selectedOrientedPiece = -1;

            // Search through the unused pieces for one that perfectly matches the gray-edge requirements
            for (int i = 0; i < unusedPhysIds.size(); i++) {
                int physId = unusedPhysIds.get(i);

                // Test all 4 orientations of this specific physical piece
                for (int oi = 0; oi < 1024; oi++) {
                    if (inventory.physicalMapping[oi] == physId) {
                        int p = inventory.allOrientations[oi];

                        boolean match = true;
                        // The XOR logic (!=) ensures that an edge is gray ONLY if it is required to be gray
                        if (reqN != (PieceUtils.getNorth(p) == 0)) match = false;
                        if (reqS != (PieceUtils.getSouth(p) == 0)) match = false;
                        if (reqW != (PieceUtils.getWest(p) == 0)) match = false;
                        if (reqE != (PieceUtils.getEast(p) == 0)) match = false;

                        if (match) {
                            selectedListIndex = i;
                            selectedOrientedPiece = p;
                            break;
                        }
                    }
                }
                if (selectedListIndex != -1) break; // We found a valid piece for this spot!
            }

            // Place the piece and remove it from the inventory
            if (selectedListIndex != -1) {
                simulatedBoard[spot] = selectedOrientedPiece;
                unusedPhysIds.remove(selectedListIndex);
            } else {
                // FALLBACK: In case of inventory mismatch, just take the first available piece
                int fallbackPhysId = unusedPhysIds.remove(0);
                for (int oi = 0; oi < 1024; oi++) {
                    if (inventory.physicalMapping[oi] == fallbackPhysId) {
                        simulatedBoard[spot] = inventory.allOrientations[oi];
                        break;
                    }
                }
            }
        }

        // 4. Scan the board and count the edge conflicts
        int totalConflicts = 0;

        for (int row = 0; row < 16; row++) {
            for (int col = 0; col < 16; col++) {
                int idx = row * 16 + col;
                int currentPiece = simulatedBoard[idx];

                // Check East edge
                if (col < 15) {
                    int eastNeighbor = simulatedBoard[idx + 1];
                    if (PieceUtils.getEast(currentPiece) != PieceUtils.getWest(eastNeighbor)) {
                        totalConflicts++;
                    }
                }

                // Check South edge
                if (row < 15) {
                    int southNeighbor = simulatedBoard[idx + 16];
                    if (PieceUtils.getSouth(currentPiece) != PieceUtils.getNorth(southNeighbor)) {
                        totalConflicts++;
                    }
                }
            }
        }

        // 5. Log and Save
        logger.info(">>> [FULL BOARD SCAN] Simulated an edge-aware fully laid board. Total internal edge conflicts: %d / 480", totalConflicts);

        if (totalConflicts < 30) {
            logger.warn(">>> [!!!] WOW! You are mathematically incredibly close to a full solution!");
        }

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
        if (isGpuBusy) {
//            logger.info(timestamp() + "[STATUS] GPU Phase 2 is processing millions of moves per second...");
            return;
        }

        long now = System.currentTimeMillis();
        long elapsed = now - lastThroughputReportTime;

        if (elapsed < 2000) {
            return;
        }

        long cpuTrials = globalCpuTrialCount.getAndSet(0);
        long gpuTrials = globalGpuTrialCount.getAndSet(0);

        double cpuTps = cpuTrials / (elapsed / 1000.0);
        double gpuTps = gpuTrials / (elapsed / 1000.0);

        if (cpuTps != 0 || gpuTps != 0) {
            logger.info("[SPEED] CPU Phase 1: %,.0f/s  |  GPU Phase 2/3: %,.0f/s", cpuTps, gpuTps);
            ;
        }

        lastThroughputReportTime = now;
    }

    void retreat(int targetStep, String logMessage) {
        deepestStep = Math.max(0, targetStep);
        for (int s = deepestStep; s < 256; s++) {
            bestBoard[buildOrder[s]] = -1;
        }
        if (lockCenter) {
            bestBoard[135] = targetPiece;
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

    void updateDisplay(int score, int[][] displayBoard) {
        Eternity.updateDisplay(score, this.absoluteHighScore, displayBoard);
    }

    /**
     * Enumeration of board construction strategies which define the order
     * in which grid positions are filled.
     */
    public enum BuildStrategy {
        /**
         * Standard left-to-right, top-to-bottom placement.
         */
        TYPEWRITER,
        /**
         * Inward-moving spiral, prioritizing the completion of board edges first.
         */
        SPIRAL
    }

    // ==========================================================
    // BUCAS EXPORTER & UPLOADER
    // ==========================================================
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
                writer.write(bucasLink + "\n");
            }

            logger.info(">>> Saved local Bucas-link: " + linkFile.getName());

            uploadToDrive(linkFile, "text/plain", saveProfile);

        } catch (Exception e) {
            System.err.println(">>> [ERROR] Couldn't save/upload Bucas-link: " + e.getMessage());
        }
    }

    private void printPhysicalBoard(int[] board, int score) {
        logger.info(">>> PHYSICAL PIECE-NUMBERS FOR RECORD (" + score + " PIECES):");
        logger.info("------------------------------------------------------------------");
        for (int row = 0; row < 16; row++) {
            StringBuilder sb = new StringBuilder();
            for (int col = 0; col < 16; col++) {
                int p = board[row * 16 + col];
                if (p == -1 || p == -2) {
                    sb.append(String.format("%4s", "---"));
                } else {
                    int physId = -1;
                    for (int i = 0; i < 1024; i++) {
                        if (inventory.allOrientations[i] == p) {
                            physId = inventory.physicalMapping[i];
                            break;
                        }
                    }
                    sb.append(String.format("%4d", physId + 1));
                }
            }
            logger.info(sb.toString());
        }
        logger.info("------------------------------------------------------------------");
    }

    /**
     * Saves a "Full Board" variant into its own dedicated directory.
     * Generates unique filenames with timestamps to preserve ALL generated variants
     * for later analysis.
     * * @param simulatedBoard The full 256-piece board array
     * @param baseScore The number of correct pieces placed before the random fill
     * @param conflicts The total number of edge color mismatches on the full board
     */
    private void saveFullBoardVariant(int[] simulatedBoard, int baseScore, int conflicts) {
        // 1. Create the appropriate directory (e.g., TYPEWRITER_LOCKED_FULL)
        String fullProfileFolder = saveProfile + "_FULL";
        java.io.File folder = new java.io.File(fullProfileFolder);
        if (!folder.exists()) {
            folder.mkdirs();
        }

        // 2. Generate a unique filename: Base[Score]_Errors[Count]_[Timestamp]
        // Example output: Base215_Errors42_194532_842
        String timeId = java.time.LocalTime.now().format(java.time.format.DateTimeFormatter.ofPattern("HHmmss_SSS"));
        String baseFilename = "Errors" +  conflicts + "_Base" + baseScore + "_" + timeId;

        // 3. Save the CSV file mapping (Physical piece numbers 1-256)
        try (java.io.PrintWriter writer = new java.io.PrintWriter(new java.io.File(folder, baseFilename + ".csv"))) {
            for (int row = 0; row < 16; row++) {
                StringBuilder line = new StringBuilder();
                for (int col = 0; col < 16; col++) {
                    int p = simulatedBoard[row * 16 + col];
                    int physId = -1;

                    // Translate from the bit-packed orientation ID to the physical piece ID
                    for (int oi = 0; oi < 1024; oi++) {
                        if (inventory.allOrientations[oi] == p) {
                            physId = inventory.physicalMapping[oi];
                            break;
                        }
                    }

                    // +1 because standard Eternity piece datasets are 1-256 (not 0-255)
                    line.append(physId + 1);
                    if (col < 15) line.append(",");
                }
                writer.println(line.toString());
            }
        } catch (Exception e) {
            logger.error(">>> Error saving Full Board CSV: %d", e.getMessage());
        }

        // 4. Save the Bucas link (allowing rapid visual inspection in the browser)
        try (java.io.PrintWriter writer = new java.io.PrintWriter(new java.io.File(folder, baseFilename + "_link.txt"))) {
            writer.println("Base Score (Correctly placed foundation): " + baseScore);
            writer.println("Edge Conflicts (Mismatches): " + conflicts);
            writer.println("\nClick the Bucas Link below to view this specific variant:");
            writer.println(BucasExporter.exportBoard(simulatedBoard));
        } catch (Exception e) {
            logger.error(">>> Error saving Full Board Bucas Link: %d", e.getMessage());
        }

        // 5. PICTURE / IMAGE GENERATION:
        // If your RecordManager has a method to explicitly draw and save a .png
        // (e.g., RecordManager.saveImage(board, filepath)), you can call it here:
        RecordManager.saveImage(buildDisplayBoard(simulatedBoard), new java.io.File(folder, baseFilename + ".png").getAbsolutePath());
    }

    /**
     * STRICT QUALITY CONTROL:
     * Scans the entire board to ensure every placed piece perfectly matches its neighbors.
     * This acts as an absolute firewall against corrupted GPU "Frankenstein" boards.
     */
    private boolean verifyBoardStrict(int[] board) {
        for (int row = 0; row < 16; row++) {
            for (int col = 0; col < 16; col++) {
                int idx = row * 16 + col;
                int p = board[idx];

                // Skip empty spots
                if (p == -1 || p == -2) continue;

                // Check East edge (if the piece to the right is placed)
                if (col < 15) {
                    int eastP = board[idx + 1];
                    if (eastP != -1 && eastP != -2) {
                        if (PieceUtils.getEast(p) != PieceUtils.getWest(eastP)) return false;
                    }
                }

                // Check South edge (if the piece below is placed)
                if (row < 15) {
                    int southP = board[idx + 16];
                    if (southP != -1 && southP != -2) {
                        if (PieceUtils.getSouth(p) != PieceUtils.getNorth(southP)) return false;
                    }
                }
            }
        }
        return true; // The board is mathematically flawless!
    }

    private void generateSpiralOrder() {
        int top = 0, bottom = 15;
        int left = 0, right = 15;
        int idx = 0;

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

    class CpuSearchWorker implements Callable<Boolean> {
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
                if (centerPhysicalIdx != -1) {
                    localUsed[centerPhysicalIdx] = true;
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
            if (manualOverrideRequested || currentBatchSize.get() >= activeBatch) {
                return false;
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
            if (localBoard[boardIdx] != -1 || (lockCenter && boardIdx == 135)) {
                return solve(step + 1);
            }

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
                if (seedPool.size() >= activeBatch) {
                    return false;
                }

                int orientationIdx = orientIdxs[(i + offset) % candidateCount];
                int p = inventory.allOrientations[orientationIdx];
                int physicalIdx = inventory.physicalMapping[orientationIdx];
                if (boardIdx == poisonedIndex && p == poisonedPiece) {
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
                int reqE = (col == 15) ? 0 : (localBoard[idx + 17] != -1 ? PieceUtils.getWest(localBoard[idx + 17]) : CompatibilityIndex.WILDCARD);
                int reqW = (col == 0) ? 0 : (localBoard[idx + 15] != -1 ? PieceUtils.getEast(localBoard[idx + 15]) : CompatibilityIndex.WILDCARD);
                int reqS = (row == 14) ? 0 : (localBoard[idx + 32] != -1 ? PieceUtils.getNorth(localBoard[idx + 32]) : CompatibilityIndex.WILDCARD);

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
                        if (compatIndex.hasAnyCandidate(nextReqN, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, localUsed)) {
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
                int reqN = (row == 0) ? 0 : (localBoard[idx - 15] != -1 ? PieceUtils.getSouth(localBoard[idx - 15]) : CompatibilityIndex.WILDCARD);
                int reqE = (col == 14) ? 0 : (localBoard[idx + 2] != -1 ? PieceUtils.getWest(localBoard[idx + 2]) : CompatibilityIndex.WILDCARD);
                int reqS = (row == 15) ? 0 : (localBoard[idx + 17] != -1 ? PieceUtils.getNorth(localBoard[idx + 17]) : CompatibilityIndex.WILDCARD);

                if (!compatIndex.hasAnyCandidate(reqN, reqE, reqS, reqW, localUsed)) {
                    return false;
                }
            }
            return true;
        }
    }
}
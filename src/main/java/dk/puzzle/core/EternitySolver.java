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
    /**
     * The piece count threshold that triggers Phase 3 Large Neighborhood Search (LNS).
     */
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
//    private final boolean useEvolutionaryStrategy; // New flag for evolutionary solver
    private final boolean lockCenter;

    // Threading and Concurrency
    private final ExecutorService executor;
    private final int numCores;
    private final int targetBatchSize = 50000;
    volatile int userBatchSizeOverride = -1;
    private final ConcurrentLinkedQueue<int[]> seedPool = new ConcurrentLinkedQueue<>();
    private final AtomicInteger currentBatchSize = new AtomicInteger(0);
    private int fatalGpuBugCount = 0;
    private final Object displayLock = new Object(); // For synchronizing UI updates

    // Board State
    private final int[] flatResumeBoard = new int[256]; // Board state for CPU workers to resume from
    private final boolean[] usedPhysicalPieces = new boolean[256]; // Tracks used physical pieces for CPU workers
    final int[] tabuTenure = new int[256]; // Tabu list for LNS
    final int[] buildOrder = new int[256]; // Order of filling board positions
    final int[] bestBoard = new int[256]; // Best board found in current phase/branch
    final int[] globalBestBoard = new int[256]; // Overall best board found
    final TopBoardRegistry topBoards = new TopBoardRegistry(); // Registry of top boards for LNS

    // Puzzle Specifics
    private final int targetPiece; // The center piece
    private int centerPhysicalIdx = -1; // Physical index of the center piece

    // Solver Profile
    private final String saveProfile;
    private final BuildStrategy currentStrategy;

    // Metrics and Progress Tracking
    private final AtomicLong globalCpuTrialCount = new AtomicLong(0);
    private final AtomicLong globalGpuTrialCount = new AtomicLong(0);
    private final Set<Integer> uniqueMaxScoreHashes = ConcurrentHashMap.newKeySet(); // For tracking unique boards at max score
    private volatile int currentSeedDepth = SEED_DEPTH;
    volatile int absoluteHighScore = 0; // Overall highest score achieved
    volatile int deepestStep = 0; // Deepest step reached in current search branch
    private int currentRepairIteration = 0; // Counter for LNS iterations
    private int consecutiveExtinctions = 0; // Counter for consecutive failures in LNS
    volatile boolean manualOverrideRequested = false;
    volatile int manualBaseCampTarget = 0;
    volatile double extinctionThreshold = 0.98; // For GUI Bridge
    private volatile int lastReportedDepth = 0; // For climbing tracker
    private int consecutiveGpuStagnation = 0; // For GPU Phase 2 deadlock detection
    private long lastThroughputReportTime = System.currentTimeMillis();
    private long totalRepairVariationsTested = 0;
    private long repairLoopsCounter = 0;
    private volatile boolean isGpuBusy = false;
    volatile int stagnationLimitMinutes = 20;
    private volatile int poisonedIndex = -1; // For global Tabu
    private volatile int poisonedPiece = -1; // For global Tabu
    private ScheduledExecutorService repairReporterScheduler; // Unused, can be removed if not planned for future use

    // HINT STRATEGY FIELDS
    private static final int[] HINT_POSITIONS = {221, 45, 210, 34};
    private static final int[] HINT_PHYSICAL_NUMBERS = {249, 181, 255, 208};
    private static final int[] HINT_ROTATIONS = {1, 0, 0, 0};
    private final int[] hintPackedValues = new int[4];
    private final int[] hintPhysicalIndices = new int[4];

    /**
     * Finds the bit-packed representation of a piece given its physical number and target rotation.
     *
     * @param physicalNumber The 1-based physical number of the piece.
     * @param targetRotation The 0-based rotation index (0-3).
     * @return The bit-packed integer representation of the piece in the specified rotation, or -1 if not found.
     */
    private int findPackedPiece(int physicalNumber, int targetRotation) {
        int physIdx = physicalNumber - 1; // pieces.csv is 1-based, inventory is 0-based
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

    /**
     * Constructs a new {@code EternitySolver} and initializes the search environment.
     *
     * <p>Initializes the build order based on the selected strategy, configures
     * the compatibility index, and attempts to resume search from the latest
     * persistent checkpoint matching the strategy profile.</p>
     *
     * @param inventory             The inventory containing all physical pieces and their orientations.
     * @param trueCenterPiece       The bit-packed representation of the mandatory centerpiece.
     * @param useGpu                Whether to enable OpenCL/GPU acceleration for Phase 2 and 3.
     * @param strategy              The board-filling pattern (e.g., SPIRAL or TYPEWRITER).
     * @param lockCenter            If true, ensures the centerpiece remains fixed at index 135.
     */
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

        this.compatIndex = new CompatibilityIndex(inventory.allOrientations, inventory.physicalMapping);
        this.surgeon = new SurgeonHeuristics(lockCenter, 0.70);

        loadCheckpointAndHints();
    }



    /**
     * Loads the checkpoint and pre-locks hint pieces into the board.
     */
    private void loadCheckpointAndHints() {
        SolverState loadedState = CheckpointManager.loadSmartState(saveProfile);

        if (loadedState != null) {
            // 1. Restore the board
            restoreBoardState(loadedState.bestBoard);

            // 2. Restore the memory!
            System.arraycopy(loadedState.tabuTenure, 0, this.tabuTenure, 0, 256);
            this.uniqueMaxScoreHashes.addAll(loadedState.uniqueMaxScoreHashes);

            // 3. Restore the TopBoards registry
            for (int[] historicBoard : loadedState.topBoardsRegistry) {
                this.topBoards.offer(historicBoard, loadedState.score);
            }

            logger.info(">>> SUCCESS: Loaded checkpoint AND restored historic solver memory!");
        } else {
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
                    System.arraycopy(bestBoard, 0, flatResumeBoard, 0, 256);
                    updateDisplay(absoluteHighScore, buildDisplayBoard(globalBestBoard));
                    logger.info(">>> SUCCESS: Loaded checkpoint fully! Engine locked at " + absoluteHighScore + " pieces.");
                } else {
                    logger.warn(">>> [WARNING] Smart Load found the file, but it was EMPTY or CORRUPTED inside!");
                }
            } else {
                logger.info(">>> [WARNING] Smart Load failed and returned null. Starting from scratch.");
            }
        }

        // RESOLVE AND PRE-LOCK HINTS INTO ALL MASTER BOARDS
        for (int h = 0; h < 4; h++) {
            int packed = findPackedPiece(HINT_PHYSICAL_NUMBERS[h], HINT_ROTATIONS[h]);
            hintPackedValues[h] = packed;
            hintPhysicalIndices[h] = HINT_PHYSICAL_NUMBERS[h] - 1;

            if (packed == -1) {
                logger.error(">>> [HINTS] Could not resolve hint piece #%d (physical #%d)!", h, HINT_PHYSICAL_NUMBERS[h]);
            } else {
                // Lock them into all persistent tracking boards unconditionally
                bestBoard[HINT_POSITIONS[h]] = packed;
                globalBestBoard[HINT_POSITIONS[h]] = packed;
                flatResumeBoard[HINT_POSITIONS[h]] = packed;
                logger.info(">>> [HINTS] Hint piece #%d (physical #%d) locked at position %d", h, HINT_PHYSICAL_NUMBERS[h], HINT_POSITIONS[h]);
            }
        }
        // Ensure center is also securely locked on the flatResumeBoard
        if (lockCenter) {
            flatResumeBoard[135] = targetPiece;
        }
    }

    private void restoreBoardState(int[][] loaded) {
        if (loaded == null) return;
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
     * Manually overrides the target seed pool size for CPU seed generation.
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
     * <p>Initializes CUDA (if enabled), sets up shutdown hooks, and orchestrates
     * either the 3-phase pipeline or the evolutionary solver based on configuration.</p>
     */
    @Override
    public void run() {
        initializeGpuEngine();
        setupShutdownHook();
        startReporterThread();

        logger.info("Starting Solver Orchestrator...");

        if (this.absoluteHighScore > 0) {
            logger.info(">>> [BOOT] Checkpoint detected! Setting up Base Camp to resume search...");
            triggerBranchScrap(); // Re-initialize state based on loaded checkpoint
        }

//        if (useEvolutionaryStrategy) {
//            runEvolutionarySolver();
//        } else {
            run3PhasePipeline();
//        }
    }

    /**
     * Initializes the GPU engine if GPU usage is enabled.
     * Handles potential errors during GPU initialization.
     */
    private void initializeGpuEngine() {
        if (this.useGpu) {
            try {
                this.gpuEngine = new GpuEngine(inventory, lockCenter);
                logger.info(">>> [HARDWARE] NVIDIA CUDA GPU detected and initialized successfully.");
            } catch (Throwable t) { // MUST catch Throwable to intercept native JCuda crashes
                this.useGpu = false;
                this.gpuEngine = null;
                logger.warn(">>> [HARDWARE] No compatible NVIDIA GPU found (or JCuda missing).");
                logger.warn(">>> [HARDWARE] Switching to CPU-Only Mode. Fasten your seatbelts!");
            }
        }
    }

    /**
     * Sets up a shutdown hook to save the final checkpoint when the application exits.
     */
    private void setupShutdownHook() {
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            logger.info(">>> Shutdown hook: Saving final checkpoint...");
            synchronized (displayLock) {
                if (repairReporterScheduler != null) { // repairReporterScheduler is unused, can be removed
                    repairReporterScheduler.shutdownNow();
                }
                // Inside EternitySolver
                SolverState memoryToSave = new SolverState(
                        buildDisplayBoard(globalBestBoard),
                        absoluteHighScore,
                        this.tabuTenure, // Sends the active Tabu list
                        this.uniqueMaxScoreHashes, // Sends the hashes of explored high-score variants
                        this.topBoards.getRawRegistry() // You'll need to add a getter to TopBoardRegistry to return its List<int[]>
                );

                CheckpointManager.saveSmartState(memoryToSave, saveProfile);
//                CheckpointManager.saveRecordCheckpoint(buildDisplayBoard(globalBestBoard), absoluteHighScore,
//                        saveProfile);
            }
        }));
    }

    /**
     * Starts a daemon thread to periodically report solver speed and throughput.
     */
    private void startReporterThread() {
        Thread reporterThread = new Thread(() -> {
            while (true) {
                try {
                    Thread.sleep(5000);
                } catch (InterruptedException ignored) {
                    Thread.currentThread().interrupt(); // Restore interrupt status
                    return;
                }
                reportSpeed();
            }
        });
        reporterThread.setDaemon(true);
        reporterThread.start();
    }

    /**
     * Orchestrates the traditional 3-phase pipeline (CPU Seed Gen -> GPU Deep DFS -> GPU Surgeon LNS).
     * This method contains the main loop for the 3-phase strategy.
     */
    private void run3PhasePipeline() {
        logger.info(">>> Running 3-Phase Pipeline Orchestrator...");
        long lastPeriodicSave = System.currentTimeMillis();

        while (true) {
            try {
                handleManualOverride();

                if (this.useGpu) {
                    if (deepestStep >= LNS_THRESHOLD) {
                        runPhase3_GpuSurgeon();
                    } else if (seedPool.size() >= targetBatchSize) {
                        runPhase2_GpuDeepDfs();
                    } else {
                        runPhase1_CpuSeedGen();
                    }
                } else { // CPU-only mode
                    if (deepestStep >= LNS_THRESHOLD) {
                        logger.info(">>> [CPU MODE] Skipping Surgeon mode (GPU only). Triggering CPU Teardown...");
                        consecutiveExtinctions += 15; // Force a teardown to try a new branch
                        triggerBranchScrap();
                    } else {
                        runPhase1_CpuSeedGen();
                    }
                }
                handlePeriodicSave(lastPeriodicSave);
            } catch (Exception e) {
                logger.error(">>> [FATAL ERROR] PIPELINE CRASHED: ", e);
                if (repairReporterScheduler != null) { // repairReporterScheduler is unused, can be removed
                    repairReporterScheduler.shutdownNow();
                }
                try {
                    Thread.sleep(5000);
                } catch (InterruptedException ignored) {
                    Thread.currentThread().interrupt(); // Restore interrupt status
                }
            }
        }
    }

    /**
     * Implements an evolutionary solver strategy using GPU for deep DFS.
     * This method continuously evolves a population of seeds based on their performance.
     */
    private void runEvolutionarySolver() {
        if (!useGpu || gpuEngine == null) {
            logger.error("Evolutionary Solver requires GPU. Exiting.");
            return;
        }
        logger.info(">>> Running Evolutionary Solver...");

        Random random = new Random(System.currentTimeMillis()); // Use a truly random seed
        List<int[]> currentSeeds = new ArrayList<>();

        // Initialize with a few random seeds or from checkpoint
        if (absoluteHighScore > 0) {
            currentSeeds.add(globalBestBoard.clone()); // Use globalBestBoard for initial seed
            logger.info(">>> Initializing evolutionary solver with checkpoint board.");
        } else {
            // Generate a few initial random seeds if no checkpoint
            for (int i = 0; i < numCores * 2; i++) { // Start with a few seeds per core
                int[] blankBoard = new int[256];
                Arrays.fill(blankBoard, -1);
                if (lockCenter) blankBoard[135] = targetPiece;
                currentSeeds.add(blankBoard);
            }
            logger.info(">>> Initializing evolutionary solver with blank boards.");
        }

        int highScore = absoluteHighScore;
        int[] currentBestBoard = globalBestBoard.clone(); // Keep track of the best board found
        int round = 0;

        while (true) {
            round++;
            long t0 = System.currentTimeMillis();

            // --- GPU Round ---
            // The GPU will try to extend each seed as far as possible
            GpuEngine.GpuResult result = gpuEngine.runDeepDfs(
                    currentSeeds, 0, highScore, currentBestBoard, buildOrder
            );

            long elapsed = System.currentTimeMillis() - t0;
            globalGpuTrialCount.addAndGet(result.stepsTaken()); // Update global trial count

            if (result.solved()) {
                logger.info("SOLVED! Round %d", round);
                handleVictory(currentBestBoard); // Use handleVictory for solution saving
                return;
            }

            if (result.newHighScore() > highScore) {
                highScore = result.newHighScore();
                System.arraycopy(currentBestBoard, 0, globalBestBoard, 0, 256); // Update global bestBoard
                absoluteHighScore = highScore; // Update global absoluteHighScore
                updateDisplay(highScore, buildDisplayBoard(globalBestBoard)); // Update display
                RecordManager.saveRecord(buildDisplayBoard(globalBestBoard), absoluteHighScore, saveProfile); // Save record
                logger.info("[Round %d] New highscore: %d pieces (%d ms)",
                        round, highScore, elapsed);
            } else {
                logger.info("[Round %d] No progress. Highscore: %d (%d ms)",
                        round, highScore, elapsed);
            }

            // --- Select best seeds for next round ---
            int nextGenerationSize = (userBatchSizeOverride > 0) ? userBatchSizeOverride : targetBatchSize / 1000; // Example: 50 seeds for next gen
            if (nextGenerationSize == 0) nextGenerationSize = numCores * 2; // Ensure at least some seeds

            List<int[]> nextSeeds = SeedSelector.selectBest(
                    currentSeeds,           // boards that were run
                    result.threadDepths(),    // GPU's maxStepReached per thread
                    nextGenerationSize,     // keep the same number of seeds
                    random
            );

            logger.info("   Seeds: %d -> %d (elite: %d, mutated: %d, random: %d)",
                    currentSeeds.size(),
                    nextSeeds.size(),
                    Math.max(1, nextGenerationSize / 10),
                    Math.max(1, nextGenerationSize * 4 / 10),
                    nextGenerationSize - Math.max(1, nextGenerationSize / 10) - Math.max(1, nextGenerationSize * 4 / 10)
            );

            currentSeeds = nextSeeds;

            // Manual override check within the evolutionary loop
            handleManualOverrideEvolutionary(highScore);
        }
    }

    /**
     * Handles manual override requests within the main solver loop.
     */
    private void handleManualOverride() {
        if (manualOverrideRequested) {
            manualOverrideRequested = false;
            retreat(manualBaseCampTarget, ">>> User Override...");
            try {
                Thread.sleep(1000);
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt(); // Restore interrupt status
            }
        }
    }

    /**
     * Handles manual override requests specifically for the evolutionary solver.
     * @param currentHighScore The current high score of the evolutionary solver.
     */
    private void handleManualOverrideEvolutionary(int currentHighScore) {
        if (manualOverrideRequested) {
            manualOverrideRequested = false;
            int oldHighScore = this.absoluteHighScore;
            this.absoluteHighScore = manualBaseCampTarget; // Update global high score
            retreat(manualBaseCampTarget, ">>> User Override in Evolutionary Solver...");
            // Re-initialize currentSeeds based on the new base camp
            // This part needs to be handled by the caller (runEvolutionarySolver)
            // as currentSeeds is a local variable there.
            // For now, we just update global state and let the next iteration pick it up.
            logger.info(">>> Evolutionary solver re-initialized with new base camp. Old high score: %d, New base camp: %d", oldHighScore, manualBaseCampTarget);
        }
    }


    /**
     * Handles periodic saving of the current best board to a checkpoint file.
     *
     * @param lastPeriodicSave The timestamp of the last periodic save.
     */
    private void handlePeriodicSave(long lastPeriodicSave) {
        if (System.currentTimeMillis() - lastPeriodicSave > 300_000) { // 5 minutes
            synchronized (displayLock) {
                SolverState memoryToSave = new SolverState(
                        buildDisplayBoard(globalBestBoard),
                        absoluteHighScore,
                        this.tabuTenure, // Sends the active Tabu list
                        this.uniqueMaxScoreHashes, // Sends the hashes of explored high-score variants
                        this.topBoards.getRawRegistry() // You'll need to add a getter to TopBoardRegistry to return its List<int[]>
                );

                CheckpointManager.saveSmartState(memoryToSave, saveProfile);//                CheckpointManager.saveRecordCheckpoint(buildDisplayBoard(globalBestBoard), absoluteHighScore,
//                        saveProfile);
            }
            System.currentTimeMillis();
        }
    }

    /**
     * Executes Phase 1: CPU Seed Generation.
     * CPU workers generate initial board configurations (seeds) up to a certain depth (SEED_DEPTH).
     * These seeds are then passed to the GPU for deeper exploration.
     */
    private void runPhase1_CpuSeedGen() {
        Arrays.fill(usedPhysicalPieces, false);

        // Count how many pieces are currently locked in our Base Camp
        int lockedPieces = 0;
        for (int p : flatResumeBoard) {
            if (p != -1 && p != -2) {
                lockedPieces++;
            }
        }

        // Dynamic Handoff: The CPU should always generate seeds 8 pieces deeper than the Base Camp
        currentSeedDepth = (lockedPieces > 0) ? Math.max(SEED_DEPTH, lockedPieces + userCpuHandoffDepth) : SEED_DEPTH;

        currentBatchSize.set(seedPool.size());

        try {
            int activeBatch = (userBatchSizeOverride > 0) ? userBatchSizeOverride : targetBatchSize;
            List<CpuSearchWorker> workers = new ArrayList<>();
            for (int i = 0; i < numCores; i++) {
                workers.add(new CpuSearchWorker(activeBatch));
            }

            List<Future<Boolean>> futures = executor.invokeAll(workers);

            for (Future<Boolean> f : futures) {
                try {
                    f.get(); // If the thread died from an error, it gets thrown HERE.
                } catch (ExecutionException e) {
                    logger.error(">>> [FATAL CPU CRASH] A worker thread died: ", e.getCause());
                }
            }

            // DEADLOCK DETECTOR
            if (this.useGpu && seedPool.isEmpty()) {
                logger.warn(">>> [PHASE 1 DEADLOCK] CPU proved that NO alternative seeds exist for this Base Camp!");
                consecutiveExtinctions++;
                triggerBranchScrap();
            } else if (!this.useGpu) { // CPU-only mode
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

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt(); // Restore interrupt status
            logger.warn("CPU Seed Generation interrupted: " + e.getMessage());
        }
    }

    /**
     * Executes Phase 2: GPU Deep DFS Explorer.
     * Takes a batch of seeds generated by the CPU and performs a deep Depth-First Search on the GPU.
     * Updates the global high score if a better board is found.
     */
    private void runPhase2_GpuDeepDfs() {
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
            System.out.printf("%s >>> [CLIMBING] Current pieces placed: %d / 256",
                    LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss")), result.newHighScore());
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
                logger.error(">>> [FATAL GPU BUG] The GPU returned a board with an illegal edge conflict!");

                int piecesPlaced = countPieces(bestBoardOut);

                if (piecesPlaced >= 240) {
                    logger.info(">>> [RESCUE] High-score board (" + piecesPlaced + ") detected. Executing emergency surgery...");

                    int validPieces = rescueBoard(bestBoardOut);
                    logger.info(">>> [RESCUE] Surgeon removed conflicting pieces. Rescued board has " + validPieces + " valid pieces.");

                    // If the rescued board is still better than our previous deepest step, keep it!
                    if (validPieces > deepestStep) {
                        deepestStep = validPieces;
                        System.arraycopy(bestBoardOut, 0, bestBoard, 0, 256);
                        topBoards.offer(bestBoardOut, deepestStep);
                        updateDisplay(deepestStep, buildDisplayBoard(bestBoard));
                        logger.info(">>> [RESCUE SUCCESS] Rescued board established as new Base Camp!");
                    }
                } else {
                    fatalGpuBugCount++;
                    if (fatalGpuBugCount > 50) {
                        logger.info(">>> [DEEP RESET] Too many conflicts at the finish line. Removing 20 pieces to break the deadlock.");
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

                logger.info(">>> PHASE 2 (GPU) BROKE RECORD! NEW HIGH SCORE: %d <<<", absoluteHighScore);
                analyzeFullBoardPotential(globalBestBoard);

            } else if (deepestStep == absoluteHighScore) {
                int boardHash = Arrays.hashCode(bestBoardOut);
                if (uniqueMaxScoreHashes.add(boardHash)) {
                    analyzeFullBoardPotential(bestBoardOut);
                    logger.info(">>> [PROGRESS] GPU found a new unique variant of %d-pieced board! (Uniquely found " +
                                    "%d times so far)",
                            absoluteHighScore, uniqueMaxScoreHashes.size());
                }
            }
        } else {
            consecutiveGpuStagnation++;
        }

        if (consecutiveGpuStagnation < 10) {
            List<int[]> nextSeeds = SeedSelector.selectBest(
                    seeds,
                    result.threadDepths(),
                    targetBatchSize,
                    new Random()
            );

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
    private void runPhase3_GpuSurgeon() {
        int numClones = 50000;
        int holesToPunch = userSurgeonHoles;

        currentRepairIteration++;
        repairLoopsCounter++;
        totalRepairVariationsTested += numClones;

        int[] sourceBoard = topBoards.nextForRepair();
        if (sourceBoard == null) {
            sourceBoard = bestBoard; // fallback if registry is empty
        }

        List<int[]> variations;
        int actualHoles;
        if (consecutiveExtinctions > 20) {
            actualHoles = (deepestStep > 250) ? 5 : Math.round(holesToPunch / 2.5f);
        } else {
            actualHoles = (deepestStep > 200) ? 5 : holesToPunch;
        }
        variations = surgeon.excavateFrontier(
                sourceBoard, numClones, actualHoles, // Less aggressive initially
                tabuTenure, currentRepairIteration, deepestStep, buildOrder);
        int scoreBefore = deepestStep;
        int[] bestBoardOut = new int[256];
        GpuEngine.GpuResult result = gpuEngine.runRepairMode(variations, deepestStep, bestBoardOut);

        // CLIMBING TRACKER: Print to the log ONLY when the GPU reaches a new depth for this branch
        if (result.newHighScore() > lastReportedDepth) {
            logger.info(">>> [CLIMBING] Current pieces placed: %d / 256", result.newHighScore());
            System.out.printf("%s >>> [CLIMBING] Current pieces placed: %d / 256",
                    LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss")), result.newHighScore());
            lastReportedDepth = result.newHighScore();
        }
        globalGpuTrialCount.addAndGet(result.stepsTaken());

        if (result.solved()) {
            handleVictory(bestBoardOut);
        }

        if (result.newHighScore() > scoreBefore) {
            if (!verifyBoardStrict(bestBoardOut)) {
                logger.error(">>> [FATAL GPU BUG] The GPU returned a board with an illegal edge conflict!");

                int piecesPlaced = countPieces(bestBoardOut);

                if (piecesPlaced >= 240) {
                    logger.info(">>> [RESCUE] High-score board (" + piecesPlaced + ") detected. Executing emergency surgery...");

                    int validPieces = rescueBoard(bestBoardOut);
                    logger.info(">>> [RESCUE] Surgeon removed conflicting pieces. Rescued board has " + validPieces + " valid pieces.");

                    // If the rescued board is still better than our previous deepest step, keep it!
                    if (validPieces > deepestStep) {
                        deepestStep = validPieces;
                        System.arraycopy(bestBoardOut, 0, bestBoard, 0, 256);
                        topBoards.offer(bestBoardOut, deepestStep);
                        updateDisplay(deepestStep, buildDisplayBoard(bestBoard));
                        logger.info(">>> [RESCUE SUCCESS] Rescued board established as new Base Camp!");
                    }
                } else {
                    fatalGpuBugCount++;
                    if (fatalGpuBugCount > 50) {
                        logger.info(">>> [DEEP RESET] Too many conflicts at the finish line. Removing 20 pieces to break the deadlock.");
                        triggerBranchScrap();
                        fatalGpuBugCount = 0;
                    }
                }
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

                logger.info(">>> PHASE 3 (SURGEON) BROKE RECORD! NEW HIGH SCORE: %d <<<", absoluteHighScore);
                analyzeFullBoardPotential(globalBestBoard);
            } else if (deepestStep == absoluteHighScore) {
                int boardHash = Arrays.hashCode(bestBoardOut);
                if (uniqueMaxScoreHashes.add(boardHash)) {
                    analyzeFullBoardPotential(bestBoardOut);
                    logger.info(">>> [PROGRESS] Surgeon got a new unique variant of %d-pieced board! (Uniquely found " +
                                    "%d times so far)",
                            absoluteHighScore, uniqueMaxScoreHashes.size());
                } else {
                    consecutiveExtinctions++; // Treat duplicate as a form of stagnation

                    // Force the Tabu system to ban this exact duplicate configuration
                    updateTabuList(bestBoardOut);
                }
            } else {
                consecutiveExtinctions++;
                updateTabuList(bestBoardOut); // Add to tabu to avoid re-exploring this path too soon
            }
        } else {
            consecutiveExtinctions++;
            if (consecutiveExtinctions >= 20) { // More aggressive teardown if stuck
                triggerBranchScrap();
            }
        }
    }

    /**
     * Updates the Tabu list based on changes between the old best board and a newly found better board.
     * Pieces that have changed position or value are marked as "tabu" for a certain number of future iterations.
     *
     * @param newBoard The newly found best board.
     */
    void updateTabuList(int[] newBoard) {
        // RELAXED TENURE: Shorter memory allows the Surgeon to re-optimize areas much faster.
        int tenureLength = 5 + (absoluteHighScore / 25);

        for (int i = 0; i < 256; i++) {
            // --- THE SURGEON SHIELD ---
            // Ban the Surgeon from ever touching the 4 hints + the 135 center piece
            if (i == 135 || i == 221 || i == 45 || i == 210 || i == 34) {
                tabuTenure[i] = Integer.MAX_VALUE; // Permanently tabu
                continue;
            }

            if (newBoard[i] != bestBoard[i] && newBoard[i] != -1) {
                tabuTenure[i] = currentRepairIteration + tenureLength;
            }
        }
    }

    /**
     * Clears all entries from the Tabu list, making all board positions available for modification.
     */
    void clearTabu() {
        Arrays.fill(tabuTenure, 0);
        logger.info(">>> [TABU] All tenures cleared for fresh Phase 3 cycle.");
    }

    /**
     * Handles a "Dead End" in Phase 3 (Surgeon) or Phase 2 (GPU Stagnation).
     * Employs an Adaptive Large Neighborhood Search (ALNS) strategy combined with
     * Simulated Annealing principles. Stagnation progressively increases the teardown
     * depth. Extreme stagnation triggers a massive structural rollback to escape
     * deep local optima, especially crucial for the Typewriter build strategy.
     */
    private void triggerBranchScrap() {
        // 1. Increment the frustration/stagnation counter
        consecutiveExtinctions++;

        String phaseName = this.useGpu ? "PHASE 3 (GPU)" : "CPU DEEP SEARCH";
        logger.info(">>> [!!!] DEAD END AT %s (Stagnation level: %d) [!!!]", phaseName, consecutiveExtinctions);

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

        if (consecutiveExtinctions > 25) { // Extreme stagnation, reset completely
            lockedPieces = 0;
            retreatAmount = absoluteHighScore;
            consecutiveExtinctions = 0; // Reset counter after extreme rollback
        }

        // Reset the climbing tracker so it starts reporting from the new Base Camp
        lastReportedDepth = lockedPieces;

        if (lockedPieces == 0) {
            poisonedIndex = buildOrder[0];
            poisonedPiece = globalBestBoard[poisonedIndex];

            // Translate to physical piece number for better logging
            int physId = -1;
            for (int i = 0; i < 1024; i++) {
                if (inventory.allOrientations[i] == poisonedPiece) {
                    physId = inventory.physicalMapping[i] + 1;
                    break;
                }
            }
            logger.info(">>> [GLOBAL TABU] Board reset! Banning physical piece #%d (packed: %d) at index %d to force a new branch.",
                    physId, poisonedPiece, poisonedIndex);
        } else if (absoluteHighScore > lockedPieces + 5) {
            poisonedIndex = buildOrder[lockedPieces + 2];
            poisonedPiece = globalBestBoard[poisonedIndex];
            logger.info(">>> [GLOBAL TABU] Poisoned Square Active! Piece %d is banned at index %d.", poisonedPiece,
                    poisonedIndex);
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
     *
     * @param recordBoard The current best board.
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

                        // Check edge requirements (gray side out)
                        if (atNorthEdge && PieceUtils.getNorth(p) != 0) match = false;
                        if (atSouthEdge && PieceUtils.getSouth(p) != 0) match = false;
                        if (atWestEdge && PieceUtils.getWest(p) != 0) match = false;
                        if (atEastEdge && PieceUtils.getEast(p) != 0) match = false;

                        // Check internal compatibility with already placed pieces
                        if (match && row > 0 && simulatedBoard[spot - 16] != -1) { // North neighbor
                            if (PieceUtils.getNorth(p) != PieceUtils.getSouth(simulatedBoard[spot - 16])) match = false;
                        }
                        if (match && col > 0 && simulatedBoard[spot - 1] != -1) { // West neighbor
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

            if (bestBrikIndex != -1) {
                simulatedBoard[spot] = bestOrientedPiece;
                unusedPhysIds.remove(bestBrikIndex);
            } else {
                simulatedBoard[spot] = -1; // Could not find a suitable piece
            }
        }

        // 4. Scan the board and count the edge conflicts
        int totalConflicts = 0;
        for (int row = 0; row < 16; row++) {
            for (int col = 0; col < 16; col++) {
                int idx = row * 16 + col;
                int currentPiece = simulatedBoard[idx];
                if (currentPiece == -1) continue; // Skip empty spots

                // Check East edge
                if (col < 15) {
                    int eastNeighbor = simulatedBoard[idx + 1];
                    if (eastNeighbor != -1 && PieceUtils.getEast(currentPiece) != PieceUtils.getWest(eastNeighbor)) {
                        totalConflicts++;
                    }
                }

                // Check South edge
                if (row < 15) {
                    int southNeighbor = simulatedBoard[idx + 16];
                    if (southNeighbor != -1 && PieceUtils.getSouth(currentPiece) != PieceUtils.getNorth(southNeighbor)) {
                        totalConflicts++;
                    }
                }
            }
        }

        // 5. Log and Save
        logger.info(">>> [FULL BOARD SCAN] Simulated an edge-aware fully laid board. Total internal edge conflicts: " +
                "%d / 480", totalConflicts);

        if (totalConflicts < 30) {
            logger.warn(">>> [!!!] WOW! You are mathematically incredibly close to a full solution!");
        }

        saveFullBoardVariant(simulatedBoard, absoluteHighScore, totalConflicts);
    }

    /**
     * Handles the event of finding a complete solution to the puzzle.
     * Displays the winning board, saves the record, and exits the program.
     *
     * @param winningBoard The 1D integer array representing the solved board.
     */
    private void handleVictory(int[] winningBoard) {
        logger.info(">>> ETERNITY II SOLVED BY GPU PIPELINE!!! <<<");
        updateDisplay(256, buildDisplayBoard(winningBoard));
        RecordManager.saveRecord(buildDisplayBoard(winningBoard), 256, saveProfile);
        consecutiveExtinctions = 0;
        saveAndUploadBucasLink(winningBoard, 256);
        System.exit(0);
    }

    /**
     * Reports the current CPU and GPU search throughput (trials per second).
     * This method is typically run in a separate daemon thread.
     */
    private void reportSpeed() {
        if (isGpuBusy) {
            return; // Don't report CPU speed if GPU is busy with a deep search
        }

        long now = System.currentTimeMillis();
        long elapsed = now - lastThroughputReportTime;

        if (elapsed < 2000) { // Report every 2 seconds
            return;
        }

        long cpuTrials = globalCpuTrialCount.getAndSet(0);
        long gpuTrials = globalGpuTrialCount.getAndSet(0);

        double cpuTps = cpuTrials / (elapsed / 1000.0);
        double gpuTps = gpuTrials / (elapsed / 1000.0);

        if (cpuTps != 0 || gpuTps != 0) {
            logger.info("[SPEED] CPU Phase 1: %,.0f/s  |  GPU Phase 2/3: %,.0f/s", cpuTps, gpuTps);
        }

        lastThroughputReportTime = now;
    }

    /**
     * Resets the board state to a target step, effectively "retreating" the search.
     * Pieces beyond the target step are removed. Hint pieces and the locked center
     * piece are preserved.
     *
     * @param targetStep The depth to retreat to.
     * @param logMessage An optional message to log about the retreat.
     */
    void retreat(int targetStep, String logMessage) {
        deepestStep = Math.max(0, targetStep);
        for (int s = deepestStep; s < 256; s++) {
            bestBoard[buildOrder[s]] = -1;
        }
        if (lockCenter) {
            bestBoard[135] = targetPiece;
        }

        // RE-LOCK HINTS AFTER WIPE
        for (int h = 0; h < 4; h++) {
            if (hintPackedValues[h] != -1) {
                bestBoard[HINT_POSITIONS[h]] = hintPackedValues[h];
            }
        }

        updateDisplay(countPieces(bestBoard), buildDisplayBoard(bestBoard));
        if (logMessage != null) {
            logger.info(logMessage);
        }
    }

    /**
     * Counts the number of placed pieces on a given board.
     *
     * @param board The 1D integer array representing the board.
     * @return The count of non-empty, non-hole pieces.
     */
    int countPieces(int[] board) {
        int count = 0;
        for (int p : board) {
            if (p != -1 && p != -2) {
                count++;
            }
        }
        return count;
    }

    /**
     * Converts a 1D flat board array into a 2D array suitable for display.
     *
     * @param sourceArray The 1D integer array representing the board.
     * @return A 2D integer array (16x16) for display.
     */
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

    /**
     * Updates the graphical display with the current board state and score.
     *
     * @param score        The current number of pieces placed.
     * @param displayBoard The 2D integer array representing the board to display.
     */
    private void updateDisplay(int score, int[][] displayBoard) {
        Eternity.updateDisplay(score, this.absoluteHighScore, displayBoard);
    }

    /**
     * Saves a Bucas link for the given board and uploads it to a drive.
     *
     * @param board The 1D integer array representing the board.
     * @param score The score associated with this board.
     */
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

            logger.info(">>> Saved local Bucas-link: %s", linkFile.getName());

            uploadToDrive(linkFile, "text/plain", saveProfile);

        } catch (Exception e) {
            logger.error(">>> [ERROR] Couldn't save/upload Bucas-link: %s", e.getMessage());
        }
    }

    /**
     * Prints the physical piece numbers of the board to the log.
     *
     * @param board The 1D integer array representing the board.
     * @param score The score associated with this board.
     */
    private void printPhysicalBoard(int[] board, int score) {
        logger.info(">>> PHYSICAL PIECE-NUMBERS FOR RECORD (%d PIECES):", score);
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
     *
     * @param simulatedBoard The full 256-piece board array.
     * @param baseScore      The number of correct pieces placed before the random fill.
     * @param conflicts      The total number of edge color mismatches on the full board.
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
        String baseFilename = "Errors" + conflicts + "_Base" + baseScore + "_" + timeId;

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
                    if (col < 15) {
                        line.append(",");
                    }
                }
                writer.println(line);
            }
        } catch (Exception e) {
            logger.error(">>> Error saving Full Board CSV: %s", e.getMessage());
        }

        // 4. Save the Bucas link (allowing rapid visual inspection in the browser)
        try (java.io.PrintWriter writer = new java.io.PrintWriter(new java.io.File(folder,
                baseFilename + "_link.txt"))) {
            writer.println("Base Score (Correctly placed foundation): " + baseScore);
            writer.println("Edge Conflicts (Mismatches): " + conflicts);
            writer.println("\nClick the Bucas Link below to view this specific variant:");
            writer.println(BucasExporter.exportBoard(simulatedBoard));
        } catch (Exception e) {
            logger.error(">>> Error saving Full Board Bucas Link: %s", e.getMessage());
        }

        // 5. PICTURE / IMAGE GENERATION:
        RecordManager.saveImage(buildDisplayBoard(simulatedBoard),
                new java.io.File(folder, baseFilename + ".png").getAbsolutePath());
    }

    /**
     * Scans the board and explicitly logs the exact location and colors of any edge conflict.
     * This acts as an absolute firewall against corrupted GPU "Frankenstein" boards.
     *
     * @param board The 1D integer array representing the board to verify.
     * @return True if the board is mathematically flawless, false otherwise.
     */
    boolean verifyBoardStrict(int[] board) {
        for (int row = 0; row < 16; row++) {
            for (int col = 0; col < 16; col++) {
                int idx = row * 16 + col;
                int p = board[idx];

                // Skip empty spots
                if (p == -1 || p == -2) {
                    continue;
                }

                // Check East edge
                if (col < 15) {
                    int eastP = board[idx + 1];
                    if (eastP != -1 && eastP != -2) {
                        int myEast = PieceUtils.getEast(p);
                        int theirWest = PieceUtils.getWest(eastP);
                        if (myEast != theirWest) {
                            logger.error(">>> [DIAGNOSTIC] CONFLICT EAST: Piece at idx %d (row %d, col %d) has East=%d, but neighbor at idx %d has West=%d",
                                    idx, row, col, myEast, idx + 1, theirWest);
                            return false;
                        }
                    }
                }

                // Check South edge
                if (row < 15) {
                    int southP = board[idx + 16];
                    if (southP != -1 && southP != -2) {
                        int mySouth = PieceUtils.getSouth(p);
                        int theirNorth = PieceUtils.getNorth(southP);
                        if (mySouth != theirNorth) {
                            logger.error(">>> [DIAGNOSTIC] CONFLICT SOUTH: Piece at idx %d (row %d, col %d) has South=%d, but neighbor at idx %d has North=%d",
                                    idx, row, col, mySouth, idx + 16, theirNorth);
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }

    /**
     * Emergency surgery to strip out conflicts and create a clean crater
     * around the bad pieces, while protecting the hints and center.
     */
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
                    // Punch out the piece and its immediate neighbors to create a crater.
                    // Protecting center (135) and hints (221, 45, 210, 34).
                    int[] neighbors = {idx, idx + 1, idx + 16, idx - 1, idx - 16};

                    for (int n : neighbors) {
                        if (n >= 0 && n < 256) {
                            if (n != 135 && n != 221 && n != 45 && n != 210 && n != 34) {
                                rescuedBoard[n] = -1;
                            }
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

        // Copy the rescued board back into the original array
        System.arraycopy(rescuedBoard, 0, corruptedBoard, 0, 256);
        return validPieces;
    }

    /**
     * Generates the build order for the Spiral strategy, filling the `buildOrder` array.
     */
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

    /**
     * A registry that tracks the top high-scoring boards to provide diversity
     * for Phase 3 repair operations.
     */
    class TopBoardRegistry {
        private final List<int[]> registry = new ArrayList<>();
        private int currentIndex = 0;
        private static final int MAX_CAPACITY = 20;

        /**
         * Offers a board to the registry. If the board is unique and the registry
         * is not at max capacity, it is added.
         *
         * @param board The board to offer.
         * @param score The score associated with the board (unused in current implementation).
         */
        public synchronized void offer(int[] board, int score) {
            // Check if this board configuration is already in the registry
            int boardHash = Arrays.hashCode(board);
            for (int[] existing : registry) {
                if (Arrays.hashCode(existing) == boardHash) return;
            }

            if (registry.size() < MAX_CAPACITY) {
                registry.add(Arrays.copyOf(board, 256));
            } else {
                // For simplicity, if at max capacity, we don't add new boards unless they are better.
                // A more advanced strategy would replace the worst board.
            }
        }

        /**
         * Retrieves the next board from the registry in a round-robin fashion for repair.
         *
         * @return The next board to be repaired, or null if the registry is empty.
         */
        public synchronized int[] nextForRepair() {
            if (registry.isEmpty()) return null;
            int[] board = registry.get(currentIndex);
            currentIndex = (currentIndex + 1) % registry.size();
            return board;
        }

        /**
         * Clears all boards from the registry.
         */
        public synchronized void clear() {
            registry.clear();
            currentIndex = 0;
        }

        public List getRawRegistry() {
            return registry;
        }
    }

    /**
     * CpuSearchWorker is a Callable task that performs a Depth-First Search
     * on the CPU to generate initial board seeds up to SEED_DEPTH.
     * These seeds are then added to a shared pool for GPU processing.
     */
    private class CpuSearchWorker implements Callable<Boolean> {
        private final int[] localBoard = new int[256];
        private final int[] localResumeBoard = new int[256];
        private final boolean[] localUsed = new boolean[256];
        private final Random rnd = new Random();
        private final int activeBatch;
        private long localTrialCount = 0;

        /**
         * Constructs a CpuSearchWorker.
         * Initializes the worker's local board state from the global `flatResumeBoard`
         * and marks already placed pieces as used.
         *
         * @param activeBatch The target batch size for seed generation.
         */
        public CpuSearchWorker(int activeBatch) {
            this.activeBatch = activeBatch;
            System.arraycopy(flatResumeBoard, 0, localBoard, 0, 256);
            System.arraycopy(flatResumeBoard, 0, localResumeBoard, 0, 256);
            System.arraycopy(usedPhysicalPieces, 0, localUsed, 0, 256); // This seems redundant as usedPhysicalPieces is always reset in runPhase1_CpuSeedGen

            // Mark all pieces from flatResumeBoard as used
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
            // Lock center piece if applicable
            if (lockCenter) {
                localBoard[135] = targetPiece;
                if (centerPhysicalIdx != -1) {
                    localUsed[centerPhysicalIdx] = true;
                }
            }
            // Lock hint pieces
            for (int h = 0; h < 4; h++) {
                int packed = hintPackedValues[h];
                if (packed != -1) {
                    localBoard[HINT_POSITIONS[h]] = packed;
                    localUsed[hintPhysicalIndices[h]] = true;
                }
            }
        }

        /**
         * The main computation of the worker, performing the DFS.
         *
         * @return True if a full solution is found (unlikely for seed generation), false otherwise.
         * @throws Exception if an error occurs during computation.
         */
        @Override
        public Boolean call() throws Exception {
            boolean result = solve(0);
            globalCpuTrialCount.addAndGet(localTrialCount); // Flush remaining trials
            return result;
        }

        /**
         * Recursive DFS method to generate seeds.
         *
         * @param step The current depth (number of pieces placed).
         * @return True if a full solution is found, false otherwise.
         */
        private boolean solve(int step) {
            if (manualOverrideRequested || (useGpu && currentBatchSize.get() >= activeBatch)) {
                return false;
            }

            if (step == 256) { // Full solution found by CPU
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
            if (localTrialCount >= 1000) { // Batch updates to avoid memory contention
                globalCpuTrialCount.addAndGet(localTrialCount);
                localTrialCount = 0;
            }

            // Update deepestStep and display if progress is made
            if (step > deepestStep) {
                synchronized (displayLock) {
                    if (step > deepestStep) {
                        deepestStep = step;
                        updateDisplay(deepestStep, buildDisplayBoard(localBoard));

                        if (!useGpu && deepestStep > absoluteHighScore) { // CPU-only mode high score update
                            absoluteHighScore = deepestStep;
                            System.arraycopy(localBoard, 0, globalBestBoard, 0, 256);
                            RecordManager.saveRecord(buildDisplayBoard(globalBestBoard), absoluteHighScore,
                                    saveProfile);
                            CheckpointManager.saveRecordCheckpoint(buildDisplayBoard(globalBestBoard),
                                    absoluteHighScore, saveProfile);
                            consecutiveExtinctions = 0;
                            saveAndUploadBucasLink(globalBestBoard, absoluteHighScore);
                            logger.info(">>> [CPU MODE] NEW HIGH SCORE: %d ", absoluteHighScore);
                        }
                    }
                }
            }

            // If using GPU, and reached seed depth, add to pool and backtrack
            if (useGpu && step == currentSeedDepth) {
                int[] seed = new int[256];
                System.arraycopy(localBoard, 0, seed, 0, 256);
                seedPool.add(seed);
                currentBatchSize.incrementAndGet();
                return false; // Force backtracking to find more seeds
            }

            int boardIdx = buildOrder[step];
            // Skip already placed pieces (e.g., from checkpoint, hints, or locked center)
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

            // Filter candidates based on flatResumeBoard (if resuming from a checkpoint)
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
                if (step > 200) { // Log dead ends only for deeper searches
                    logger.error(">>> [DEAD END] No candidate at index %d (step %d)!", boardIdx, step);
                }
                return false;
            }

            int[] orientIdxs = new int[candidateCount];
            int k = 0;
            for (int oi = candidates.nextSetBit(0); oi >= 0; oi = candidates.nextSetBit(oi + 1)) {
                orientIdxs[k++] = oi;
            }
            int offset = rnd.nextInt(candidateCount); // Randomize starting point for candidates

            for (int i = 0; i < candidateCount; i++) {
                if (useGpu && seedPool.size() >= activeBatch) { // Stop if seed pool is full
                    return false;
                }

                int orientationIdx = orientIdxs[(i + offset) % candidateCount];
                int p = inventory.allOrientations[orientationIdx];
                int physicalIdx = inventory.physicalMapping[orientationIdx];

                // Additional checks for edge compatibility (redundant if compatIndex is perfect, but safe)
                if (eastReq != CompatibilityIndex.WILDCARD && PieceUtils.getEast(p) != eastReq) continue;
                if (southReq != CompatibilityIndex.WILDCARD && PieceUtils.getSouth(p) != southReq) continue;

                // Global Tabu check
                if (boardIdx == poisonedIndex && p == poisonedPiece) {
                    continue;
                }

                if (passesLookahead(p, step, row, col, boardIdx)) {
                    localBoard[boardIdx] = p;
                    localUsed[physicalIdx] = true;
                    int ghost = localResumeBoard[boardIdx]; // Save original resume piece
                    localResumeBoard[boardIdx] = -1; // Mark as placed for this branch

                    if (solve(step + 1)) {
                        return true;
                    }

                    // Backtrack
                    localBoard[boardIdx] = -1;
                    localUsed[physicalIdx] = false;
                    localResumeBoard[boardIdx] = ghost; // Restore original resume piece
                }
            }
            return false;
        }

        /**
         * Performs a 2-step lookahead to check if placing a piece `p` at `idx`
         * would lead to an immediate dead-end for its neighbors.
         *
         * @param p    The piece to check.
         * @param step The current step in the DFS.
         * @param row  The row of the current position.
         * @param col  The column of the current position.
         * @param idx  The 1D index of the current position.
         * @return True if the placement passes the lookahead check, false otherwise.
         */
        private boolean passesLookahead(int p, int step, int row, int col, int idx) {
            // Check South neighbor
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

                // 2-step lookahead for south-south neighbor
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

            // Check East neighbor
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

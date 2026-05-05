package dk.puzzle;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public class MasterSolverPBP implements Runnable {

    // ==========================================================
    // SYSTEM SETTINGS & COMPONENTS
    // ==========================================================
    public static final int SWISS_CHEESE_LEVEL = 195;
    private final PieceInventory inventory;
    private final CompatibilityIndex compatIndex;
    private final SurgeonHeuristics surgeon;
    private GpuEngine gpuEngine;
    private final boolean useGpu;

    // Threading and execution
    private final ExecutorService executor;
    private final int numCores;
    private final int targetBatchSize = 10000;
    private volatile int userBatchSizeOverride = -1;
    private final AtomicInteger currentBatchSize = new AtomicInteger(0);

    // ==========================================================
    // STATE & BOARD DATA
    // ==========================================================
    private final int[] flatBoard = new int[256];
    private final int[] bestBoard = new int[256];
    private final int[] flatResumeBoard = new int[256];
    private final boolean[] usedPhysicalPieces = new boolean[256];
    private final int[] tabuTenure = new int[256];
    private final int[] buildOrder = new int[256];

    private volatile int absoluteHighScore = 0;
    private volatile int deepestStep = 0;
    private int currentRepairIteration = 0;
    private int consecutiveExtinctions = 0;
    private int consecutiveExhaustions = 0;

    private final boolean lockCenter;
    private final int targetPiece;
    private int centerPhysicalIdx = -1;
    private final String saveProfile;
    private final BuildStrategy currentStrategy;
    private final Object displayLock = new Object();

    // ==========================================================
    // METRICS & OVERRIDES
    // ==========================================================
    private final AtomicLong globalCpuTrialCount = new AtomicLong(0);
    private final AtomicLong globalGpuTrialCount = new AtomicLong(0);
    private long lastThroughputReportTime = System.currentTimeMillis();
    private long lastProgressTimestamp = System.currentTimeMillis();
    private long repairStartTime = 0;
    private long repairLoopsCounter = 0;
    private long totalRepairVariationsTested = 0;
    private long lastRepairPrintTime = 0;

    private volatile int stagnationLimitMinutes = 20;
    private volatile boolean manualOverrideRequested = false;
    private volatile int manualBaseCampTarget = 0;
    private volatile double extinctionThreshold = 0.98;

    public MasterSolverPBP(PieceInventory inventory, int trueCenterPiece, boolean useGpu, BuildStrategy strategy, boolean lockCenter) {
        Arrays.fill(flatBoard, -1);
        Arrays.fill(bestBoard, -1);
        Arrays.fill(flatResumeBoard, -1);

        this.inventory = inventory;
        this.targetPiece = trueCenterPiece;
        this.currentStrategy = strategy;
        this.lockCenter = lockCenter;
        this.saveProfile = strategy.name() + (lockCenter ? "_LOCKED" : "_UNLOCKED");

        // Start CPU Thread Pool
        this.numCores = Runtime.getRuntime().availableProcessors();
        this.executor = Executors.newFixedThreadPool(numCores);
        System.out.println(timestamp() + ">>> Multithreading activity with " + numCores + " dedicated CPU-kernels.");

        // Find the physical index of the center piece
        for (int i = 0; i < 1024; i++) {
            if (inventory.allOrientations[i] == targetPiece) {
                this.centerPhysicalIdx = inventory.physicalMapping[i];
                break;
            }
        }

        // Generate build order (Typewriter by default based on our recent changes)
        for (int i = 0; i < 256; i++) {
            buildOrder[i] = i;
        }

        // Initialize sub-components
        this.compatIndex = new CompatibilityIndex(inventory.allOrientations, inventory.physicalMapping);
        this.surgeon = new SurgeonHeuristics(lockCenter, 0.70); // 70% targeted holes
        this.useGpu = useGpu;

        // 1. Initialize the session high score from the persistent record file
        this.absoluteHighScore = RecordManager.getHighScore(saveProfile);

        // Load Checkpoints
        int[][] loaded = CheckpointManager.loadSmartCheckpoint(saveProfile);
        if (loaded != null) {
            int loadedCount = 0;

            // Tæl manuelt, hvor mange fysiske brikker filen rent faktisk indeholder
            for (int r = 0; r < 16; r++) {
                if (loaded[r] == null) continue; // <--- SIKKERHEDS-TJEK: Spring tomme rækker over
                for (int c = 0; c < 16; c++) {
                    int p = loaded[r][c];
                    if (p != -1 && p != 0) {
                        bestBoard[r * 16 + c] = p;
                        loadedCount++;
                    }
                }
            }

            if (loadedCount > 0) {
                // 2. Set the current search depth to the checkpoint's level
                this.deepestStep = loadedCount;

                // 3. Only update the absolute high score if the checkpoint board is actually better
                if (loadedCount > this.absoluteHighScore) {
                    this.absoluteHighScore = loadedCount;
                }

                if (lockCenter) bestBoard[135] = targetPiece;

                System.arraycopy(bestBoard, 0, flatBoard, 0, 256);
                System.arraycopy(bestBoard, 0, flatResumeBoard, 0, 256);
                updateDisplay(absoluteHighScore, buildDisplayBoard(bestBoard));
                System.out.println(timestamp() + ">>> Loaded checkpoint at " + loadedCount + " pieces. Record remains: " + absoluteHighScore);
            }
        }
    }

    public void setStagnationLimit(int minutes) {
        this.stagnationLimitMinutes = minutes;
    }

    public void setBatchSizeOverride(int size) {
        this.userBatchSizeOverride = size;
    }

    public void triggerManualOverride(int targetBaseCamp) {
        this.manualBaseCampTarget = targetBaseCamp;
        this.manualOverrideRequested = true;
    }

    public void setExtinctionThreshold(double threshold) {
        this.extinctionThreshold = threshold;
    }

    public void setTargetedHolesPercentage(double percentage) {
        if (this.surgeon != null) {
            this.surgeon.setTargetedHolesPercentage(percentage);
        }
    }

    private String timestamp() {
        return "[" + LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss")) + "] ";
    }

    @Override
    public void run() {
        try {
            // Register a Shutdown Hook to save when the program is closed or stopped
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                System.out.println("\n" + timestamp() + ">>> Shutdown hook: Saving final checkpoint...");
                synchronized (displayLock) {
                    CheckpointManager.saveRecordCheckpoint(buildDisplayBoard(bestBoard), absoluteHighScore, saveProfile);
                }
            }));

            System.out.println(timestamp() + "Starting Hybrid Orchestrator (Profile: " + saveProfile + ")...");
            long lastPeriodicSave = System.currentTimeMillis();

            if (this.useGpu) {
                this.gpuEngine = new GpuEngine(inventory, lockCenter);
            } else {
                this.gpuEngine = null;
            }

            while (true) {
                reportSpeed();
                // =======================================================
                // MANUAL OVERRIDE CONTROL
                // =======================================================
                if (manualOverrideRequested) {
                    manualOverrideRequested = false;
                    System.out.println("\n" + timestamp() + ">>> [!] USER FORCED JUMP TO " + manualBaseCampTarget + " PIECES [!] <<<");

                    absoluteHighScore = manualBaseCampTarget;
                    retreat(manualBaseCampTarget, timestamp() + ">>> Board reset. Waiting 1 second for UI to sync...");

                    deepestStep = manualBaseCampTarget;
                    lastProgressTimestamp = System.currentTimeMillis();
                    resetCounters();
                    currentBatchSize.set(0);

                    // Freeze for 1 second so the GUI updates and the user can visually confirm the jump
                    try {
                        Thread.sleep(1000);
                    } catch (InterruptedException ignored) {
                    }
                }

                handleGlobalStagnation();
                reportSpeed();

                // =======================================================
                // PHASE ROUTING
                // =======================================================
                if (deepestStep < SWISS_CHEESE_LEVEL || gpuEngine == null) {
                    runCpuPhase();
                } else {
                    runGpuSurgeonPhase();
                }

                if (System.currentTimeMillis() - lastPeriodicSave > 300_000) {
                    synchronized (displayLock) {
                        CheckpointManager.saveRecordCheckpoint(buildDisplayBoard(bestBoard), absoluteHighScore, saveProfile);
                    }
                    lastPeriodicSave = System.currentTimeMillis();
                }
            }
        } finally {
            CheckpointManager.saveWorkingState(buildDisplayBoard(bestBoard));
        }
    }

    // ==========================================================
    // PHASE 1: CPU BULLDOZER LOGIC
    // ==========================================================
    private void runCpuPhase() {
        int lockedPieces = Math.max(0, deepestStep - 40);
        prepareSearchIteration(lockedPieces);

        boolean spaceExhausted = true;
        try {
            if (executeCpuSeedGeneration()) {
                return; // Interrupted or fully solved by CPU
            }

            // Check if CPU is stuck
            int foundSeeds = currentBatchSize.get();
            if (foundSeeds > 0) {
                spaceExhausted = false; // CPU is progressing, keep going
                currentBatchSize.set(0);
            } else {
                spaceExhausted = true; // CPU found zero paths forward
            }

        } catch (PoisonedBaseCampException e) {
            spaceExhausted = false;
            triggerStagnationKick(); // If manually thrown
        } catch (ExecutionException | InterruptedException e) {
            e.printStackTrace();
            return;
        }

        if (spaceExhausted) {
            handleCpuExhaustion();
        }
    }

    private void prepareSearchIteration(int lockedPieces) {
        Arrays.fill(flatResumeBoard, -1);
        Arrays.fill(usedPhysicalPieces, false);

        if (lockedPieces > 0) {
            for (int step = 0; step < lockedPieces; step++) {
                int idx = buildOrder[step];
                if (lockCenter && idx == 135) continue;
                flatResumeBoard[idx] = bestBoard[idx];
            }
        }
        currentBatchSize.set(0);
    }

    private boolean executeCpuSeedGeneration() throws InterruptedException, ExecutionException {
        int activeBatch = (userBatchSizeOverride > 0) ? userBatchSizeOverride : targetBatchSize;
        List<CpuSearchWorker> workers = new ArrayList<>();
        for (int i = 0; i < numCores; i++) workers.add(new CpuSearchWorker(activeBatch));

        List<Future<Boolean>> results = executor.invokeAll(workers);
        for (Future<Boolean> f : results) {
            if (f.get()) return true;
        }
        return false;
    }

    private void handleCpuExhaustion() {
        consecutiveExhaustions++;
        if (consecutiveExhaustions >= 15) {
            // CPU is repeatedly finding 0 paths. Retreat slightly to find a healthy branch.
            int retreatAmount = 5 + new Random().nextInt(30);
            retreat(deepestStep - retreatAmount, timestamp() + ">>> [RECOVERY] Zero paths found. Retreating " + retreatAmount + " pieces...");
            consecutiveExhaustions = 0;
        } else {
            deepestStep = absoluteHighScore;
        }
    }

    // ==========================================================
    // PHASE 2: GPU SURGEON (LNS) LOGIC
    // ==========================================================
    private void runGpuSurgeonPhase() {
        if (repairStartTime == 0) repairStartTime = System.currentTimeMillis();
        int numClones = 10000;
        int holesToPunch = 15;

        totalRepairVariationsTested += numClones;
        repairLoopsCounter++;
        currentRepairIteration++;

        // 1. Generate Swiss Cheese boards
        List<int[]> swissCheeseBoards = surgeon.punchHoles(bestBoard, numClones, holesToPunch, tabuTenure, currentRepairIteration, deepestStep, buildOrder);
        // Print visual updates for the Surgeon every 2 seconds
        long now = System.currentTimeMillis();
        if (now - lastRepairPrintTime > 2000) {
            double secondsSinceLastPrint = (now - lastRepairPrintTime) / 1000.0;
            double batchesPerSec = repairLoopsCounter / secondsSinceLastPrint;
            long secondsRunning = (now - repairStartTime) / 1000;
            String timeFormatted = String.format("%02d:%02d:%02d", secondsRunning / 3600, (secondsRunning % 3600) / 60, secondsRunning % 60);

            System.out.println(String.format("%s >>> [REPAIR MODE] Uptime: %s | Speed: %.1f batches/sec | Total variations: %,d",
                    timestamp(), timeFormatted, batchesPerSec, totalRepairVariationsTested));

            // Display the "holes" on the GUI
            int[] visualBoard = new int[256];
            System.arraycopy(swissCheeseBoards.get(0), 0, visualBoard, 0, 256);
            for (int i = 0; i < 256; i++) {
                if (visualBoard[i] == -2) visualBoard[i] = -1;
            }
            updateDisplay(countPieces(visualBoard), buildDisplayBoard(visualBoard));

            lastRepairPrintTime = now;
            repairLoopsCounter = 0;
        }

        // 2. Send boards to GPU
        int scoreBefore = deepestStep;
        int[] bestBoardOut = new int[256];
        GpuEngine.GpuResult result = gpuEngine.runRepairMode(swissCheeseBoards, deepestStep, bestBoardOut);

        // 3. Collect statistics and logic
        globalGpuTrialCount.addAndGet(result.stepsTaken);

        if (result.solved) {
            System.out.println("\n" + timestamp() + ">>> ETERNITY II SOLVED BY REPAIR MODE (GPU)!!! <<<");
            updateDisplay(256, buildDisplayBoard(bestBoardOut));
            RecordManager.saveRecord(buildDisplayBoard(bestBoardOut), 256, saveProfile);
            System.exit(0);
        }

        if (result.newHighScore > scoreBefore) {
            // Succes! Kirurgen forbedrede denne gren!
            deepestStep = result.newHighScore;
            updateTabuList(bestBoardOut);
            System.arraycopy(bestBoardOut, 0, bestBoard, 0, 256);
            updateDisplay(deepestStep, buildDisplayBoard(bestBoard));

            // Tjek om denne nye gren OGSÅ slår The All-Time High Score
            if (deepestStep > absoluteHighScore) {
                absoluteHighScore = deepestStep;
                RecordManager.saveRecord(buildDisplayBoard(bestBoard), absoluteHighScore, saveProfile);
                CheckpointManager.saveRecordCheckpoint(buildDisplayBoard(bestBoard), absoluteHighScore, saveProfile);
                System.out.println("\n" + timestamp() + ">>> SURGEON BROKE THE RECORD! NEW HIGH SCORE: " + absoluteHighScore + " <<<");
            } else {
                // Vi forbedrer os, men er ikke nået op til rekorden endnu
                System.out.println(timestamp() + ">>> SURGEON PROGRESSED THE BRANCH TO: " + deepestStep + " PIECES <<<");
            }

            consecutiveExtinctions = 0;
            consecutiveExhaustions = 0;
            resetCounters();
        } else {
            // Failure! Local optimum detected.
            consecutiveExtinctions++;
            if (consecutiveExtinctions >= 10) {
                triggerStagnationKick();
            }
        }
        currentBatchSize.set(0);
    }

    private void updateTabuList(int[] newBoard) {
        int changedPieces = 0;
        for (int i = 0; i < 256; i++) {
            // If a piece changed position, lock it down to force the search elsewhere
            if (newBoard[i] != bestBoard[i] && newBoard[i] != -1) {
                tabuTenure[i] = currentRepairIteration + 25; // Protected for 25 rounds
                changedPieces++;
            }
        }
        if (changedPieces > 0) {
            System.out.println(timestamp() + ">>> TABU SEARCH: Protected " + changedPieces + " recently moved pieces for 25 batches.");
        }
    }

    private void triggerStagnationKick() {
        int kickSize = 60; // Tear down 60 pieces to escape the local optimum
        int targetStep = Math.max(0, deepestStep - kickSize);

//        System.out.println("\n" + timestamp() + ">>> [!!!] DEAD END AT DEPTH " + deepestStep + " PIECES.");
        retreat(targetStep, timestamp() + ">>> Rebuilding foundation from piece " + targetStep);
        consecutiveExtinctions = 0;
    }

    // ==========================================================
    // HELPER METHODS
    // ==========================================================
    private void handleGlobalStagnation() {
        if (currentStrategy == BuildStrategy.TYPEWRITER) return; // Only used for legacy Spiral
        long minutesSinceProgress = (System.currentTimeMillis() - lastProgressTimestamp) / 60000;
        if (minutesSinceProgress >= stagnationLimitMinutes && deepestStep > 0 && deepestStep <= SWISS_CHEESE_LEVEL) {
            int deepRetreat = 40 + new Random().nextInt(41);
            String msg = "\n" + timestamp() + " [!!!] AUTONOMOUS DEEP EXTINCTION [!!!]\n" +
                    timestamp() + " No progression in the last " + minutesSinceProgress + " minutes.\n" +
                    timestamp() + " Forces Base Camp all the way down to piece : " + deepRetreat + "\n";
            retreat(deepRetreat, msg);
            lastProgressTimestamp = System.currentTimeMillis();
            Arrays.fill(flatResumeBoard, -1);
        }
    }

    private void reportSpeed() {
        long now = System.currentTimeMillis();
        long elapsed = now - lastThroughputReportTime;
        if (elapsed >= 5000) {
            long cpuTrials = globalCpuTrialCount.getAndSet(0);
            long gpuTrials = globalGpuTrialCount.getAndSet(0);

            double cpuTps = cpuTrials / (elapsed / 1000.0);
            double gpuTps = gpuTrials / (elapsed / 1000.0);

            if (cpuTrials > 0 || gpuTrials > 0) {
                System.out.printf("%s[SPEED] CPU: %,.0f trials/sec  |  GPU: %,.0f trials/sec\n",
                        timestamp(), cpuTps, gpuTps);
            }
            lastThroughputReportTime = now;
        }
    }

    private void retreat(int targetStep, String logMessage) {
        deepestStep = Math.max(0, targetStep);
        for (int s = deepestStep; s < 256; s++) bestBoard[buildOrder[s]] = -1;
        if (lockCenter) bestBoard[135] = targetPiece;

        updateDisplay(countPieces(bestBoard), buildDisplayBoard(bestBoard));
        if (logMessage != null) System.out.println(logMessage);
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
            if (sourceArray[i] == -1) continue;
            int r = i / 16;
            int c = i % 16;
            displayBoard[r][c] = sourceArray[i];
        }
        return displayBoard;
    }

    private void updateDisplay(int score, int[][] displayBoard) {
        Main.updateDisplay(score, this.absoluteHighScore, displayBoard);
    }

    public enum BuildStrategy {TYPEWRITER, SPIRAL}

    private static class EvolutionLeapException extends RuntimeException {
    }

    private static class PoisonedBaseCampException extends RuntimeException {
    }

    // ==========================================================
    // CPU SEARCH WORKER (The BitMask Bulldozer)
    // ==========================================================
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

            // Mark resumed pieces as used
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
        }

        @Override
        public Boolean call() {
            return solve(0);
        }

        private boolean solve(int step) {
            if (manualOverrideRequested) {
                return false; // Interrupt gracefully
            }
            if (currentBatchSize.get() >= activeBatch) {
                return false;
            }

            localTrialCount++;
            if (localTrialCount >= 100000) {
                globalCpuTrialCount.addAndGet(localTrialCount);
                localTrialCount = 0;

                // YIELD TO THE SURGEON:
                // If we reached the endgame, force the CPU to abort so the GPU Surgeon can take over.
                if (gpuEngine != null && deepestStep >= SWISS_CHEESE_LEVEL) {
                    currentBatchSize.set(activeBatch);
                    return false;
                }
            }

            if (step == 256) return true;

            // Register depth BEFORE handoff logic
            updateProgress(step);

            int boardIdx = buildOrder[step];
            if (localBoard[boardIdx] != -1) {
                return solve(step + 1);
            }

            if (lockCenter && boardIdx == 135) {
                return solve(step + 1);
            }

            int row = boardIdx / 16;
            int col = boardIdx % 16;

            int northReq = (row == 0) ? 0 : (localBoard[boardIdx - 16] != -1 ? PieceUtils.getSouth(localBoard[boardIdx - 16]) : CompatibilityIndex.WILDCARD);
            int southReq = (row == 15) ? 0 : (localBoard[boardIdx + 16] != -1 ? PieceUtils.getNorth(localBoard[boardIdx + 16]) : CompatibilityIndex.WILDCARD);
            int westReq = (col == 0) ? 0 : (localBoard[boardIdx - 1] != -1 ? PieceUtils.getEast(localBoard[boardIdx - 1]) : CompatibilityIndex.WILDCARD);
            int eastReq = (col == 15) ? 0 : (localBoard[boardIdx + 1] != -1 ? PieceUtils.getWest(localBoard[boardIdx + 1]) : CompatibilityIndex.WILDCARD);

            // 1. Query the BitMask Index
            java.util.BitSet candidates = compatIndex.candidatesFor(northReq, eastReq, southReq, westReq);
            compatIndex.andNotUsed(candidates, localUsed);

            // 2. Force ghost piece usage if applicable
            if (localResumeBoard[boardIdx] != -1) {
                int resumeP = localResumeBoard[boardIdx];
                for (int oi = candidates.nextSetBit(0); oi >= 0; oi = candidates.nextSetBit(oi + 1)) {
                    if (inventory.allOrientations[oi] != resumeP) candidates.clear(oi);
                }
            }

            int candidateCount = candidates.cardinality();
            if (candidateCount == 0) return false;

            // 3. Convert to randomized array
            int[] orientIdxs = new int[candidateCount];
            int k = 0;
            for (int oi = candidates.nextSetBit(0); oi >= 0; oi = candidates.nextSetBit(oi + 1)) {
                orientIdxs[k++] = oi;
            }
            int offset = rnd.nextInt(candidateCount);

            // 4. Test pieces
            for (int i = 0; i < candidateCount; i++) {
                if (currentBatchSize.get() >= activeBatch) return false;

                int orientationIdx = orientIdxs[(i + offset) % candidateCount];
                int p = inventory.allOrientations[orientationIdx];
                int physicalIdx = inventory.physicalMapping[orientationIdx];

                // Check "The Future"
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
                            CheckpointManager.saveRecordCheckpoint(buildDisplayBoard(bestBoard), absoluteHighScore, saveProfile);
                            System.out.println(timestamp() + ">>> NEW ALL-TIME HIGH SCORE: " + absoluteHighScore + " PIECES! <<<");
                        }
                    }
                }
            }
        }

        private boolean passesLookahead(int p, int step, int row, int col, int idx) {
            // --- Check south neighbor ---
            if (row < 15 && localBoard[idx + 16] == -1) {
                int reqN = PieceUtils.getSouth(p);
                int reqE = (col == 15) ? 0 : (localBoard[idx + 17] != -1 ? PieceUtils.getWest(localBoard[idx + 17]) : CompatibilityIndex.WILDCARD);
                int reqW = (col == 0) ? 0 : (localBoard[idx + 15] != -1 ? PieceUtils.getEast(localBoard[idx + 15]) : CompatibilityIndex.WILDCARD);
                int reqS = (row == 14) ? 0 : (localBoard[idx + 32] != -1 ? PieceUtils.getNorth(localBoard[idx + 32]) : CompatibilityIndex.WILDCARD);

                java.util.BitSet southCandidates = compatIndex.candidatesFor(reqN, reqE, reqS, reqW);
                compatIndex.andNotUsed(southCandidates, localUsed);

                if (southCandidates.isEmpty()) return false;

                // BITMASK SECONDARY LOOKAHEAD (Solves the flat-edge problem)
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

            // --- Check east neighbor ---
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
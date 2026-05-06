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
    // THE 3-PHASE CLAUDE ARCHITECTURE SETTINGS
    // ==========================================================
    public static final int SEED_DEPTH = 40;        // CPU bygger frø op til 40 brikker
    public static final int LNS_THRESHOLD = 200;    // Kirurgen aktiveres ved 200 brikker
    private final ConcurrentLinkedQueue<int[]> seedPool = new ConcurrentLinkedQueue<>();

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
    private final AtomicInteger currentBatchSize = new AtomicInteger(0);

    // Board State
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

    private final boolean lockCenter;
    private final int targetPiece;
    private int centerPhysicalIdx = -1;
    private final String saveProfile;
    private final BuildStrategy currentStrategy;
    private final Object displayLock = new Object();

    // Metrics
    private final AtomicLong globalCpuTrialCount = new AtomicLong(0);
    private final AtomicLong globalGpuTrialCount = new AtomicLong(0);
    private long lastThroughputReportTime = System.currentTimeMillis();
    private long lastRepairPrintTime = 0;
    private long totalRepairVariationsTested = 0;
    private long repairLoopsCounter = 0;
    private volatile boolean isGpuBusy = false;

    private volatile int stagnationLimitMinutes = 20;
    private volatile boolean manualOverrideRequested = false;
    private volatile int manualBaseCampTarget = 0;
    private volatile double extinctionThreshold = 0.98; // GUI Bridge

    public MasterSolverPBP(PieceInventory inventory, int trueCenterPiece, boolean useGpu, BuildStrategy strategy, boolean lockCenter) {
        Arrays.fill(flatBoard, -1);
        Arrays.fill(bestBoard, -1);
        Arrays.fill(flatResumeBoard, -1);

        this.inventory = inventory;
        this.targetPiece = trueCenterPiece;
        this.currentStrategy = strategy;
        this.lockCenter = lockCenter;
        this.saveProfile = strategy.name() + (lockCenter ? "_LOCKED" : "_UNLOCKED");
        this.useGpu = useGpu;

        this.numCores = Runtime.getRuntime().availableProcessors();
        this.executor = Executors.newFixedThreadPool(numCores);
        System.out.println(timestamp() + ">>> Multithreading active with " + numCores + " cores.");

        for (int i = 0; i < 1024; i++) {
            if (inventory.allOrientations[i] == targetPiece) {
                this.centerPhysicalIdx = inventory.physicalMapping[i];
                break;
            }
        }

        for (int i = 0; i < 256; i++) buildOrder[i] = i;

        this.compatIndex = new CompatibilityIndex(inventory.allOrientations, inventory.physicalMapping);
        this.surgeon = new SurgeonHeuristics(lockCenter, 0.70);

        // Load Checkpoint Robustly
        int[][] loaded = CheckpointManager.loadSmartCheckpoint(saveProfile);
        if (loaded != null) {
            int loadedCount = 0;
            for (int r = 0; r < 16; r++) {
                if (loaded[r] == null) continue;
                for (int c = 0; c < 16; c++) {
                    int p = loaded[r][c];
                    if (p != -1 && p != 0) {
                        bestBoard[r * 16 + c] = p;
                        loadedCount++;
                    }
                }
            }
            if (loadedCount > 0) {
                this.absoluteHighScore = loadedCount;
                this.deepestStep = loadedCount;
                if (lockCenter) bestBoard[135] = targetPiece;
                System.arraycopy(bestBoard, 0, flatBoard, 0, 256);
                System.arraycopy(bestBoard, 0, flatResumeBoard, 0, 256);
                updateDisplay(absoluteHighScore, buildDisplayBoard(bestBoard));
                System.out.println(timestamp() + ">>> Loaded checkpoint: " + absoluteHighScore + " pieces.");
            }
        }
    }

    // GUI BRIDGES
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

    private String timestamp() {
        return "[" + LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss")) + "] ";
    }

    @Override
    public void run() {
        if (this.useGpu) {
            this.gpuEngine = new GpuEngine(inventory, lockCenter);
        }

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\n" + timestamp() + ">>> Shutdown hook: Saving final checkpoint...");
            synchronized (displayLock) {
                CheckpointManager.saveRecordCheckpoint(buildDisplayBoard(bestBoard), absoluteHighScore, saveProfile);
            }
        }));

        Thread reporterThread = new Thread(() -> {
            while (true) {
                try { Thread.sleep(5000); } catch (InterruptedException ignored) {}
                reportSpeed();
            }
        });
        reporterThread.setDaemon(true);
        reporterThread.start();

        System.out.println(timestamp() + "Starting 3-Phase Pipeline Orchestrator...");
        long lastPeriodicSave = System.currentTimeMillis();

        while (true) {
            try { // <--- NYT: Grib alle fejl!
                if (manualOverrideRequested) {
                    manualOverrideRequested = false;
                    absoluteHighScore = manualBaseCampTarget;
                    retreat(manualBaseCampTarget, timestamp() + ">>> User Override...");
                    try { Thread.sleep(1000); } catch (InterruptedException ignored) {}
                }

                // THE HOLY GRAIL PIPELINE ROUTING
                if (gpuEngine != null && deepestStep >= LNS_THRESHOLD) {
                    runPhase3_GpuSurgeon();
                }
                else if (gpuEngine != null && seedPool.size() >= targetBatchSize) {
                    runPhase2_GpuDeepDfs();
                }
                else {
                    runPhase1_CpuSeedGen();
                }

                if (System.currentTimeMillis() - lastPeriodicSave > 300_000) {
                    synchronized (displayLock) {
                        CheckpointManager.saveRecordCheckpoint(buildDisplayBoard(bestBoard), absoluteHighScore, saveProfile);
                    }
                    lastPeriodicSave = System.currentTimeMillis();
                }
            } catch (Exception e) {
                // NYT: Hvis noget crasher, får vi det at vide i loggen!
                System.out.println("\n" + timestamp() + ">>> [FATAL ERROR] PIPELINE CRASHED: ");
                e.printStackTrace();
                try { Thread.sleep(5000); } catch (InterruptedException ignored) {}
            }
        }
    }

    // ==========================================================
    // PHASE 1: CPU SEED GENERATOR
    // ==========================================================
    private void runPhase1_CpuSeedGen() {
        // Only recompute lockedPieces / resumeBoard if deepestStep actually changed
        int lockedPieces = Math.max(0, absoluteHighScore - 40); // use absoluteHighScore, not deepestStep
        Arrays.fill(flatResumeBoard, -1);
        Arrays.fill(usedPhysicalPieces, false);

        if (lockedPieces > 0) {
            for (int step = 0; step < lockedPieces; step++) {
                int idx = buildOrder[step];
                if (lockCenter && idx == 135) continue;
                flatResumeBoard[idx] = bestBoard[idx];
            }
        }

        currentBatchSize.set(seedPool.size());

        try {
            int activeBatch = (userBatchSizeOverride > 0) ? userBatchSizeOverride : targetBatchSize;
            List<CpuSearchWorker> workers = new ArrayList<>();
            for (int i = 0; i < numCores; i++) workers.add(new CpuSearchWorker(activeBatch));

            executor.invokeAll(workers);
            // Hvis puljen ikke blev fuld, betyder det at CPU'en udtømte rummet.
            if (seedPool.size() < activeBatch && deepestStep > 0) {
                deepestStep = Math.max(0, deepestStep - 10); // Backtrack
            }
        } catch (InterruptedException e) {
            return;
        }
    }

    // ==========================================================
    // PHASE 2: GPU DEEP DFS EXPLORER
    // ==========================================================
    private void runPhase2_GpuDeepDfs() {
        List<int[]> seeds = new ArrayList<>();
        for (int i = 0; i < targetBatchSize; i++) {
            int[] s = seedPool.poll();
            if (s != null) seeds.add(s);
        }
        if (seeds.isEmpty()) return;

        int scoreBefore = deepestStep;
        int[] bestBoardOut = new int[256];

        // ==========================================
        // NYT: Fortæl systemet at GPU'en dykker!
        // ==========================================
        isGpuBusy = true;
        System.out.println(timestamp() + ">>> GPU Phase 2 diving deep with 50.000 seeds. This may take 10-40 seconds...");

        GpuEngine.GpuResult result = gpuEngine.runDeepDfs(
                seeds, SEED_DEPTH, absoluteHighScore, bestBoardOut, buildOrder);
        globalGpuTrialCount.addAndGet(result.stepsTaken);
        isGpuBusy = false;
        // ==========================================
        System.out.printf("%s>>> GPU Phase 2 complete. Steps taken: %,d across %d seeds (avg %,.0f/seed)%n",
                timestamp(), result.stepsTaken, seeds.size(),
                (double) result.stepsTaken / seeds.size());

        if (result.solved) {
            handleVictory(bestBoardOut);
        }

        if (result.newHighScore > scoreBefore) {
            deepestStep = result.newHighScore;
            System.arraycopy(bestBoardOut, 0, bestBoard, 0, 256);
            updateDisplay(deepestStep, buildDisplayBoard(bestBoard));

            if (deepestStep > absoluteHighScore) {
                absoluteHighScore = deepestStep;
                RecordManager.saveRecord(buildDisplayBoard(bestBoard), absoluteHighScore, saveProfile);
                System.out.println("\n" + timestamp() + ">>> PHASE 2 (GPU) BROKE RECORD! NEW HIGH SCORE: " + absoluteHighScore + " <<<");
            } else {
                System.out.println(timestamp() + ">>> PHASE 2 (GPU) Pushed seed to: " + deepestStep + " pieces.");
            }
        }
    }

    // ==========================================================
    // PHASE 3: GPU SURGEON (LNS)
    // ==========================================================
    private void runPhase3_GpuSurgeon() {
        int numClones = 50000;
        int holesToPunch = 40;

        currentRepairIteration++;
        repairLoopsCounter++;
        totalRepairVariationsTested += numClones;

        List<int[]> swissCheeseBoards = surgeon.punchHoles(bestBoard, numClones, holesToPunch, tabuTenure, currentRepairIteration, deepestStep, buildOrder);

        long now = System.currentTimeMillis();
        if (now - lastRepairPrintTime > 2000) {
            System.out.println(timestamp() + ">>> [PHASE 3: LNS SURGEON] Working on High-Depth Board. Testing " + holesToPunch + " holes...");
            lastRepairPrintTime = now;
        }

        int scoreBefore = deepestStep;
        int[] bestBoardOut = new int[256];
        GpuEngine.GpuResult result = gpuEngine.runRepairMode(swissCheeseBoards, deepestStep, bestBoardOut);

        globalGpuTrialCount.addAndGet(result.stepsTaken);

        if (result.solved) handleVictory(bestBoardOut);

        if (result.newHighScore > scoreBefore) {
            deepestStep = result.newHighScore;
            updateTabuList(bestBoardOut);
            System.arraycopy(bestBoardOut, 0, bestBoard, 0, 256);
            updateDisplay(deepestStep, buildDisplayBoard(bestBoard));

            if (deepestStep > absoluteHighScore) {
                absoluteHighScore = deepestStep;
                RecordManager.saveRecord(buildDisplayBoard(bestBoard), absoluteHighScore, saveProfile);
                System.out.println("\n" + timestamp() + ">>> PHASE 3 (SURGEON) BROKE RECORD! NEW HIGH SCORE: " + absoluteHighScore + " <<<");
            }
            consecutiveExtinctions = 0;
        } else {
            consecutiveExtinctions++;
            if (consecutiveExtinctions >= 10) {
                triggerBranchScrap();
            }
        }
    }

    private void updateTabuList(int[] newBoard) {
        for (int i = 0; i < 256; i++) {
            if (newBoard[i] != bestBoard[i] && newBoard[i] != -1) {
                tabuTenure[i] = currentRepairIteration + 25;
            }
        }
    }

    private void triggerBranchScrap() {
        System.out.println("\n" + timestamp() + ">>> [!!!] DEAD END AT PHASE 3 [!!!]");
        System.out.println(timestamp() + ">>> Scrapping this branch, pulling new Phase 2 seeds from absoluteHighScore base...");

        // Reset deepestStep to seed depth to re-enter Phase 2,
        // but keep bestBoard and absoluteHighScore intact!
        deepestStep = SEED_DEPTH;
        consecutiveExtinctions = 0;

        // Re-anchor the resume board to the best known board so Phase 1
        // generates seeds that are diverse variations of the best known state,
        // not from scratch.
        int lockedPieces = absoluteHighScore - 40;
        if (lockedPieces > 0) {
            Arrays.fill(flatResumeBoard, -1);
            for (int step = 0; step < lockedPieces; step++) {
                int idx = buildOrder[step];
                if (lockCenter && idx == 135) continue;
                flatResumeBoard[idx] = bestBoard[idx];
            }
        }
    }

    private void handleVictory(int[] winningBoard) {
        System.out.println("\n" + timestamp() + ">>> ETERNITY II SOLVED BY GPU PIPELINE!!! <<<");
        updateDisplay(256, buildDisplayBoard(winningBoard));
        RecordManager.saveRecord(buildDisplayBoard(winningBoard), 256, saveProfile);
        System.exit(0);
    }

    // ==========================================================
    // HELPER METHODS
    // ==========================================================
    private void reportSpeed() {
        if (isGpuBusy) {
            System.out.println(timestamp() + "[STATUS] GPU Phase 2 er under vandet! Kværner millioner af træk i dybden...");
            return;
        }

        long now = System.currentTimeMillis();
        long elapsed = now - lastThroughputReportTime;

        if (elapsed < 2000) return;

        long cpuTrials = globalCpuTrialCount.getAndSet(0);
        long gpuTrials = globalGpuTrialCount.getAndSet(0);

        double cpuTps = cpuTrials / (elapsed / 1000.0);
        double gpuTps = gpuTrials / (elapsed / 1000.0);

        System.out.printf("%s[SPEED] CPU Phase 1: %,.0f/s  |  GPU Phase 2/3: %,.0f/s\n",
                timestamp(), cpuTps, gpuTps);

        lastThroughputReportTime = now;
    }

    private void retreat(int targetStep, String logMessage) {
        deepestStep = Math.max(0, targetStep);
        for (int s = deepestStep; s < 256; s++) bestBoard[buildOrder[s]] = -1;
        if (lockCenter) bestBoard[135] = targetPiece;
        updateDisplay(countPieces(bestBoard), buildDisplayBoard(bestBoard));
        if (logMessage != null) System.out.println(logMessage);
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
            displayBoard[i / 16][i % 16] = sourceArray[i];
        }
        return displayBoard;
    }

    private void updateDisplay(int score, int[][] displayBoard) {
        Main.updateDisplay(score, this.absoluteHighScore, displayBoard);
    }

    public enum BuildStrategy { TYPEWRITER, SPIRAL }

    // ==========================================================
    // CPU SEARCH WORKER (PHASE 1 SEED GENERATOR)
    // ==========================================================
    private class CpuSearchWorker implements Callable<Boolean> {
        private final int[] localBoard = new int[256];
        private final int[] localResumeBoard = new int[256];
        private final boolean[] localUsed = new boolean[256];
        private final Random rnd = new Random();
        private final int activeBatch;
        private long localTrialCount = 0; // NY: Tilbagefører Trial-speedometeret!

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
        }

        @Override
        public Boolean call() {
            boolean result = solve(0);
            // Tøm resterne til speedometeret, inden tråden dør!
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

            if (step == SEED_DEPTH) {
                int[] seed = new int[256];
                System.arraycopy(localBoard, 0, seed, 0, 256);
                seedPool.add(seed);

                currentBatchSize.incrementAndGet();

                return false; // Returner FALSE for at tvinge CPU'en til at finde et NYT frø!
            }

            int boardIdx = buildOrder[step];
            if (localBoard[boardIdx] != -1 || (lockCenter && boardIdx == 135)) {
                return solve(step + 1);
            }

            int row = boardIdx / 16;
            int col = boardIdx % 16;
            int northReq = (row == 0) ? 0 : (localBoard[boardIdx - 16] != -1 ? PieceUtils.getSouth(localBoard[boardIdx - 16]) : CompatibilityIndex.WILDCARD);
            int southReq = (row == 15) ? 0 : (localBoard[boardIdx + 16] != -1 ? PieceUtils.getNorth(localBoard[boardIdx + 16]) : CompatibilityIndex.WILDCARD);
            int westReq  = (col == 0) ? 0 : (localBoard[boardIdx - 1] != -1 ? PieceUtils.getEast(localBoard[boardIdx - 1])  : CompatibilityIndex.WILDCARD);
            int eastReq  = (col == 15) ? 0 : (localBoard[boardIdx + 1] != -1 ? PieceUtils.getWest(localBoard[boardIdx + 1]) : CompatibilityIndex.WILDCARD);

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
                if (seedPool.size() >= activeBatch) return false;

                int orientationIdx = orientIdxs[(i + offset) % candidateCount];
                int p = inventory.allOrientations[orientationIdx];
                int physicalIdx = inventory.physicalMapping[orientationIdx];

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
                int reqW = (col == 0)  ? 0 : (localBoard[idx + 15] != -1 ? PieceUtils.getEast(localBoard[idx + 15]) : CompatibilityIndex.WILDCARD);
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
                int reqN = (row == 0)  ? 0 : (localBoard[idx - 15] != -1 ? PieceUtils.getSouth(localBoard[idx - 15]) : CompatibilityIndex.WILDCARD);
                int reqE = (col == 14) ? 0 : (localBoard[idx + 2]  != -1 ? PieceUtils.getWest(localBoard[idx + 2])   : CompatibilityIndex.WILDCARD);
                int reqS = (row == 15) ? 0 : (localBoard[idx + 17] != -1 ? PieceUtils.getNorth(localBoard[idx + 17]) : CompatibilityIndex.WILDCARD);

                if (!compatIndex.hasAnyCandidate(reqN, reqE, reqS, reqW, localUsed)) return false;
            }
            return true;
        }
    }
}
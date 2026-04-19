package dk.puzzle;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;

public class TypewriterSolver implements Runnable {

    private final PieceInventory inventory;
    private final int[] bestBoard = new int[256];
    private final int targetPiece;
    private final boolean lockCenter;
    private final String saveProfile;

    private volatile int absoluteHighScore = 0;
    private final Object displayLock = new Object();

    private final int numCores;
    private final ExecutorService executor;

    // --- HYPER-OPTIMERING: Pre-kalkulerede Arrays ---
    private final int[] cornerPool;
    private final int[] edgePool;
    private final int[] interiorPool;

    private final boolean[] isTopEdge = new boolean[256];
    private final boolean[] isBottomEdge = new boolean[256];
    private final boolean[] isLeftEdge = new boolean[256];
    private final boolean[] isRightEdge = new boolean[256];
    private final int[] categoryMap = new int[256]; // 2=Corner, 1=Edge, 0=Interior

    public TypewriterSolver(PieceInventory inventory, int trueCenterPiece, boolean lockCenter) {
        Arrays.fill(bestBoard, -1);
        this.inventory = inventory;
        this.targetPiece = trueCenterPiece;
        this.lockCenter = lockCenter;

        this.numCores = Runtime.getRuntime().availableProcessors();
        this.executor = Executors.newFixedThreadPool(numCores);
        this.saveProfile = "TYPEWRITER" + (lockCenter ? "_LOCKED" : "_UNLOCKED");

        this.cornerPool = inventory.corners.stream().mapToInt(i -> i).toArray();
        this.edgePool = inventory.edges.stream().mapToInt(i -> i).toArray();
        this.interiorPool = inventory.interior.stream().mapToInt(i -> i).toArray();

        // 2. Pre-kalkuler hele brættets geometri (CPU'en skal ikke regne det ud undervejs!)
        for (int i = 0; i < 256; i++) {
            int r = i / 16;
            int c = i % 16;
            isTopEdge[i] = (r == 0);
            isBottomEdge[i] = (r == 15);
            isLeftEdge[i] = (c == 0);
            isRightEdge[i] = (c == 15);

            int b_req = 0;
            if (isTopEdge[i] || isBottomEdge[i]) b_req++;
            if (isLeftEdge[i] || isRightEdge[i]) b_req++;
            categoryMap[i] = b_req;
        }

        int[][] loaded = CheckpointManager.load(saveProfile);
        if (loaded != null) {
            int highestStepLoaded = -1;
            for (int step = 0; step < 256; step++) {
                int r = step / 16;
                int c = step % 16;
                if (loaded[r] != null) {
                    int p = loaded[r][c];
                    if (p != -1 && p != 0) {
                        bestBoard[step] = p;
                        highestStepLoaded = Math.max(highestStepLoaded, step);
                    }
                }
            }
            if (highestStepLoaded > 0) {
                this.absoluteHighScore = highestStepLoaded;
                if (lockCenter) bestBoard[135] = targetPiece;
                System.out.println(">>> Indlæste Skrivemaskine Checkpoint! High Score starter ved: " + absoluteHighScore + " brikker.");

                Main.updateDisplay(absoluteHighScore, buildDisplayBoard(bestBoard));
            }
        }
    }

    private String timestamp() {
        return "[" + LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss")) + "] ";
    }

    private class TypewriterWorker implements Callable<Boolean> {
        private final int[] localBoard = new int[256];
        private final boolean[] localUsed = new boolean[256];
        private final Random rnd = new Random();

        public TypewriterWorker() {
            Arrays.fill(localBoard, -1);
            if (lockCenter) {
                localBoard[135] = targetPiece;
                for (int i = 0; i < 1024; i++) {
                    if (inventory.allOrientations[i] == targetPiece) {
                        localUsed[inventory.physicalMapping[i]] = true;
                        break;
                    }
                }
            }
        }

        @Override
        public Boolean call() {
            return solve(0);
        }

        private boolean solve(int step) {
            if (step == 256) return true;

            if (lockCenter && step == 135) return solve(step + 1);

            if (step > absoluteHighScore) {
                synchronized(displayLock) {
                    if (step > absoluteHighScore) {
                        absoluteHighScore = step;
                        System.arraycopy(localBoard, 0, bestBoard, 0, 256);
                        int[][] displayBoard = buildDisplayBoard(bestBoard);
                        Main.updateDisplay(step, displayBoard);
                        RecordManager.saveRecord(displayBoard, absoluteHighScore, saveProfile);
                        CheckpointManager.save(displayBoard, saveProfile);

                        System.out.println(timestamp() + ">>> NEW TYPEWRITER HIGHSCORE: " + absoluteHighScore + " PIECES! <<<");
                    }
                }
            }

            int cat = categoryMap[step];
            int[] activePool = (cat == 2) ? cornerPool : (cat == 1) ? edgePool : interiorPool;
            int poolSize = activePool.length;

            if (poolSize == 0) return false;

            int n_req = isTopEdge[step] ? 0 : ((localBoard[step - 16] >> 8) & 0xFF); // Hent Syd-farven fra brikken over
            int w_req = isLeftEdge[step] ? 0 : ((localBoard[step - 1] >> 16) & 0xFF); // Hent Øst-farven fra brikken til venstre

            boolean checkSouthEdge = isBottomEdge[step];
            boolean checkEastEdge = isRightEdge[step];

            int offset = rnd.nextInt(poolSize);

            for (int i = 0; i < poolSize; i++) {
                int orientationIdx = activePool[(i + offset) % poolSize];
                int physicalIdx = inventory.physicalMapping[orientationIdx];

                if (localUsed[physicalIdx]) continue;

                int p = inventory.allOrientations[orientationIdx];

                int p_north = (p >> 24) & 0xFF;
                int p_east  = (p >> 16) & 0xFF;
                int p_south = (p >> 8)  & 0xFF;
                int p_west  = p & 0xFF;

                if (p_north != n_req) continue;
                if (p_west != w_req) continue;
                if (checkSouthEdge && p_south != 0) continue;
                if (checkEastEdge && p_east != 0) continue;

                localBoard[step] = p;
                localUsed[physicalIdx] = true;

                if (solve(step + 1)) return true;

                // Backtrack
                localBoard[step] = -1;
                localUsed[physicalIdx] = false;
            }
            return false;
        }
    }

    @Override
    public void run() {
        System.out.println(timestamp() + "Starter OPTIMERET Skrivemaskine Solver (" + numCores + " CPU Kerner)...");

        List<TypewriterWorker> workers = Arrays.asList(new TypewriterWorker[numCores]);
        for (int i = 0; i < numCores; i++) workers.set(i, new TypewriterWorker());

        try {
            List<Future<Boolean>> results = executor.invokeAll(workers);
            for (Future<Boolean> f : results) {
                if (f.get()) {
                    System.out.println(timestamp() + ">>> PUSLESPIL LØST! <<<");
                    return;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println(timestamp() + "Søgning udtømt.");
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
}
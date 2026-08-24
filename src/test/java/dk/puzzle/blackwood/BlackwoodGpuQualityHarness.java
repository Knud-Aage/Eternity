package dk.puzzle.blackwood;

import dk.puzzle.core.Eternity;
import dk.puzzle.gpu.BlackwoodGpuEngine;
import dk.puzzle.model.PieceInventory;
import dk.puzzle.tools.HoleSolver;

import java.util.*;

/**
 * Follow-up to {@link BlackwoodGpuBreadthDepthHarness}: that harness showed maxDepth is flat
 * (243-244) across thread counts at equal wall-clock, with meanDepth actually favouring fewer
 * threads. But raw search depth isn't the tracked metric -- post-HoleSolver conflict count is.
 * This harness runs the SAME harvest-and-score pipeline production uses (evaluateAndMaybeSave's
 * decode/solveConflicts/countConflicts, via {@link BlackwoodGpuEngine#readThreadBestBoards}) at a
 * few thread counts, equal wall-clock each, and compares actual conflict-count distributions
 * rather than depth. Never saves anything -- pure measurement, same convention as the breadth
 * harness's saveThreshold=999.
 */
public class BlackwoodGpuQualityHarness {

    private static final String PIECES_PATH = "src/main/resources/JBlackwood_Pieces.txt";
    private static final long STEP_BUDGET = 20_000L;
    private static final long RUN_MILLIS = 60_000L;
    private static final long HARVEST_EVERY_MILLIS = 10_000L;
    private static final int HARVEST_SAMPLE = 8;
    private static final int HARVEST_MIN_DEPTH = 235; // slightly relaxed vs production's 240 -- fewer
    // threads means fewer chances at any single snapshot, want enough candidates to compare.
    private static final int SCORING_TRIALS = 5000; // matches production's own scoring budget
    private static final int[] THREAD_COUNTS = {1024, 256, 64};

    public static void main(String[] args) throws Exception {
        System.out.println("=== Blackwood GPU quality-vs-thread-count sweep ===");
        System.out.printf("%,d ms per config, harvest every %,d ms, sample %d deepest (min depth %d), %d HoleSolver trials%n%n",
                RUN_MILLIS, HARVEST_EVERY_MILLIS, HARVEST_SAMPLE, HARVEST_MIN_DEPTH, SCORING_TRIALS);

        BlackwoodSolver solver = new BlackwoodSolver(999, null, 1, PIECES_PATH); // never saves
        solver.prepare();
        BwGpuTables.GpuTableSet tables = BwGpuTables.build(solver);

        List<BwPiece> pieces = BwUtil.getPieces(PIECES_PATH);
        BwPiece[] pieceByNumber = new BwPiece[257];
        for (BwPiece p : pieces) pieceByNumber[p.pieceNumber()] = p;
        PieceInventory inventory = new PieceInventory(Eternity.loadPieces());

        BlackwoodGpuEngine engine = new BlackwoodGpuEngine();

        for (int threads : THREAD_COUNTS) {
            engine.uploadTables(tables);
            engine.uploadSeeds(List.of(), new int[0], 0, 0);
            engine.resetEpoch();

            Set<String> seen = new HashSet<>();
            List<Integer> conflictsFound = new ArrayList<>();
            int highScore = 0;
            int launches = 0;
            long nextHarvest = HARVEST_EVERY_MILLIS;
            long t0 = System.nanoTime();

            while ((System.nanoTime() - t0) / 1_000_000L < RUN_MILLIS) {
                long seedBase = System.nanoTime() ^ ((long) launches * 0x9E3779B97F4A7C15L);
                int[] bestBoardOut = new int[256];
                BlackwoodGpuEngine.GpuResult r =
                        engine.runBlackwoodDfs(seedBase, STEP_BUDGET, threads, highScore, bestBoardOut);
                highScore = Math.max(highScore, r.newHighScore());
                launches++;

                long elapsed = (System.nanoTime() - t0) / 1_000_000L;
                if (elapsed >= nextHarvest) {
                    nextHarvest += HARVEST_EVERY_MILLIS;
                    harvestAndScore(engine, r.threadDepths(), pieceByNumber, inventory, seen, conflictsFound);
                }
            }
            // Final harvest so the last ~10s window isn't wasted.
            harvestAndScore(engine, engine.runBlackwoodDfs(
                            System.nanoTime(), STEP_BUDGET, threads, highScore, new int[256]).threadDepths(),
                    pieceByNumber, inventory, seen, conflictsFound);

            report(threads, launches, conflictsFound);
        }
    }

    private static void harvestAndScore(BlackwoodGpuEngine engine, int[] threadDepths,
                                         BwPiece[] pieceByNumber, PieceInventory inventory,
                                         Set<String> seen, List<Integer> conflictsFound) {
        int numThreads = threadDepths.length;
        int[] allBoards = engine.readThreadBestBoards(numThreads);

        Integer[] order = new Integer[numThreads];
        for (int i = 0; i < numThreads; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Integer.compare(threadDepths[b], threadDepths[a]));

        int scored = 0;
        for (int idx = 0; idx < numThreads && scored < HARVEST_SAMPLE; idx++) {
            int t = order[idx];
            if (threadDepths[t] < HARVEST_MIN_DEPTH) break; // sorted, nothing deeper remains
            int[] board = Arrays.copyOfRange(allBoards, t * 256, (t + 1) * 256);
            String fp = fingerprintOf(board);
            if (!seen.add(fp)) continue;

            try {
                BwRotatedPiece[] rotatedBoard = new BwRotatedPiece[256];
                for (int i = 0; i < 256; i++) {
                    rotatedBoard[i] = (board[i] == -1) ? BwRotatedPiece.EMPTY : BwGpuTables.unpack(board[i]);
                }
                String boardString = BwUtil.buildBoardString(rotatedBoard, pieceByNumber);
                String link = boardString.substring(boardString.lastIndexOf("https://"));
                int[] decoded = HoleSolver.decodeBoardAuto(link, inventory, false);
                HoleSolver.ConflictSolveResult result = HoleSolver.solveConflicts(decoded, inventory, false, SCORING_TRIALS);
                int conflicts = BlackwoodGpuRunner.countConflicts(result.bestBoard());
                conflictsFound.add(conflicts);
                scored++;
            } catch (Exception e) {
                System.err.println("  scoring failed for a candidate: " + e.getMessage());
            }
        }
    }

    private static String fingerprintOf(int[] board) {
        StringBuilder sb = new StringBuilder(1024);
        for (int i = 0; i < 256; i++) {
            if (board[i] == -1) { sb.append("..,"); continue; }
            BwRotatedPiece p = BwGpuTables.unpack(board[i]);
            sb.append(p.pieceNumber()).append(':').append(p.rotations()).append(',');
        }
        return sb.toString();
    }

    private static void report(int threads, int launches, List<Integer> conflicts) {
        System.out.printf("--- %d threads, %d launches, %d boards scored ---%n", threads, launches, conflicts.size());
        if (conflicts.isEmpty()) {
            System.out.println("  (nothing reached the harvest depth floor)");
            return;
        }
        List<Integer> sorted = new ArrayList<>(conflicts);
        Collections.sort(sorted);
        double mean = conflicts.stream().mapToInt(Integer::intValue).average().orElse(0);
        System.out.printf("  best: %d   median: %d   mean: %.1f   worst: %d%n",
                sorted.get(0), sorted.get(sorted.size() / 2), mean, sorted.get(sorted.size() - 1));
        Map<Integer, Integer> hist = new TreeMap<>();
        for (int c : conflicts) hist.merge(c, 1, Integer::sum);
        hist.forEach((k, v) -> System.out.printf("    %2d conflicts: %d%n", k, v));
        System.out.println();
    }
}

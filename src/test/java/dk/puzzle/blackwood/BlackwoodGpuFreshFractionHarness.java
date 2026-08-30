package dk.puzzle.blackwood;

import dk.puzzle.core.Eternity;
import dk.puzzle.gpu.BlackwoodGpuEngine;
import dk.puzzle.model.PieceInventory;
import dk.puzzle.tools.HoleSolver;
import dk.puzzle.util.PieceUtils;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * A/Bs {@code BlackwoodGpuRunner.FRESH_FRACTION_PERCENT}, which was set to 25% as a guess.
 *
 * <p><b>Why this measures what it measures.</b> The two obvious metrics are both misleading here:</p>
 * <ul>
 *   <li><b>Max depth</b> is useless with seeding on -- every seeded arm pins at the deepest seed's
 *       own depth (252), so all arms tie regardless of how much real work they did.</li>
 *   <li><b>Conflicts of the single best board</b> is worse than useless: {@code runBlackwoodDfs}
 *       selects that board by DEPTH, so in seeded arms it is typically a seed replayed back
 *       unchanged. That would score every seeded arm at the seed's own conflict count and make a
 *       pure-exploration arm look bad, measuring the seed pool rather than the configuration.</li>
 * </ul>
 *
 * <p>So this reads back the whole population's best boards and asks the question the fresh fraction
 * actually exists to answer: how many DISTINCT boards did this configuration produce that are NOT
 * simply seeds handed back, and how good are they? A configuration that only ever regurgitates its
 * seeds has produced nothing, however deep its numbers look.</p>
 *
 * <p>Conflict scoring runs on a bounded sample of the distinct novel boards, since HoleSolver
 * completion costs about a second each.</p>
 *
 * <p>'Harness' suffix keeps Surefire from collecting it; needs real CUDA hardware.</p>
 */
public class BlackwoodGpuFreshFractionHarness {

    private static final String PIECES_PATH = "src/main/resources/JBlackwood_Pieces.txt";
    private static final int NUM_THREADS = 1024;
    private static final long STEP_BUDGET = 50_000L;
    private static final int MIN_SEED_DEPTH = 245;
    private static final int MAX_SEEDS = 256;
    private static final int MAX_RETREAT = 40;
    private static final int SCORING_TRIALS = 5000;

    private static final long WALL_MILLIS = 180_000L;
    private static final int[] FRESH_FRACTIONS = {0, 25, 50, 100};
    /** Distinct novel boards to actually score per arm (HoleSolver is ~1s each). */
    private static final int SCORE_SAMPLE = 25;
    /** Only boards at least this deep are worth scoring -- shallower ones cannot compete. */
    private static final int SCORE_MIN_DEPTH = 240;

    public static void main(String[] args) throws Exception {
        BlackwoodSolver solver = new BlackwoodSolver(999, null, 1, PIECES_PATH); // never saves
        solver.prepare();
        BwGpuTables.GpuTableSet tables = BwGpuTables.build(solver);
        PieceInventory inventory = new PieceInventory(Eternity.loadPieces());

        List<BwPiece> pieceList = BwUtil.getPieces(PIECES_PATH);
        BwPiece[] pieceByNumber = new BwPiece[257];
        for (BwPiece p : pieceList) pieceByNumber[p.pieceNumber()] = p;

        Path home = Path.of(System.getProperty("user.home"));
        List<Path> dirs = List.of(
                home.resolve("EternitySolutions_GPU"),
                home.resolve("EternitySolutions_CSharpCPU"),
                home.resolve("EternitySolutions_JavaCPU"));

        List<BwSeedLoader.Seed> candidates = BwSeedLoader.load(dirs, MIN_SEED_DEPTH, 120, tables.stepBoardIdx());
        List<BwSeedLoader.Seed> seeds = BwSeedLoader.rankByConflicts(candidates, inventory, SCORING_TRIALS, MAX_SEEDS);
        if (seeds.isEmpty()) {
            System.out.println("No seeds available -- this A/B is meaningless without them.");
            return;
        }

        List<int[]> encoded = new ArrayList<>();
        int[] depths = new int[seeds.size()];
        Set<String> seedFingerprints = new HashSet<>();
        for (int i = 0; i < seeds.size(); i++) {
            encoded.add(seeds.get(i).stepEncoded());
            depths[i] = seeds.get(i).depth();
            seedFingerprints.add(fingerprintOfSeed(seeds.get(i), tables.stepBoardIdx()));
        }

        System.out.println("=== Fresh-fraction A/B ===");
        System.out.printf("%d seeds, conflicts %d..%d, best %d conflicts. %d threads, %ds per arm.%n%n",
                seeds.size(), seeds.get(0).conflicts(), seeds.get(seeds.size() - 1).conflicts(),
                seeds.get(0).conflicts(), NUM_THREADS, WALL_MILLIS / 1000);
        System.out.printf("%7s %9s %8s %9s %9s %9s %9s %9s%n",
                "fresh%", "launches", "maxDep", "meanDep", "distinct", "novel", "bestConf", "medConf");

        BlackwoodGpuEngine engine = new BlackwoodGpuEngine();
        for (int fresh : FRESH_FRACTIONS) {
            runArm(engine, tables, inventory, pieceByNumber, encoded, depths, seedFingerprints, fresh);
        }

        System.out.println();
        System.out.println("distinct = distinct best-boards held by the population at the end");
        System.out.println("novel    = those distinct boards that are NOT a seed handed back unchanged");
        System.out.println("bestConf/medConf = HoleSolver conflicts over a sample of the novel boards");
        System.out.println("(seed pool's own best is printed above -- a novel board must beat it to matter)");
    }

    private static void runArm(BlackwoodGpuEngine engine, BwGpuTables.GpuTableSet tables,
                               PieceInventory inventory, BwPiece[] pieceByNumber,
                               List<int[]> encoded, int[] depths, Set<String> seedFingerprints, int freshPercent) {
        engine.uploadTables(tables);
        engine.uploadSeeds(encoded, depths, MAX_RETREAT, freshPercent);
        engine.resetEpoch();

        int highScore = 0, launches = 0;
        int[] threadDepths = null;
        long deadline = System.nanoTime() + WALL_MILLIS * 1_000_000L;
        while (System.nanoTime() < deadline) {
            BlackwoodGpuEngine.GpuResult r = engine.runBlackwoodDfs(
                    System.nanoTime() ^ ((long) launches * 0x9E3779B97F4A7C15L),
                    STEP_BUDGET, NUM_THREADS, highScore, new int[256]);
            highScore = Math.max(highScore, r.newHighScore());
            threadDepths = r.threadDepths();
            launches++;
        }

        double meanDepth = 0;
        int maxDepth = 0;
        for (int d : threadDepths) { meanDepth += d; maxDepth = Math.max(maxDepth, d); }
        meanDepth /= threadDepths.length;

        // Population read-back: dedupe, drop anything that is just a seed returned unchanged.
        int[] allBoards = engine.readThreadBestBoards(NUM_THREADS);
        Set<String> distinct = new HashSet<>();
        List<int[]> novelBoards = new ArrayList<>();
        for (int t = 0; t < NUM_THREADS; t++) {
            int[] board = Arrays.copyOfRange(allBoards, t * 256, (t + 1) * 256);
            String fp = fingerprintOfBoard(board);
            if (!distinct.add(fp)) continue;
            if (seedFingerprints.contains(fp)) continue;
            if (threadDepths[t] < SCORE_MIN_DEPTH) continue;
            novelBoards.add(board);
        }

        List<Integer> conflicts = new ArrayList<>();
        for (int i = 0; i < Math.min(SCORE_SAMPLE, novelBoards.size()); i++) {
            int c = scoreBoard(novelBoards.get(i), inventory, pieceByNumber);
            if (c >= 0) conflicts.add(c);
        }
        conflicts.sort(Integer::compareTo);

        String bestConf = conflicts.isEmpty() ? "-" : String.valueOf(conflicts.get(0));
        String medConf = conflicts.isEmpty() ? "-" : String.valueOf(conflicts.get(conflicts.size() / 2));
        System.out.printf("%7d %9d %8d %9.1f %9d %9d %9s %9s%n",
                freshPercent, launches, maxDepth, meanDepth, distinct.size(), novelBoards.size(), bestConf, medConf);
    }

    /** Completes a packed GPU board through HoleSolver and returns its conflict count, or -1. */
    private static int scoreBoard(int[] board, PieceInventory inventory, BwPiece[] pieceByNumber) {
        try {
            BwRotatedPiece[] rotated = new BwRotatedPiece[256];
            for (int i = 0; i < 256; i++) {
                rotated[i] = (board[i] == -1) ? BwRotatedPiece.EMPTY : BwGpuTables.unpack(board[i]);
            }
            String boardString = BwUtil.buildBoardString(rotated, pieceByNumber);
            String link = boardString.substring(boardString.lastIndexOf("https://"));
            int[] decoded = HoleSolver.decodeBoardAuto(link, inventory, false);
            HoleSolver.ConflictSolveResult result =
                    HoleSolver.solveConflicts(decoded, inventory, false, SCORING_TRIALS);
            int[] best = result.bestBoard();
            int conflicts = 0;
            for (int r = 0; r < 16; r++) {
                for (int c = 0; c < 16; c++) {
                    int i = r * 16 + c;
                    if (c < 15 && PieceUtils.getEast(best[i]) != PieceUtils.getWest(best[i + 1])) conflicts++;
                    if (r < 15 && PieceUtils.getSouth(best[i]) != PieceUtils.getNorth(best[i + 16])) conflicts++;
                }
            }
            return conflicts;
        } catch (Exception e) {
            return -1;
        }
    }

    /** Board identity as (pieceNumber,rotation) per cell -- comparable across seed and GPU forms. */
    private static String fingerprintOfBoard(int[] board) {
        StringBuilder sb = new StringBuilder(1024);
        for (int i = 0; i < 256; i++) {
            if (board[i] == -1) { sb.append("..,"); continue; }
            BwRotatedPiece p = BwGpuTables.unpack(board[i]);
            sb.append(p.pieceNumber()).append(':').append(p.rotations()).append(',');
        }
        return sb.toString();
    }

    private static String fingerprintOfSeed(BwSeedLoader.Seed seed, int[] stepBoardIdx) {
        int[] byBoardIdx = new int[256];
        Arrays.fill(byBoardIdx, -1);
        for (int step = 0; step < seed.depth(); step++) byBoardIdx[stepBoardIdx[step]] = seed.stepEncoded()[step];
        StringBuilder sb = new StringBuilder(1024);
        for (int i = 0; i < 256; i++) {
            if (byBoardIdx[i] < 0) { sb.append("..,"); continue; }
            sb.append(byBoardIdx[i] >> 2).append(':').append(byBoardIdx[i] & 3).append(',');
        }
        return sb.toString();
    }
}

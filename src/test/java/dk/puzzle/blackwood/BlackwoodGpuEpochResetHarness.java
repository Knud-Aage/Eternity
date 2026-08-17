package dk.puzzle.blackwood;

import dk.puzzle.gpu.BlackwoodGpuEngine;

/**
 * Measures what {@code BlackwoodGpuRunner.EPOCH_LAUNCHES} actually costs.
 *
 * <p>Production rebuilds the candidate tables every 60 launches, which forces
 * {@link BlackwoodGpuEngine#resetEpoch()} -- a resumed cursor into a replaced table would point at
 * the wrong candidates. That reset discards every thread's in-progress search and sends all of them
 * back to step 0.</p>
 *
 * <p>Why that might matter more than it looks: {@link BlackwoodGpuBreadthDepthHarness} showed the
 * kernel climbing from scratch to ~247 pieces in about 30 seconds, and at production's launch
 * cadence a 60-launch epoch is only ~1 minute long. If most of an epoch is spent just re-climbing
 * ground the previous epoch already covered, the search would never accumulate enough sustained
 * backtracking to push past the wall -- which would explain a GPU that reaches 247 in seconds but
 * took three days of running to reach 251.</p>
 *
 * <p>Both arms get identical wall-clock time and identical thread counts. The only difference is
 * whether the epoch reset fires.</p>
 */
public class BlackwoodGpuEpochResetHarness {

    private static final String PIECES_PATH = "src/main/resources/JBlackwood_Pieces.txt";
    private static final long STEP_BUDGET = 50_000L;
    private static final long WALL_MILLIS = 90_000L;
    private static final int EPOCH_LAUNCHES = 60; // production's value
    private static final int[] THREAD_COUNTS = {16384, 1024};

    public static void main(String[] args) throws Exception {
        System.out.println("=== Epoch-reset cost ===");
        System.out.printf("wall=%ds per arm, stepBudget=%,d, production epoch=%d launches%n%n",
                WALL_MILLIS / 1000, STEP_BUDGET, EPOCH_LAUNCHES);

        BlackwoodSolver solver = new BlackwoodSolver(999, null, 1, PIECES_PATH); // never saves
        solver.prepare();
        BwGpuTables.GpuTableSet tables = BwGpuTables.build(solver);
        BlackwoodGpuEngine engine = new BlackwoodGpuEngine();

        System.out.printf("%8s %14s %9s %8s %8s%n", "threads", "epochReset", "launches", "maxDepth", "meanDep");
        for (int threads : THREAD_COUNTS) {
            run(engine, tables, threads, true);
            run(engine, tables, threads, false);
        }
    }

    private static void run(BlackwoodGpuEngine engine, BwGpuTables.GpuTableSet tables,
                            int threads, boolean withReset) {
        engine.uploadTables(tables);
        engine.resetEpoch();

        int highScore = 0, launches = 0;
        int[] depths = null;
        long deadline = System.nanoTime() + WALL_MILLIS * 1_000_000L;
        while (System.nanoTime() < deadline) {
            if (withReset && launches > 0 && launches % EPOCH_LAUNCHES == 0) {
                engine.uploadTables(tables);
                engine.resetEpoch();
            }
            long seedBase = System.nanoTime() ^ ((long) launches * 0x9E3779B97F4A7C15L);
            BlackwoodGpuEngine.GpuResult r =
                    engine.runBlackwoodDfs(seedBase, STEP_BUDGET, threads, highScore, new int[256]);
            highScore = Math.max(highScore, r.newHighScore());
            depths = r.threadDepths();
            launches++;
        }

        double mean = 0;
        int max = 0;
        if (depths != null) {
            for (int d : depths) { mean += d; max = Math.max(max, d); }
            mean /= depths.length;
        }
        // maxDepth here is the live population's best at the final launch; highScore is the
        // best seen at any point, which is what production would actually have saved.
        System.out.printf("%8d %14s %9d %8d %8.1f   (bestEverSeen=%d)%n",
                threads, withReset ? "every 60" : "never", launches, max, mean, highScore);
    }
}

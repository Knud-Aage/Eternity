package dk.puzzle.blackwood;

import dk.puzzle.gpu.BlackwoodGpuEngine;

/**
 * Measures where the Blackwood GPU kernel actually loses efficiency, instead of assuming.
 *
 * <p>Named with the 'Harness' suffix so Surefire's default {@code **&#47;*Test.java} pattern never
 * collects it -- it needs real CUDA hardware and the instrumented PTX. Run
 * {@code build-blackwood-profile-ptx.ps1} first; this loads
 * {@code SolveBlackwoodKernel.profile.ptx}, not the production kernel.</p>
 *
 * <p>Deliberately bounded (a handful of small launches, sub-second each) rather than
 * {@code BlackwoodGpuRunner}'s infinite loop, so it can be run repeatedly and also serve as a
 * fixed target for Nsight Compute.</p>
 *
 * <p>Reports two independent things:</p>
 * <ul>
 *   <li><b>Mean active lanes per warp per search step</b> -- the direct test of the warp-divergence
 *       theory. 32.0 = no divergence; a low number is the "GPU running at a fraction of its
 *       efficiency" claim, quantified.</li>
 *   <li><b>Cross-thread depth spread</b> -- from {@code threadDepths}, which the production runner
 *       already receives every launch and discards. This is the separate question of whether some
 *       threads get far deeper than others, which is what a work-stealing queue would address.</li>
 * </ul>
 */
public class BlackwoodGpuProfileHarness {

    private static final String PIECES_PATH = "src/main/resources/JBlackwood_Pieces.txt";
    private static final int NUM_THREADS = 2048;
    private static final long STEP_BUDGET = 20_000L;
    private static final int LAUNCHES = 5;

    // Mirrors the BW_PC_* defines in SolveBlackwoodKernel.cu.
    private static final int PC_WARP_ITERATIONS = 0;
    private static final int PC_ACTIVE_LANE_SUM = 1;
    private static final int PC_GENERAL_WARP_SAMPLES = 2;
    private static final int PC_GENERAL_MIXED = 3;
    private static final int PC_RESEED_WARP_SAMPLES = 4;
    private static final int PC_RESEED_MIXED = 5;
    private static final int PC_RESEED_EVENTS = 6;
    private static final int PC_CNT_SUM = 7;
    private static final int PC_CNT_SAMPLES = 8;
    private static final int PC_CNT_MAX = 9;

    public static void main(String[] args) throws Exception {
        System.out.println("=== Blackwood GPU divergence profile ===");
        System.out.printf("numThreads=%d stepBudget=%d launches=%d%n", NUM_THREADS, STEP_BUDGET, LAUNCHES);

        BlackwoodSolver solver = new BlackwoodSolver(999, null, 1, PIECES_PATH);
        BlackwoodGpuEngine engine = new BlackwoodGpuEngine(true);

        solver.prepare();
        engine.uploadTables(BwGpuTables.build(solver));
        engine.resetEpoch();
        engine.resetProfileCounters();

        long totalNodes = 0;
        long totalMillis = 0;
        int[] lastDepths = null;

        for (int i = 0; i < LAUNCHES; i++) {
            long seedBase = System.nanoTime() ^ ((long) i * 0x9E3779B97F4A7C15L);
            long t0 = System.nanoTime();
            BlackwoodGpuEngine.GpuResult r =
                    engine.runBlackwoodDfs(seedBase, STEP_BUDGET, NUM_THREADS, 0, new int[256]);
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            totalNodes += r.nodesTaken();
            totalMillis += ms;
            lastDepths = r.threadDepths();
            System.out.printf("  launch %d: %d ms, nodes=%d, highScore=%d%n", i, ms, r.nodesTaken(), r.newHighScore());
        }

        long[] c = engine.readProfileCounters();

        System.out.println();
        System.out.println("--- Warp divergence (the headline number) ---");
        double meanActive = c[PC_WARP_ITERATIONS] == 0
                ? 0 : (double) c[PC_ACTIVE_LANE_SUM] / c[PC_WARP_ITERATIONS];
        System.out.printf("mean active lanes per warp per step : %.2f / 32   (%.1f%% warp efficiency)%n",
                meanActive, 100.0 * meanActive / 32.0);
        System.out.printf("warp-iterations sampled             : %,d%n", c[PC_WARP_ITERATIONS]);

        System.out.println();
        System.out.println("--- Hypothesis counters ---");
        pct("warps split on 'has candidates'", c[PC_GENERAL_MIXED], c[PC_GENERAL_WARP_SAMPLES]);
        pct("warps split on 'needs reseed'  ", c[PC_RESEED_MIXED], c[PC_RESEED_WARP_SAMPLES]);
        System.out.printf("reseed events (lanes)               : %,d%n", c[PC_RESEED_EVENTS]);

        System.out.println();
        System.out.println("--- Candidate-list lengths ---");
        double meanCnt = c[PC_CNT_SAMPLES] == 0 ? 0 : (double) c[PC_CNT_SUM] / c[PC_CNT_SAMPLES];
        System.out.printf("mean cnt : %.2f   max cnt : %d   samples : %,d%n", meanCnt, c[PC_CNT_MAX], c[PC_CNT_SAMPLES]);

        System.out.println();
        System.out.println("--- Cross-thread depth spread (work-stealing's target) ---");
        reportDepths(lastDepths);

        System.out.println();
        System.out.printf("throughput (instrumented, NOT production-representative): %.1f Mnodes/s%n",
                totalMillis == 0 ? 0 : totalNodes / (totalMillis * 1000.0));
    }

    private static void pct(String label, long num, long den) {
        System.out.printf("%s : %,d / %,d = %.1f%%%n", label, num, den, den == 0 ? 0.0 : 100.0 * num / den);
    }

    private static void reportDepths(int[] depths) {
        if (depths == null || depths.length == 0) {
            System.out.println("(no depth data)");
            return;
        }
        int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;
        double sum = 0;
        for (int d : depths) {
            min = Math.min(min, d);
            max = Math.max(max, d);
            sum += d;
        }
        double mean = sum / depths.length;
        double varSum = 0;
        for (int d : depths) varSum += (d - mean) * (d - mean);
        System.out.printf("min=%d  max=%d  mean=%.1f  stddev=%.1f  (n=%d)%n",
                min, max, mean, Math.sqrt(varSum / depths.length), depths.length);
    }
}

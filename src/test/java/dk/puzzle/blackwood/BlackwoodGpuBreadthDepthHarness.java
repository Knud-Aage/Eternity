package dk.puzzle.blackwood;

import dk.puzzle.gpu.BlackwoodGpuEngine;

/**
 * Tests the hypothesis the divergence profile pointed at: that this kernel's problem is not warp
 * divergence (measured at 100% warp efficiency, zero reseed stalls by
 * {@link BlackwoodGpuProfileHarness}) but a breadth/depth mismatch.
 *
 * <p>Blackwood's algorithm derives its quality from sustained backtracking depth within a single
 * search lineage. The GPU's node throughput turned out to be roughly the same as the 28-thread CPU
 * solver's -- so running 16384 threads doesn't buy more total search, it just splits the same
 * search across ~500x more lineages, each getting ~1/500th the depth progress.</p>
 *
 * <p>This sweep holds TOTAL node budget constant and varies only how it's divided: many shallow
 * lineages vs few deep ones. Per-launch budget stays small and the launch count rises instead, so
 * every configuration stays far under the WDDM TDR watchdog while per-thread depth still
 * accumulates (thread state persists across launches within an epoch).</p>
 *
 * <p>If concentrating the same compute into fewer, deeper lineages reaches materially deeper, the
 * fix is a thread-count change, not a kernel rewrite.</p>
 */
public class BlackwoodGpuBreadthDepthHarness {

    private static final String PIECES_PATH = "src/main/resources/JBlackwood_Pieces.txt";
    private static final long STEP_BUDGET = 20_000L;
    /** threads x STEP_BUDGET x launches is held equal across every row. */
    private static final int[] THREAD_COUNTS = {16384, 4096, 1024, 256, 64};
    // Kept deliberately modest: low thread counts leave the GPU badly under-occupied, so the
    // narrow rows dominate wall-clock time even though every row does identical total work.
    private static final long TOTAL_NODES_TARGET = 16384L * STEP_BUDGET * 2;

    public static void main(String[] args) throws Exception {
        System.out.println("=== Blackwood GPU breadth-vs-depth sweep ===");
        System.out.printf("constant total node budget = %,d   stepBudget/launch = %,d%n%n",
                TOTAL_NODES_TARGET, STEP_BUDGET);
        System.out.printf("%8s %9s %12s %9s %8s %8s %9s%n",
                "threads", "launches", "nodes", "wall_ms", "maxDepth", "meanDep", "Mnodes/s");

        BlackwoodSolver solver = new BlackwoodSolver(999, null, 1, PIECES_PATH); // never saves
        solver.prepare();
        BwGpuTables.GpuTableSet tables = BwGpuTables.build(solver);

        BlackwoodGpuEngine engine = new BlackwoodGpuEngine();

        for (int threads : THREAD_COUNTS) {
            int launches = (int) (TOTAL_NODES_TARGET / ((long) threads * STEP_BUDGET));

            // Fresh epoch per row: every thread starts from a fresh attempt, so each row is an
            // independent trial rather than inheriting the previous row's accumulated depth.
            engine.uploadTables(tables);
            engine.resetEpoch();

            long nodes = 0;
            long t0 = System.nanoTime();
            int[] depths = null;
            int highScore = 0;
            for (int i = 0; i < launches; i++) {
                long seedBase = System.nanoTime() ^ ((long) i * 0x9E3779B97F4A7C15L);
                BlackwoodGpuEngine.GpuResult r =
                        engine.runBlackwoodDfs(seedBase, STEP_BUDGET, threads, highScore, new int[256]);
                nodes += r.nodesTaken();
                highScore = Math.max(highScore, r.newHighScore());
                depths = r.threadDepths();
            }
            long ms = (System.nanoTime() - t0) / 1_000_000L;

            double mean = 0;
            int max = 0;
            if (depths != null) {
                for (int d : depths) { mean += d; max = Math.max(max, d); }
                mean /= depths.length;
            }
            System.out.printf("%8d %9d %12d %9d %8d %8.1f %9.1f%n",
                    threads, launches, nodes, ms, max, mean, ms == 0 ? 0 : nodes / (ms * 1000.0));
        }

        System.out.println();
        System.out.println("maxDepth is the number that matters -- it's what gets saved and tracked.");

        // Equal total nodes is the clean scientific comparison, but production doesn't spend nodes,
        // it spends time -- and narrow configurations buy their depth at a large throughput cost.
        // This second table is the one that should drive the NUM_THREADS decision.
        System.out.println();
        System.out.println("=== Equal WALL-CLOCK comparison (what production actually trades) ===");
        System.out.printf("%8s %9s %12s %8s %8s%n", "threads", "launches", "nodes", "maxDepth", "meanDep");
        final long budgetMillis = 30_000L;
        for (int threads : THREAD_COUNTS) {
            engine.uploadTables(tables);
            engine.resetEpoch();

            long nodes = 0;
            int highScore = 0;
            int launches = 0;
            int[] depths = null;
            long deadline = System.nanoTime() + budgetMillis * 1_000_000L;
            while (System.nanoTime() < deadline) {
                long seedBase = System.nanoTime() ^ ((long) launches * 0x9E3779B97F4A7C15L);
                BlackwoodGpuEngine.GpuResult r =
                        engine.runBlackwoodDfs(seedBase, STEP_BUDGET, threads, highScore, new int[256]);
                nodes += r.nodesTaken();
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
            System.out.printf("%8d %9d %12d %8d %8.1f%n", threads, launches, nodes, max, mean);
        }
    }
}

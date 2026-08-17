package dk.puzzle.blackwood;

import dk.puzzle.gpu.BlackwoodGpuEngine;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Verifies GPU seeding and measures whether it actually helps.
 *
 * <p>Three stages, in order of how cheap they are to trust:</p>
 * <ol>
 *   <li><b>Host-side seed validation</b> -- every loaded board must be a legal piece multiset (no
 *       piece used twice). A seed that fails this would corrupt the kernel's pieceUsed bitmask.</li>
 *   <li><b>Replay fidelity</b> -- the kernel must be able to re-place each seed through its own
 *       candidate tables. Reported as seedShortfalls; a large count means the seed boards are not
 *       reachable through the current tables and seeding is quietly degrading to shallow starts.</li>
 *   <li><b>Equal-wall-clock A/B</b> -- seeded vs unseeded, same thread count and time budget,
 *       compared on depth. This is the only stage that answers whether seeding is worth using.</li>
 * </ol>
 *
 * <p>'Harness' suffix keeps Surefire from collecting it; needs real CUDA hardware.</p>
 */
public class BlackwoodGpuSeedingHarness {

    private static final String PIECES_PATH = "src/main/resources/JBlackwood_Pieces.txt";
    private static final int NUM_THREADS = 1024;
    private static final long STEP_BUDGET = 50_000L;
    private static final long WALL_MILLIS = 90_000L;
    private static final int MIN_SEED_DEPTH = 245;
    private static final int MAX_SEEDS = 256;
    private static final int MAX_RETREAT = 40;

    public static void main(String[] args) throws Exception {
        BlackwoodSolver solver = new BlackwoodSolver(999, null, 1, PIECES_PATH); // never saves
        solver.prepare();
        BwGpuTables.GpuTableSet tables = BwGpuTables.build(solver);

        // --- Stage 1: load and validate seeds host-side ---
        Path home = Path.of(System.getProperty("user.home"));
        List<Path> dirs = List.of(
                home.resolve("EternitySolutions_GpuBlackwood"),
                home.resolve("EternitySolutions"),
                home.resolve("EternitySolutions_drop239"),
                home.resolve("Documents").resolve("EternitySolutions_JavaPort"));

        List<BwSeedLoader.Seed> seeds = BwSeedLoader.load(dirs, MIN_SEED_DEPTH, MAX_SEEDS, tables.stepBoardIdx());
        System.out.println("=== Stage 1: seed pool ===");
        System.out.printf("loaded %d seed board(s) at depth >= %d%n", seeds.size(), MIN_SEED_DEPTH);
        if (seeds.isEmpty()) {
            System.out.println("No seeds found -- nothing to verify. Check the save directories.");
            return;
        }
        System.out.printf("deepest=%d  shallowest=%d%n", seeds.get(0).depth(), seeds.get(seeds.size() - 1).depth());
        System.out.println("deepest source: " + seeds.get(0).source().getFileName());

        int illegal = 0;
        for (BwSeedLoader.Seed seed : seeds) {
            Set<Integer> used = new HashSet<>();
            for (int step = 0; step < seed.depth(); step++) {
                int pieceNumber = seed.stepEncoded()[step] >> 2;
                if (!used.add(pieceNumber)) { illegal++; break; }
            }
        }
        System.out.printf("duplicate-piece check: %d of %d seeds ILLEGAL%n", illegal, seeds.size());
        if (illegal > 0) {
            System.out.println("!!! refusing to continue -- illegal seeds would corrupt the kernel's piece bitmask");
            System.exit(1);
        }

        List<int[]> encoded = new ArrayList<>();
        int[] depths = new int[seeds.size()];
        for (int i = 0; i < seeds.size(); i++) {
            encoded.add(seeds.get(i).stepEncoded());
            depths[i] = seeds.get(i).depth();
        }

        // --- Stage 2: replay fidelity on the GPU ---
        BlackwoodGpuEngine engine = new BlackwoodGpuEngine();
        engine.uploadTables(tables);
        engine.uploadSeeds(encoded, depths, 0); // retreat 0: every thread must reach its seed's full depth
        engine.resetEpoch();

        System.out.println();
        System.out.println("=== Stage 2: replay fidelity (maxRetreat=0, so depth should match the pool) ===");
        BlackwoodGpuEngine.GpuResult probe =
                engine.runBlackwoodDfs(System.nanoTime(), 1L, NUM_THREADS, 0, new int[256]);
        int shortfalls = engine.readAndResetSeedShortfalls();
        int minDepth = Integer.MAX_VALUE, maxDepth = 0;
        for (int d : probe.threadDepths()) { minDepth = Math.min(minDepth, d); maxDepth = Math.max(maxDepth, d); }
        System.out.printf("after a 1-node launch: thread depth min=%d max=%d, seedShortfalls=%d of %d threads%n",
                minDepth, maxDepth, shortfalls, NUM_THREADS);
        System.out.println(shortfalls == 0
                ? "OK -- every thread replayed its seed to full depth"
                : "WARNING -- some seeds are not reachable through the current candidate tables");

        // --- Stage 3: equal-wall-clock A/B ---
        System.out.println();
        System.out.println("=== Stage 3: seeded vs unseeded, equal wall-clock ===");
        System.out.printf("%10s %9s %8s %8s%n", "arm", "launches", "maxDepth", "meanDep");
        runArm(engine, tables, encoded, depths, false);
        runArm(engine, tables, encoded, depths, true);
    }

    private static void runArm(BlackwoodGpuEngine engine, BwGpuTables.GpuTableSet tables,
                               List<int[]> encoded, int[] depths, boolean seeded) {
        engine.uploadTables(tables);
        if (seeded) {
            engine.uploadSeeds(encoded, depths, MAX_RETREAT);
        } else {
            engine.uploadSeeds(List.of(), new int[0], 0);
        }
        engine.resetEpoch();

        int highScore = 0, launches = 0;
        int[] depthsOut = null;
        long deadline = System.nanoTime() + WALL_MILLIS * 1_000_000L;
        while (System.nanoTime() < deadline) {
            long seedBase = System.nanoTime() ^ ((long) launches * 0x9E3779B97F4A7C15L);
            BlackwoodGpuEngine.GpuResult r =
                    engine.runBlackwoodDfs(seedBase, STEP_BUDGET, NUM_THREADS, highScore, new int[256]);
            highScore = Math.max(highScore, r.newHighScore());
            depthsOut = r.threadDepths();
            launches++;
        }

        double mean = 0;
        int max = 0;
        if (depthsOut != null) {
            for (int d : depthsOut) { mean += d; max = Math.max(max, d); }
            mean /= depthsOut.length;
        }
        System.out.printf("%10s %9d %8d %8.1f   (bestEverSeen=%d)%n",
                seeded ? "seeded" : "unseeded", launches, max, mean, highScore);
    }
}

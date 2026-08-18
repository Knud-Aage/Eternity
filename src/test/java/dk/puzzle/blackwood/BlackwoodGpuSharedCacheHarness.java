package dk.puzzle.blackwood;

import dk.puzzle.gpu.BlackwoodGpuEngine;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Verifies the 2026-08-18 shared-memory caching change (solveBlackwoodDfsShared vs
 * solveBlackwoodDfs). Both entry points call the exact same runBlackwoodDfsBody() and differ only
 * in whether the four hot per-step tables are read from __constant__ or a block-local __shared__
 * copy of it -- so unlike most A/Bs in this project, this one has a MUCH stronger correctness bar
 * available: given the same seedBase and the same starting state, the two kernels must produce
 * bit-identical results. Any divergence means the shared-memory population or the parameterization
 * refactor has a real bug, not an expected difference (seedBase only matters for freshly-init'd
 * threads -- d_needsInit -- so it stays deterministic even across chained/resumed launches).
 *
 * <p>Uses exactly ONE {@code BlackwoodGpuEngine} throughout (found the hard way: every constructor
 * call does its own {@code cuCtxCreate}, and a second instance's context silently becomes "current"
 * for the process, breaking the first instance's launches with CUDA_ERROR_INVALID_HANDLE -- nothing
 * in this codebase was ever built for multiple engine instances to coexist). {@link
 * BlackwoodGpuEngine#resetEpoch()} between configurations is what gives each one a genuinely fresh,
 * identical starting state instead of a two-engine constructor.</p>
 *
 * <p>Test 3 is the separate, actually-interesting question: does caching the tables measurably
 * change depth-record pace? Equal-GPU-time, unseeded (matches what was actually running when this
 * harness paused it), success metric is depth records -- not nodesTaken/throughput, which this
 * project has been burned by trusting before (GpuEngine.java:137-153).</p>
 *
 * <p>'Harness' suffix keeps Surefire from collecting it; needs real CUDA hardware, and must not
 * run concurrently with another GPU process.</p>
 */
public class BlackwoodGpuSharedCacheHarness {

    private static final String PIECES_PATH = "src/main/resources/JBlackwood_Pieces.txt";
    private static final int NUM_THREADS = 1024;
    private static final long STEP_BUDGET = 50_000L;
    private static final int CHAINED_LAUNCHES = 3;
    private static final long AB_WALL_MILLIS = 180_000L; // 3 min/arm -- a quick smoke-level read, not a definitive one

    public static void main(String[] args) throws Exception {
        BlackwoodSolver solver = new BlackwoodSolver(999, null, 1, PIECES_PATH); // never saves
        solver.prepare();
        BwGpuTables.GpuTableSet tables = BwGpuTables.build(solver);

        BlackwoodGpuEngine engine = new BlackwoodGpuEngine();
        engine.uploadTables(tables);

        System.out.println("=== Shared-memory cache verification ===");
        testChainedEquivalence(engine);
        testStructuralNoDuplicates(engine);
        testEqualTimeDepthAB(engine);

        System.out.println();
        System.out.println(">>> ALL SHARED-CACHE VERIFICATIONS PASSED");
    }

    private static void testChainedEquivalence(BlackwoodGpuEngine engine) {
        System.out.println("Test 1: constant-memory vs shared-memory, " + CHAINED_LAUNCHES + " chained launches, same seeds...");

        engine.setSharedCacheEnabled(false);
        engine.resetEpoch();
        int constHigh = 0;
        long constNodes = 0;
        int[] constDepths = null;
        int[] boardConstFinal = new int[256];
        for (int i = 0; i < CHAINED_LAUNCHES; i++) {
            long seedBase = 0x1234567890ABCDEFL ^ ((long) i * 0x9E3779B97F4A7C15L); // fixed, identical on both sides
            int[] boardConst = new int[256];
            BlackwoodGpuEngine.GpuResult r = engine.runBlackwoodDfs(seedBase, STEP_BUDGET, NUM_THREADS, constHigh, boardConst);
            if (r.newHighScore() > constHigh) boardConstFinal = boardConst;
            constHigh = r.newHighScore();
            constNodes += r.nodesTaken();
            constDepths = r.threadDepths();
        }
        System.out.printf("  constant: highScore=%d, totalNodes=%d%n", constHigh, constNodes);

        engine.setSharedCacheEnabled(true);
        engine.resetEpoch();
        int sharedHigh = 0;
        long sharedNodes = 0;
        int[] sharedDepths = null;
        int[] boardSharedFinal = new int[256];
        for (int i = 0; i < CHAINED_LAUNCHES; i++) {
            long seedBase = 0x1234567890ABCDEFL ^ ((long) i * 0x9E3779B97F4A7C15L); // same sequence as above
            int[] boardShared = new int[256];
            BlackwoodGpuEngine.GpuResult r = engine.runBlackwoodDfs(seedBase, STEP_BUDGET, NUM_THREADS, sharedHigh, boardShared);
            if (r.newHighScore() > sharedHigh) boardSharedFinal = boardShared;
            sharedHigh = r.newHighScore();
            sharedNodes += r.nodesTaken();
            sharedDepths = r.threadDepths();
        }
        System.out.printf("  shared  : highScore=%d, totalNodes=%d%n", sharedHigh, sharedNodes);

        if (constHigh != sharedHigh) {
            throw new IllegalStateException("highScore mismatch -- const=" + constHigh + " shared=" + sharedHigh);
        }
        if (constNodes != sharedNodes) {
            throw new IllegalStateException("total nodesTaken mismatch -- const=" + constNodes + " shared=" + sharedNodes);
        }
        if (!Arrays.equals(constDepths, sharedDepths)) {
            throw new IllegalStateException("final threadDepths arrays differ between const and shared");
        }
        if (!Arrays.equals(boardConstFinal, boardSharedFinal)) {
            throw new IllegalStateException("best-board output differs between const and shared");
        }
        System.out.println("  PASSED: constant-memory and shared-memory kernels are bit-identical across " + CHAINED_LAUNCHES + " chained launches");
    }

    private static void testStructuralNoDuplicates(BlackwoodGpuEngine engine) {
        System.out.println("Test 2: structural check (no duplicate physical pieces) under shared-cache mode...");

        engine.setSharedCacheEnabled(true);
        engine.resetEpoch();

        int[] boardOut = new int[256];
        long seedBase = System.nanoTime();
        BlackwoodGpuEngine.GpuResult r = engine.runBlackwoodDfs(seedBase, 500_000L, NUM_THREADS, 0, boardOut);
        r = engine.runBlackwoodDfs(seedBase ^ 1L, 500_000L, NUM_THREADS, r.newHighScore(), boardOut); // exercises the persisted-resume path too

        Set<Integer> seen = new HashSet<>();
        int placed = 0;
        for (int cell : boardOut) {
            if (cell == -1) continue;
            placed++;
            BwRotatedPiece p = BwGpuTables.unpack(cell);
            if (!seen.add(p.pieceNumber())) {
                throw new IllegalStateException("Duplicate piece number " + p.pieceNumber() + " on shared-cache board");
            }
        }
        System.out.println("  PASSED: " + placed + " placed pieces, all distinct piece numbers (highScore=" + r.newHighScore() + ")");
    }

    private static void testEqualTimeDepthAB(BlackwoodGpuEngine engine) {
        System.out.println();
        System.out.println("Test 3: equal-GPU-time depth-record A/B, unseeded, " + (AB_WALL_MILLIS / 1000) + "s/arm...");
        System.out.printf("%10s %9s %8s %8s%n", "arm", "launches", "maxDep", "meanDep");

        runArm(engine, false, "constant");
        runArm(engine, true, "shared");

        System.out.println();
        System.out.println("Quick smoke-level read only -- not a substitute for a real multi-hour A/B.");
        System.out.println("The real question this answers is directional: does caching help, hurt, or wash out.");
    }

    private static void runArm(BlackwoodGpuEngine engine, boolean sharedCache, String label) {
        engine.setSharedCacheEnabled(sharedCache);
        engine.resetEpoch();

        int highScore = 0, launches = 0;
        int[] depths = null;
        long deadline = System.nanoTime() + AB_WALL_MILLIS * 1_000_000L;
        while (System.nanoTime() < deadline) {
            long seedBase = System.nanoTime() ^ ((long) launches * 0x9E3779B97F4A7C15L);
            BlackwoodGpuEngine.GpuResult r = engine.runBlackwoodDfs(seedBase, 200_000L, NUM_THREADS, highScore, new int[256]);
            highScore = Math.max(highScore, r.newHighScore());
            depths = r.threadDepths();
            launches++;
        }

        double meanDepth = 0;
        int maxDepth = 0;
        if (depths != null) {
            for (int d : depths) { meanDepth += d; maxDepth = Math.max(maxDepth, d); }
            meanDepth /= depths.length;
        }
        System.out.printf("%10s %9d %8d %8.1f%n", label, launches, maxDepth, meanDepth);
    }
}

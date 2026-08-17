package dk.puzzle.blackwood;

import dk.puzzle.gpu.BlackwoodGpuEngine;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Standalone launcher for the GPU-native Blackwood kernel ({@code SolveBlackwoodKernel.cu}),
 * mirroring {@link BlackwoodSolver}'s own standalone-launcher convention (own {@code main()}, no
 * GUI/{@code StartupDialog} wiring -- see {@code run-blackwood.cmd}).
 *
 * <p>Each GPU launch resumes every thread's persisted in-progress search rather than restarting it
 * (see {@code SolveBlackwoodKernel.cu}'s 2026-08-04 header note) -- so table rebuilds can no longer
 * happen every launch (a resumed cursor into a replaced table would point at the wrong candidates).
 * Instead, {@link BlackwoodSolver#prepare()} + {@link BwGpuTables#build} + a full thread-state reset
 * ({@link BlackwoodGpuEngine#resetEpoch()}) happen only once every {@link #EPOCH_LAUNCHES} launches
 * ("epoch" boundaries); every other launch just resumes. {@code solver} here is used only for its
 * {@code prepare()} output (the candidate tables) -- {@code solvePuzzle()}/{@code run()} are never
 * called; the actual search happens entirely on the GPU.</p>
 */
public class BlackwoodGpuRunner {

    private static final Logger logger = LogManager.getLogger(BlackwoodGpuRunner.class);

    private static final String PIECES_PATH = "src/main/resources/JBlackwood_Pieces.txt";
    private static final int SAVE_THRESHOLD = 190; // matches BlackwoodSolver's own default, for direct comparability
    // 2026-08-17, measured (BlackwoodGpuBreadthDepthHarness): 16384 was chosen for SM saturation --
    // i.e. for raw node throughput -- but throughput turns out to be nearly irrelevant to the metric
    // that matters. At equal wall-clock, 16384 threads searched 47.5 BILLION nodes and 64 threads
    // searched 0.55 billion (86x fewer), yet reached the same max depth within 2 pieces. Spreading
    // the budget over more lineages just makes every lineage shallower: at equal TOTAL NODES, max
    // depth went 160 (16384 threads) -> 246 (64 threads).
    // 1024 keeps the population mean depth high without giving up the parallelism entirely; it and
    // 16384 tied on max depth at equal wall-clock, but 1024 held a much deeper mean (205 vs 174).
    private static final int NUM_THREADS = 1024;
    private static final long INITIAL_STEP_BUDGET = 50_000L;
    private static final long MIN_STEP_BUDGET = 1_000L;
    private static final long FAST_LAUNCH_MILLIS = 500L;  // below this, double the budget next launch
    private static final long SLOW_LAUNCH_MILLIS = 1_500L; // above this, halve it (staying under the ~2000ms WDDM TDR default)

    // How many launches share one table generation + persisted thread-search-state before
    // everything resets fresh (a table rebuild re-randomizes candidate order, so every resume
    // cursor into the old tables must be discarded with it -- see BlackwoodGpuEngine.resetEpoch).
    //
    // 2026-08-17, measured (BlackwoodGpuEpochResetHarness), equal wall-clock per arm, reset being
    // the ONLY difference:
    //     16384 threads: mean population depth  76.7 (reset every 60)  ->  174.4 (never)
    //      1024 threads: mean population depth 100.6 (reset every 60)  ->  205.1 (never)
    // Max depth likewise 236->248 and 245->248. The old value was costing roughly HALF the
    // population's accumulated depth: climbing from scratch to ~247 takes ~30s, and at this
    // launch cadence a 60-launch epoch is only about a minute, so most of every epoch was spent
    // re-covering ground the previous epoch had already covered, over and over.
    //
    // The rebuild's purpose is diversification, but that is already supplied per-thread (each
    // thread re-randomizes its own bottomSides every attempt from its own RNG stream), so the
    // global rebuild was buying very little of it at a very high price. Kept rather than removed
    // outright so tables never go stale indefinitely -- just at an interval that no longer
    // truncates the sustained backtracking this algorithm depends on.
    private static final long EPOCH_LAUNCHES = 20_000;

    // Seeding from previously saved deep boards. The kernel reaches ~247 pieces from scratch in
    // about 30 seconds but took three days of running to reach 251, so nearly all of the compute
    // spent re-deriving the first ~247 pieces is spent re-covering known ground. Resuming from the
    // deepest boards already on disk puts every thread at the frontier instead.
    // MIN_SEED_DEPTH is deliberately just below the best boards on disk -- shallower boards would
    // dilute the pool without adding frontier coverage.
    private static final int MIN_SEED_DEPTH = 245;
    private static final int MAX_SEEDS = 256;
    // How far back from a seed's tip a thread may randomly pull before resuming. Needed for
    // diversity (candidate order is global, so same board + same depth = duplicated work), and it
    // also lets threads explore alternatives that branch off well below the tip.
    private static final int MAX_RETREAT = 40;

    public static void main(String[] args) throws Exception {
        // NOT Documents: it is OneDrive-redirected by Known Folder Move on this machine, so every
        // board saved here was being uploaded to the cloud. UserProfile is never redirected.
        // Same reasoning (and the same override-by-env-var escape hatch) as the C# solver's own
        // save path. Override with ETERNITY_GPU_SOLUTIONS_DIR to put boards on another drive.
        String configuredDir = System.getenv("ETERNITY_GPU_SOLUTIONS_DIR");
        Path outputDir = (configuredDir == null || configuredDir.isBlank())
                ? Path.of(System.getProperty("user.home"), "EternitySolutions_GpuBlackwood")
                : Path.of(configuredDir);

        BlackwoodSolver solver = new BlackwoodSolver(SAVE_THRESHOLD, outputDir, 1, PIECES_PATH);

        List<BwPiece> pieces = BwUtil.getPieces(PIECES_PATH);
        BwPiece[] pieceByNumber = new BwPiece[257];
        for (BwPiece p : pieces) {
            pieceByNumber[p.pieceNumber()] = p;
        }

        BlackwoodGpuEngine engine = new BlackwoodGpuEngine();
        long launchCounter = 0;
        int currentHighScore = scanExistingHighScore(outputDir);
        long stepBudget = INITIAL_STEP_BUDGET;

        logger.info("BlackwoodGpuRunner starting. numThreads={}, initialStepBudget={}, saveThreshold={}, epochLaunches={}, resumedHighScore={}",
                NUM_THREADS, INITIAL_STEP_BUDGET, SAVE_THRESHOLD, EPOCH_LAUNCHES, currentHighScore);

        while (true) {
            if (launchCounter % EPOCH_LAUNCHES == 0) {
                solver.prepare();
                BwGpuTables.GpuTableSet tables = BwGpuTables.build(solver);
                engine.uploadTables(tables);
                // Reload seeds at each epoch: boards saved since the last boundary (including this
                // run's own new records) become seeds for the next one, so the frontier advances.
                loadSeeds(engine, tables.stepBoardIdx(), outputDir);
                engine.resetEpoch();
                logger.info("Epoch boundary at launch {}: tables refreshed, {} seed board(s) active, all thread state reset",
                        launchCounter, engine.getNumSeeds());
            }

            long seedBase = System.nanoTime() ^ (launchCounter++ * 0x9E3779B97F4A7C15L);
            int[] bestBoardOut = new int[256];

            long launchStartNanos = System.nanoTime();
            BlackwoodGpuEngine.GpuResult result =
                    engine.runBlackwoodDfs(seedBase, stepBudget, NUM_THREADS, currentHighScore, bestBoardOut);
            long launchMillis = (System.nanoTime() - launchStartNanos) / 1_000_000L;

            // Population depth stats: runBlackwoodDfs has always returned per-thread depths and
            // this loop always discarded them. They are the direct read-out of whether threads are
            // accumulating sustained depth or being repeatedly knocked back to shallow water --
            // exactly the signal that showed the old 60-launch epoch was halving mean depth.
            // Replay shortfalls are the health check on the seed pool: a seed that isn't reachable
            // through the current candidate tables (wrong piece numbering, incompatible break
            // schedule) silently resumes shallower than intended, which would otherwise look like
            // seeding simply not helping.
            int shortfalls = engine.getNumSeeds() > 0 ? engine.readAndResetSeedShortfalls() : 0;
            logger.info("Launch {}: {} threads, stepBudget={}, {} ms, nodesTaken={}, newHighScore={}, solved={}, depth[{}], seedShortfalls={}",
                    launchCounter, NUM_THREADS, stepBudget, launchMillis,
                    result.nodesTaken(), result.newHighScore(), result.solved(),
                    describeDepths(result.threadDepths()), shortfalls);

            if (result.newHighScore() > currentHighScore || result.solved()) {
                currentHighScore = result.newHighScore();
                if (currentHighScore >= SAVE_THRESHOLD) {
                    trySave(bestBoardOut, currentHighScore, pieceByNumber, outputDir);
                }
            }

            if (launchMillis < FAST_LAUNCH_MILLIS) {
                stepBudget = stepBudget * 2;
            } else if (launchMillis > SLOW_LAUNCH_MILLIS) {
                stepBudget = Math.max(MIN_STEP_BUDGET, stepBudget / 2);
            }
        }
    }

    /**
     * Gathers the deepest boards this project has saved -- from this runner, the CPU Java port, and
     * the C# solver -- and hands them to the GPU as resume points. All three write the same grid
     * format (see {@link BwSeedLoader}). Failure here is never fatal: with no seeds the kernel just
     * starts from random corners as it always did.
     */
    private static void loadSeeds(BlackwoodGpuEngine engine, int[] stepBoardIdx, Path gpuOutputDir) {
        try {
            Path home = Path.of(System.getProperty("user.home"));
            List<Path> dirs = List.of(
                    gpuOutputDir,
                    home.resolve("EternitySolutions"),             // C# solver
                    home.resolve("EternitySolutions_drop239"),     // C# solver, tuned break schedule
                    home.resolve("Documents").resolve("EternitySolutions_JavaPort"));

            List<BwSeedLoader.Seed> seeds = BwSeedLoader.load(dirs, MIN_SEED_DEPTH, MAX_SEEDS, stepBoardIdx);
            if (seeds.isEmpty()) {
                logger.info("No seed boards at depth >= {} found; threads will start from random corners", MIN_SEED_DEPTH);
                engine.uploadSeeds(List.of(), new int[0], 0);
                return;
            }

            List<int[]> encoded = new ArrayList<>(seeds.size());
            int[] depths = new int[seeds.size()];
            for (int i = 0; i < seeds.size(); i++) {
                encoded.add(seeds.get(i).stepEncoded());
                depths[i] = seeds.get(i).depth();
            }
            engine.uploadSeeds(encoded, depths, MAX_RETREAT);
            logger.info("Seeding from {} board(s), depths {}..{}, maxRetreat={}",
                    seeds.size(), depths[depths.length - 1], depths[0], MAX_RETREAT);
        } catch (Exception e) {
            logger.warn("Seed loading failed; continuing without seeds", e);
            try {
                engine.uploadSeeds(List.of(), new int[0], 0);
            } catch (Exception ignored) {
                // already unseeded
            }
        }
    }

    private static String describeDepths(int[] depths) {
        if (depths == null || depths.length == 0) return "n/a";
        int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;
        long sum = 0;
        for (int d : depths) {
            if (d < min) min = d;
            if (d > max) max = d;
            sum += d;
        }
        return String.format("min=%d mean=%.1f max=%d", min, (double) sum / depths.length, max);
    }

    /**
     * Recovers {@code currentHighScore} from the boards already saved on disk from a previous run,
     * instead of always starting bookkeeping at 0 -- every save already contains its full board
     * (via {@link #trySave}), so the deepest board ever found is already durably persisted the
     * moment it's saved; this just stops the runner from acting like a restart erased that
     * progress. Filenames are {@code "<pieces>_<uuid>_<timestamp>.txt"}, the same convention the
     * conflict-tracking scripts already parse.
     */
    private static int scanExistingHighScore(Path outputDir) {
        if (!Files.isDirectory(outputDir)) return 0;
        int max = 0;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(outputDir)) {
            for (Path p : stream) {
                String name = p.getFileName().toString();
                int underscore = name.indexOf('_');
                if (underscore <= 0) continue;
                try {
                    int pieces = Integer.parseInt(name.substring(0, underscore));
                    if (pieces > max) max = pieces;
                } catch (NumberFormatException ignored) {
                    // not one of our save files -- skip
                }
            }
        } catch (IOException e) {
            logger.warn("Could not scan {} for existing saves, starting currentHighScore from 0", outputDir, e);
            return 0;
        }
        return max;
    }

    private static void trySave(int[] board, int maxSolveIndex, BwPiece[] pieceByNumber, Path outputDir) {
        BwRotatedPiece[] rotatedBoard = new BwRotatedPiece[256];
        for (int i = 0; i < 256; i++) {
            rotatedBoard[i] = (board[i] == -1) ? BwRotatedPiece.EMPTY : BwGpuTables.unpack(board[i]);
        }
        try {
            Path saved = BwUtil.saveBoard(rotatedBoard, maxSolveIndex, pieceByNumber, outputDir);
            logger.info("Saved new personal best: maxSolveIndex={} -> {}", maxSolveIndex, saved);
        } catch (Exception e) {
            logger.error("Failed to save board at solveIndex={}", maxSolveIndex, e);
        }
    }
}

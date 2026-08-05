package dk.puzzle.blackwood;

import dk.puzzle.gpu.BlackwoodGpuEngine;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
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
    private static final int NUM_THREADS = 16384; // GPU was badly under-saturated at 4096 (16 blocks of 256) -- bumped
    // toward MAX_THREADS=20_000 headroom in BlackwoodGpuEngine so more SMs get real work per launch. The auto-tune
    // step-budget loop below self-corrects stepBudget down to compensate for the extra per-launch work.
    private static final long INITIAL_STEP_BUDGET = 50_000L;
    private static final long MIN_STEP_BUDGET = 1_000L;
    private static final long FAST_LAUNCH_MILLIS = 500L;  // below this, double the budget next launch
    private static final long SLOW_LAUNCH_MILLIS = 1_500L; // above this, halve it (staying under the ~2000ms WDDM TDR default)

    // How many launches share one table generation + persisted thread-search-state before
    // everything resets fresh. At the ~900ms/launch cadence this has run at, 60 launches is
    // roughly a 1-minute epoch -- giving each thread up to ~60x the sustained backtracking budget
    // it had before (bounded only by epoch length, not a single launch's stepBudget), while still
    // keeping the cost of an epoch boundary (discarding in-progress state) small relative to how
    // long this process actually runs for.
    private static final long EPOCH_LAUNCHES = 60;

    public static void main(String[] args) throws Exception {
        Path outputDir = Path.of(System.getProperty("user.home"), "Documents", "EternitySolutions_GpuBlackwood");

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
                engine.resetEpoch();
                logger.info("Epoch boundary at launch {}: tables refreshed, all thread state reset to fresh attempts", launchCounter);
            }

            long seedBase = System.nanoTime() ^ (launchCounter++ * 0x9E3779B97F4A7C15L);
            int[] bestBoardOut = new int[256];

            long launchStartNanos = System.nanoTime();
            BlackwoodGpuEngine.GpuResult result =
                    engine.runBlackwoodDfs(seedBase, stepBudget, NUM_THREADS, currentHighScore, bestBoardOut);
            long launchMillis = (System.nanoTime() - launchStartNanos) / 1_000_000L;

            logger.info("Launch {}: {} threads, stepBudget={}, {} ms, nodesTaken={}, newHighScore={}, solved={}",
                    launchCounter, NUM_THREADS, stepBudget, launchMillis,
                    result.nodesTaken(), result.newHighScore(), result.solved());

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

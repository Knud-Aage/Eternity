package dk.puzzle.blackwood;

import dk.puzzle.gpu.BlackwoodGpuEngine;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.file.Path;
import java.util.List;

/**
 * Standalone launcher for the GPU-native Blackwood kernel ({@code SolveBlackwoodKernel.cu}),
 * mirroring {@link BlackwoodSolver}'s own standalone-launcher convention (own {@code main()}, no
 * GUI/{@code StartupDialog} wiring -- see {@code run-blackwood.cmd}).
 *
 * <p>Each loop iteration is one "batch": {@link BlackwoodSolver#prepare()} rebuilds the 10
 * batch-level candidate tables with fresh score+jitter (matching Blackwood's own per-outer-pass
 * re-randomization), {@link BwGpuTables#build} flattens them to GPU CSR form, and one GPU launch
 * runs {@code numThreads} independent full attempts. {@code solver} here is used only for its
 * {@code prepare()} output (the candidate tables) -- {@code solvePuzzle()}/{@code run()} are never
 * called; the actual search happens entirely on the GPU.</p>
 */
public class BlackwoodGpuRunner {

    private static final Logger logger = LogManager.getLogger(BlackwoodGpuRunner.class);

    private static final String PIECES_PATH = "src/main/resources/JBlackwood_Pieces.txt";
    private static final int SAVE_THRESHOLD = 190; // matches BlackwoodSolver's own default, for direct comparability
    private static final int NUM_THREADS = 4096; // plan's scaled comparison-readiness range is 2048-8192
    private static final long INITIAL_STEP_BUDGET = 50_000L;
    private static final long MIN_STEP_BUDGET = 1_000L;
    private static final long FAST_LAUNCH_MILLIS = 500L;  // below this, double the budget next launch
    private static final long SLOW_LAUNCH_MILLIS = 1_500L; // above this, halve it (staying under the ~2000ms WDDM TDR default)

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
        int currentHighScore = 0;
        long stepBudget = INITIAL_STEP_BUDGET;

        logger.info("BlackwoodGpuRunner starting. numThreads={}, initialStepBudget={}, saveThreshold={}",
                NUM_THREADS, INITIAL_STEP_BUDGET, SAVE_THRESHOLD);

        while (true) {
            solver.prepare();
            BwGpuTables.GpuTableSet tables = BwGpuTables.build(solver);
            engine.uploadTables(tables);

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

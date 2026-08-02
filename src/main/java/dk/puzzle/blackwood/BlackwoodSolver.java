package dk.puzzle.blackwood;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Predicate;

/**
 * Faithful port of Blackwood's {@code Program.cs} — table orchestration, the
 * per-attempt chronological-backtracking search, and the outer batch loop.
 * Runs entirely in his own raw colour numbering; see BwUtil/BwPiece.
 */
public class BlackwoodSolver {

    private static final Logger logger = LogManager.getLogger(BlackwoodSolver.class);

    static final long DEFAULT_NODE_CAP = 50_000_000_000L; // matches C# `node_count > 50000000000`
    private static final int DEFAULT_SAVE_THRESHOLD = 190; // matches the currently-running C# instance, not his original 252
    private static final int ATTEMPTS_PER_WORKER_PER_BATCH = 5;

    private final int saveThreshold;
    private final Path outputDir;
    private final int numWorkers;
    private final String piecesFilePath;

    private List<BwPiece> boardPieces;
    private BwPiece[] pieceByNumber; // index = pieceNumber, length 257

    // Rebuilt by prepare() once per outer batch; read-only for that batch's lifetime.
    // Safe publication relies on all workers being submit()'d only after prepare()
    // fully returns (ExecutorService.submit()'s happens-before guarantee) -- no
    // volatile/synchronized needed, matching how EternitySolver.CpuSearchWorker
    // already relies on the same pattern in this codebase.
    private BwRotatedPiece[][] corners;
    private BwRotatedPiece[][] leftSides;
    private BwRotatedPiece[][] topSides;
    private BwRotatedPiece[][] rightSidesWithBreaks;
    private BwRotatedPiece[][] rightSidesWithoutBreaks;
    private BwRotatedPiece[][] middlesWithBreak;
    private BwRotatedPiece[][] middlesNoBreak;
    private BwRotatedPiece[][] southStart;
    private BwRotatedPiece[][] westStart;
    private BwRotatedPiece[][] start;
    private Map<Integer, List<BwUtil.RotatedCandidate>> bottomSidePiecesRotated; // raw, re-sorted every attempt
    private BwRotatedPiece[][][] masterPieceLookup;
    private int[] boardOrderRow;
    private int[] boardOrderCol;
    private int[] breakArray;
    private int[] heuristicArray;

    // Verification instrumentation -- see plan's "solve_index==0 edge case" note.
    private final AtomicLong exhaustedAtSeedCount = new AtomicLong();

    public BlackwoodSolver() {
        this(DEFAULT_SAVE_THRESHOLD, defaultOutputDir(), defaultWorkerCount(), "src/main/resources/JBlackwood_Pieces.txt");
    }

    public BlackwoodSolver(int saveThreshold, Path outputDir, int numWorkers, String piecesFilePath) {
        this.saveThreshold = saveThreshold;
        this.outputDir = outputDir;
        this.numWorkers = numWorkers;
        this.piecesFilePath = piecesFilePath;
    }

    private static Path defaultOutputDir() {
        // Plain user.home-based heuristic, not .NET's Known Folder API -- could diverge if
        // Documents is redirected (e.g. OneDrive). Kept separate from the C# solver's own
        // Documents\EternitySolutions\ so provenance of any given save file is unambiguous.
        return Path.of(System.getProperty("user.home"), "Documents", "EternitySolutions_JavaPort");
    }

    private static int defaultWorkerCount() {
        return Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
    }

    long exhaustedAtSeedCount() {
        return exhaustedAtSeedCount.get();
    }

    /** Mirrors Prepare_Pieces_And_Heuristics(). Single-threaded; call before launching a batch of workers. */
    void prepare() throws Exception {
        if (boardPieces == null) {
            boardPieces = BwUtil.getPieces(piecesFilePath);
            pieceByNumber = new BwPiece[257];
            for (BwPiece p : boardPieces) {
                pieceByNumber[p.pieceNumber()] = p;
            }
        }

        List<BwPiece> cornerPieces = boardPieces.stream().filter(p -> p.pieceType() == 2).toList();
        List<BwPiece> sidePieces = boardPieces.stream().filter(p -> p.pieceType() == 1).toList();
        List<BwPiece> middlePieces = boardPieces.stream().filter(p -> p.pieceType() == 0 && p.pieceNumber() != 139).toList();
        BwPiece startPiece = boardPieces.stream().filter(p -> p.pieceNumber() == 139).findFirst().orElseThrow();

        Random rand = new Random(); // one instance reused across this whole prepare() call, matches C#'s lifetime

        corners = buildTable(cornerPieces, false, null, rand);

        List<BwUtil.RotatedCandidate> sidesNoBreak = new ArrayList<>();
        for (BwPiece p : sidePieces) {
            sidesNoBreak.addAll(BwUtil.getRotatedPieces(p, false));
        }
        List<BwUtil.RotatedCandidate> sidesWithBreak = new ArrayList<>();
        for (BwPiece p : sidePieces) {
            sidesWithBreak.addAll(BwUtil.getRotatedPieces(p, true));
        }

        bottomSidePiecesRotated = BwUtil.groupByLeftBottom(filterRotation(sidesNoBreak, 0)); // raw, unsorted
        leftSides = BwUtil.sortAndFreezeByScore(BwUtil.groupByLeftBottom(filterRotation(sidesNoBreak, 1)), rand);
        rightSidesWithoutBreaks = BwUtil.sortAndFreezeByScore(BwUtil.groupByLeftBottom(filterRotation(sidesNoBreak, 3)), rand);
        topSides = BwUtil.sortAndFreezeByScore(BwUtil.groupByLeftBottom(filterRotation(sidesWithBreak, 2)), rand);
        rightSidesWithBreaks = BwUtil.sortAndFreezeByScore(BwUtil.groupByLeftBottom(filterRotation(sidesWithBreak, 3)), rand);

        middlesWithBreak = buildTable(middlePieces, true, null, rand);
        middlesNoBreak = buildTable(middlePieces, false, null, rand);
        southStart = buildTable(middlePieces, false, rp -> rp.topSide() == 6, rand);
        westStart = buildTable(middlePieces, false, rp -> rp.rightSide() == 11, rand);
        start = buildTable(List.of(startPiece), false, rp -> rp.rotations() == 2, rand);

        if (corners[0] == null || corners[0].length == 0) {
            throw new IllegalStateException("corners[0] is empty -- no corner piece qualifies for LeftBottom=0; step-0 seeding would fail.");
        }

        BwUtil.BoardOrder order = BwUtil.getBoardOrder();
        boardOrderRow = order.rows();
        boardOrderCol = order.cols();
        breakArray = BwUtil.getBreakArray();
        heuristicArray = BwUtil.getHeuristicArray();

        int firstBreakIndex = BwUtil.firstBreakIndex();
        masterPieceLookup = new BwRotatedPiece[256][][];
        for (int i = 0; i < 256; i++) {
            int row = boardOrderRow[i];
            int col = boardOrderCol[i];
            if (row == 15) {
                masterPieceLookup[row * 16 + col] = (col == 15 || col == 0) ? corners : topSides;
            } else if (row == 0) {
                // Deliberately left null -- row 0 handled specially in solvePuzzle() via bottomSides/corners.
            } else if (col == 15) {
                masterPieceLookup[row * 16 + col] = (i < firstBreakIndex) ? rightSidesWithoutBreaks : rightSidesWithBreaks;
            } else if (col == 0) {
                masterPieceLookup[row * 16 + col] = leftSides;
            } else if (row == 7 && col == 7) {
                masterPieceLookup[row * 16 + col] = start;
            } else if (row == 7 && col == 6) {
                masterPieceLookup[row * 16 + col] = westStart;
            } else if (row == 6 && col == 7) {
                masterPieceLookup[row * 16 + col] = southStart;
            } else {
                masterPieceLookup[row * 16 + col] = (i < firstBreakIndex) ? middlesNoBreak : middlesWithBreak;
            }
        }
    }

    private BwRotatedPiece[][] buildTable(List<BwPiece> pieces, boolean allowBreaks, Predicate<BwRotatedPiece> filter, Random rand) {
        List<BwUtil.RotatedCandidate> all = new ArrayList<>();
        for (BwPiece piece : pieces) {
            all.addAll(BwUtil.getRotatedPieces(piece, allowBreaks));
        }
        if (filter != null) {
            all = all.stream().filter(c -> filter.test(c.rotatedPiece())).toList();
        }
        return BwUtil.sortAndFreezeByScore(BwUtil.groupByLeftBottom(all), rand);
    }

    private static List<BwUtil.RotatedCandidate> filterRotation(List<BwUtil.RotatedCandidate> list, int rotation) {
        return list.stream().filter(c -> c.rotatedPiece().rotations() == rotation).toList();
    }

    public record SolveResult(int maxSolveIndex, BwRotatedPiece[] board, long nodeCount, boolean completed) {
    }

    /** Java-only guard: the general row==0,col==0 path would otherwise read board[-1] (see plan's edge-case note). */
    static boolean attemptExhausted(int solveIndex) {
        return solveIndex < 1;
    }

    SolveResult solvePuzzle() {
        return solvePuzzle(DEFAULT_NODE_CAP);
    }

    /** Mirrors SolvePuzzle(). nodeCap is injectable for bounded tests; production uses DEFAULT_NODE_CAP. */
    SolveResult solvePuzzle(long nodeCap) {
        boolean[] pieceUsed = new boolean[257];
        int[] cumulativeHeuristicSideCount = new int[256];
        int[] pieceIndexToTryNext = new int[256];
        int[] cumulativeBreaks = new int[256];
        BwRotatedPiece[] board = new BwRotatedPiece[256];
        Arrays.fill(board, BwRotatedPiece.EMPTY);

        Random rand = new Random(); // fresh per attempt

        BwRotatedPiece[][] bottomSides = BwUtil.sortAndFreezeBottomSides(bottomSidePiecesRotated, rand);

        BwRotatedPiece[] cornerZero = corners[0];
        board[0] = cornerZero[rand.nextInt(cornerZero.length)]; // uniform pick -- see plan text for why this is faithful
        pieceUsed[board[0].pieceNumber()] = true;
        cumulativeBreaks[0] = 0;
        cumulativeHeuristicSideCount[0] = board[0].heuristicSideCount();

        int solveIndex = 1;
        int maxSolveIndex = solveIndex;
        long nodeCount = 0;

        while (true) {
            nodeCount++;

            if (solveIndex > maxSolveIndex) {
                maxSolveIndex = solveIndex;
                if (maxSolveIndex >= saveThreshold) {
                    trySave(board, maxSolveIndex);
                    if (maxSolveIndex >= 256) {
                        return new SolveResult(maxSolveIndex, board, nodeCount, true);
                    }
                }
            }

            if (nodeCount > nodeCap) {
                return new SolveResult(maxSolveIndex, board, nodeCount, false);
            }

            if (attemptExhausted(solveIndex)) {
                exhaustedAtSeedCount.incrementAndGet();
                return new SolveResult(maxSolveIndex, board, nodeCount, false);
            }

            int row = boardOrderRow[solveIndex];
            int col = boardOrderCol[solveIndex];

            if (board[row * 16 + col].pieceNumber() > 0) {
                pieceUsed[board[row * 16 + col].pieceNumber()] = false;
                board[row * 16 + col] = BwRotatedPiece.EMPTY;
            }

            BwRotatedPiece[] candidates;
            if (row == 0) {
                candidates = (col < 15)
                        ? bottomSides[board[row * 16 + (col - 1)].rightSide() * 23]
                        : corners[board[row * 16 + (col - 1)].rightSide() * 23];
            } else {
                int leftSide = (col == 0) ? 0 : board[row * 16 + (col - 1)].rightSide();
                candidates = masterPieceLookup[row * 16 + col][leftSide * 23 + board[(row - 1) * 16 + col].topSide()];
            }

            boolean foundPiece = false;
            if (candidates != null) {
                int breaksThisTurn = breakArray[solveIndex] - cumulativeBreaks[solveIndex - 1];
                int tryIndex = pieceIndexToTryNext[solveIndex];

                for (int i = tryIndex; i < candidates.length; i++) {
                    if (candidates[i].breakCount() > breaksThisTurn) {
                        break;
                    }

                    if (!pieceUsed[candidates[i].pieceNumber()]) {
                        if (solveIndex <= BwUtil.MAX_HEURISTIC_INDEX) {
                            if ((cumulativeHeuristicSideCount[solveIndex - 1] + candidates[i].heuristicSideCount())
                                    < heuristicArray[solveIndex]) {
                                break; // abandons the WHOLE scan for this solveIndex, not just this candidate
                            }
                        }

                        foundPiece = true;
                        BwRotatedPiece piece = candidates[i];
                        board[row * 16 + col] = piece;
                        pieceUsed[piece.pieceNumber()] = true;
                        cumulativeBreaks[solveIndex] = cumulativeBreaks[solveIndex - 1] + piece.breakCount();
                        cumulativeHeuristicSideCount[solveIndex] = cumulativeHeuristicSideCount[solveIndex - 1] + piece.heuristicSideCount();
                        pieceIndexToTryNext[solveIndex] = i + 1;
                        solveIndex++;
                        break;
                    }
                }
            }

            if (!foundPiece) {
                pieceIndexToTryNext[solveIndex] = 0;
                solveIndex--;
            }
        }
    }

    private void trySave(BwRotatedPiece[] board, int maxSolveIndex) {
        try {
            Path saved = BwUtil.saveBoard(board, maxSolveIndex, pieceByNumber, outputDir);
            logger.info("Saved new personal best: maxSolveIndex={} -> {}", maxSolveIndex, saved);
        } catch (Exception e) {
            logger.error("Failed to save board at solveIndex={}", maxSolveIndex, e);
        }
    }

    /** Mirrors Main()'s outer while(true) + Parallel.For. */
    public void run() {
        while (true) {
            try {
                prepare();
            } catch (Exception e) {
                logger.error("prepare() failed", e);
                return;
            }
            logger.info("Tables rebuilt; launching {} workers x {} attempts.", numWorkers, ATTEMPTS_PER_WORKER_PER_BATCH);

            ExecutorService executor = Executors.newFixedThreadPool(numWorkers);
            List<Future<?>> futures = new ArrayList<>();
            for (int w = 0; w < numWorkers; w++) {
                futures.add(executor.submit(() -> {
                    for (int x = 0; x < ATTEMPTS_PER_WORKER_PER_BATCH; x++) {
                        SolveResult r = solvePuzzle();
                        logger.info("Attempt done: maxSolveIndex={} nodeCount={} completed={}",
                                r.maxSolveIndex(), r.nodeCount(), r.completed());
                    }
                }));
            }
            for (Future<?> f : futures) {
                try {
                    f.get();
                } catch (Exception e) {
                    logger.error("Worker failed", e);
                }
            }
            executor.shutdown();
            logger.info("Batch complete. exhaustedAtSeedCount so far = {}", exhaustedAtSeedCount());
        }
    }

    public static void main(String[] args) {
        new BlackwoodSolver().run();
    }
}

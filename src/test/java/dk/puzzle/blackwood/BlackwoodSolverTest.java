package dk.puzzle.blackwood;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link BlackwoodSolver}.
 *
 * <p>{@code run()}/{@code main(String[])} (unbounded loop, real file I/O
 * under the user's Documents folder) are intentionally never invoked here,
 * matching this project's established convention (see HoleSolverTest).
 * {@code solvePuzzle} is exercised directly with a small injectable node cap
 * instead.</p>
 */
class BlackwoodSolverTest {

    private static final String PIECES_PATH = "src/main/resources/JBlackwood_Pieces.txt";

    private static BlackwoodSolver solver;
    private static BwPiece[] pieceByNumber;

    @BeforeAll
    static void prepareSharedSolver() throws Exception {
        solver = new BlackwoodSolver(190, Path.of("build", "test-output"), 1, PIECES_PATH);
        solver.prepare();

        List<BwPiece> pieces = BwUtil.getPieces(PIECES_PATH);
        pieceByNumber = new BwPiece[257];
        for (BwPiece p : pieces) {
            pieceByNumber[p.pieceNumber()] = p;
        }
    }

    @SuppressWarnings("unchecked")
    private static <T> T readPrivateField(Object target, String name) throws Exception {
        Field f = BlackwoodSolver.class.getDeclaredField(name);
        f.setAccessible(true);
        return (T) f.get(target);
    }

    @Test
    void testPrepareBuildsUsableTables() throws Exception {
        BwRotatedPiece[][] corners = readPrivateField(solver, "corners");
        assertNotNull(corners[0]);
        assertTrue(corners[0].length > 0, "corners[0] must be non-empty for step-0 seeding to work");

        BwRotatedPiece[][][] masterPieceLookup = readPrivateField(solver, "masterPieceLookup");
        BwUtil.BoardOrder order = BwUtil.getBoardOrder();
        for (int step = 0; step < 256; step++) {
            int row = order.rows()[step];
            int col = order.cols()[step];
            if (row == 0) {
                assertNull(masterPieceLookup[row * 16 + col],
                        "row 0 must be left unpopulated -- handled specially via bottomSides/corners");
            } else {
                assertNotNull(masterPieceLookup[row * 16 + col],
                        "row " + row + " col " + col + " must have a dispatch table");
            }
        }
    }

    @Test
    void testSolvePuzzleBoundedRunsProduceConsistentBoards() {
        for (int attempt = 0; attempt < 3; attempt++) {
            BlackwoodSolver.SolveResult result = solver.solvePuzzle(200_000L);

            assertTrue(result.maxSolveIndex() >= 1, "must make progress past the seeded step 0");
            assertTrue(result.nodeCount() > 0);

            assertNoDuplicatePlacedPieces(result.board());
            assertGeometryConsistentWithBreakBookkeeping(result.board());
        }
    }

    private void assertNoDuplicatePlacedPieces(BwRotatedPiece[] board) {
        Set<Integer> seen = new HashSet<>();
        for (BwRotatedPiece cell : board) {
            if (cell.pieceNumber() > 0) {
                assertTrue(seen.add(cell.pieceNumber()), "piece " + cell.pieceNumber() + " placed more than once");
            }
        }
    }

    /**
     * Reconstructs each placed cell's real N/E/S/W colours from its canonical piece + rotation
     * (independently of buildBoardString, to cross-validate rather than share a bug), and checks
     * that the number of genuine edge mismatches against already-placed west/south-ish neighbours
     * exactly equals that cell's own recorded breakCount -- ground-truthing the search's break
     * bookkeeping against real geometry rather than trusting it blindly.
     */
    private void assertGeometryConsistentWithBreakBookkeeping(BwRotatedPiece[] board) {
        BwUtil.BoardOrder order = BwUtil.getBoardOrder();
        int[][] stepOf = new int[16][16];
        for (int step = 0; step < 256; step++) {
            stepOf[order.rows()[step]][order.cols()[step]] = step;
        }

        for (int row = 0; row < 16; row++) {
            for (int col = 0; col < 16; col++) {
                BwRotatedPiece cell = board[row * 16 + col];
                if (cell.pieceNumber() == 0) {
                    continue;
                }
                int[] actual = deriveActualColors(pieceByNumber[cell.pieceNumber()], cell.rotations());

                int westRequirement;
                if (col == 0) {
                    westRequirement = 0;
                } else {
                    BwRotatedPiece westCell = board[row * 16 + (col - 1)];
                    assertTrue(westCell.pieceNumber() > 0,
                            "west predecessor of placed cell (" + row + "," + col + ") must also be placed");
                    westRequirement = deriveActualColors(pieceByNumber[westCell.pieceNumber()], westCell.rotations())[1]; // neighbour's East
                }

                int southRequirement;
                if (row == 0) {
                    southRequirement = 0;
                } else {
                    BwRotatedPiece southishCell = board[(row - 1) * 16 + col];
                    assertTrue(southishCell.pieceNumber() > 0,
                            "south-ish predecessor of placed cell (" + row + "," + col + ") must also be placed");
                    southRequirement = deriveActualColors(pieceByNumber[southishCell.pieceNumber()], southishCell.rotations())[0]; // neighbour's North
                }

                int mismatchCount = (actual[3] != westRequirement ? 1 : 0) + (actual[2] != southRequirement ? 1 : 0);
                assertEquals(mismatchCount, cell.breakCount(),
                        "breakCount bookkeeping mismatch at (" + row + "," + col + "), step=" + stepOf[row][col]);
            }
        }
    }

    /** Returns {N,E,S,W} for the given canonical piece placed at the given rotation. */
    private static int[] deriveActualColors(BwPiece p, int rotation) {
        return switch (rotation) {
            case 0 -> new int[]{p.topSide(), p.rightSide(), p.bottomSide(), p.leftSide()};
            case 1 -> new int[]{p.leftSide(), p.topSide(), p.rightSide(), p.bottomSide()};
            case 2 -> new int[]{p.bottomSide(), p.leftSide(), p.topSide(), p.rightSide()};
            case 3 -> new int[]{p.rightSide(), p.bottomSide(), p.leftSide(), p.topSide()};
            default -> throw new IllegalArgumentException("unexpected rotation " + rotation);
        };
    }

    @Test
    void testAttemptExhaustedGuardsSolveIndexBelowOne() {
        assertTrue(BlackwoodSolver.attemptExhausted(0));
        assertFalse(BlackwoodSolver.attemptExhausted(1));
        assertFalse(BlackwoodSolver.attemptExhausted(255));
    }
}

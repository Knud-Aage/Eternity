package dk.puzzle.tools;

import dk.puzzle.util.PieceUtils;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for ExactZoneSolver's deterministic, bounded logic.
 *
 * <p>{@code main(String[])} is CLI glue (never invoked). The real
 * branch-and-bound search — {@code Worker.dfs}, {@code solveLevel} (which
 * spins up a real thread pool per level), {@code captureSolution}, and
 * {@code getReporterThread} (an infinite-loop daemon thread that prints on a
 * 60s timer) — is open-ended search/concurrency plumbing exactly of the kind
 * this task says to skip rather than force onto a tiny fixture. What IS
 * covered here is everything that runs BEFORE any search starts and has a
 * pure, hand-verifiable contract: the private constructor's precomputed zone
 * tables, the static zone-construction/conflict helpers, the incumbent cost
 * accounting, and {@code Worker}'s non-recursive building blocks
 * ({@code pickCell}, {@code placementCost}, {@code scanCandidates},
 * {@code place}/{@code unplace}) — all reachable via reflection since
 * {@code Worker} is a private inner class with no public contract of its
 * own.</p>
 */
class ExactZoneSolverTest {

    // Colors within CompatibilityIndex's legal palette (0 = border, 1-22 usable).
    private static final int FILLER = 1;
    private static final int SPECIAL = 9;

    /** Fully self-consistent, conflict-free 16x16 board: FILLER inward, BORDER outward. */
    private static int[] buildUniformBoard() {
        int[] board = new int[256];
        for (int row = 0; row < 16; row++) {
            for (int col = 0; col < 16; col++) {
                int n = row == 0 ? PieceUtils.BORDER_COLOR : FILLER;
                int e = col == 15 ? PieceUtils.BORDER_COLOR : FILLER;
                int s = row == 15 ? PieceUtils.BORDER_COLOR : FILLER;
                int w = col == 0 ? PieceUtils.BORDER_COLOR : FILLER;
                board[row * 16 + col] = PieceUtils.pack(n, e, s, w);
            }
        }
        return board;
    }

    /** Uniform board with a unique SPECIAL interlock between deep-interior cells 100 and 101. */
    private static int[] buildBoardWithSignatureRegion() {
        int[] board = buildUniformBoard();
        board[100] = PieceUtils.pack(FILLER, SPECIAL, FILLER, FILLER); // row6,col4 — East faces 101
        board[101] = PieceUtils.pack(FILLER, FILLER, FILLER, SPECIAL); // row6,col5 — West faces 100
        return board;
    }

    /** Same as above but cell 100 is rotated in place, breaking its match with 101 (E/W) and 116 (S/N). */
    private static int[] buildCorruptedBoard() {
        int[] board = buildBoardWithSignatureRegion();
        board[100] = PieceUtils.rotate(board[100]);
        return board;
    }

    private static boolean[] zoneOf(int... cells) {
        boolean[] zone = new boolean[256];
        for (int c : cells) zone[c] = true;
        return zone;
    }

    private static ExactZoneSolver newSolver(int[] board, boolean[] inZone) throws Exception {
        Constructor<ExactZoneSolver> ctor = ExactZoneSolver.class.getDeclaredConstructor(int[].class, boolean[].class);
        ctor.setAccessible(true);
        return ctor.newInstance(board, inZone);
    }

    private static Object getField(Object target, String name) throws Exception {
        Field f = ExactZoneSolver.class.getDeclaredField(name);
        f.setAccessible(true);
        return f.get(target);
    }

    private static int getStaticIntField(String name) throws Exception {
        Field f = ExactZoneSolver.class.getDeclaredField(name);
        f.setAccessible(true);
        return f.getInt(null);
    }

    // ------------------------------------------------------------------
    // cellClassGreys (private static)
    // ------------------------------------------------------------------

    private static int invokeCellClassGreys(int pos) throws Exception {
        Method m = ExactZoneSolver.class.getDeclaredMethod("cellClassGreys", int.class);
        m.setAccessible(true);
        return (int) m.invoke(null, pos);
    }

    @Test
    void testCellClassGreys() throws Exception {
        assertEquals(2, invokeCellClassGreys(0), "(0,0) is a corner");
        assertEquals(2, invokeCellClassGreys(15), "(0,15) is a corner");
        assertEquals(2, invokeCellClassGreys(240), "(15,0) is a corner");
        assertEquals(2, invokeCellClassGreys(255), "(15,15) is a corner");
        assertEquals(1, invokeCellClassGreys(5), "top border, non-corner");
        assertEquals(1, invokeCellClassGreys(16), "left border, non-corner");
        assertEquals(0, invokeCellClassGreys(100), "deep interior");
    }

    // ------------------------------------------------------------------
    // Private constructor: zone tables
    // ------------------------------------------------------------------

    @Test
    void testConstructorBuildsZoneCellsAndIndexMapping() throws Exception {
        ExactZoneSolver solver = newSolver(buildBoardWithSignatureRegion(), zoneOf(100, 101, 116));

        int[] zoneCells = (int[]) getField(solver, "zoneCells");
        int[] zoneIndexOf = (int[]) getField(solver, "zoneIndexOf");
        int poolSize = (int) getField(solver, "poolSize");

        assertArrayEquals(new int[]{100, 101, 116}, zoneCells, "zoneCells must be ascending");
        assertEquals(3, poolSize);
        assertEquals(0, zoneIndexOf[100]);
        assertEquals(1, zoneIndexOf[101]);
        assertEquals(2, zoneIndexOf[116]);
        assertEquals(-1, zoneIndexOf[0], "Non-zone cells must map to -1");
    }

    @Test
    void testConstructorBuildsSideTablesForMixedFixedAndDynamicNeighbors() throws Exception {
        int[] board = buildBoardWithSignatureRegion();
        ExactZoneSolver solver = newSolver(board, zoneOf(100, 101, 116));

        int SIDE_DYNAMIC = getStaticIntField("SIDE_DYNAMIC");
        int[][] sideKind = (int[][]) getField(solver, "sideKind");
        int[][] neighborZoneIdx = (int[][]) getField(solver, "neighborZoneIdx");

        // cell 100 (zone idx 0): N/W neighbours are outside the zone (fixed FILLER),
        // E neighbour is 101 (zone -> dynamic), S neighbour is 116 (zone -> dynamic).
        assertEquals(FILLER, sideKind[0][0], "cell100 North: fixed neighbour colour");
        assertEquals(SIDE_DYNAMIC, sideKind[0][1], "cell100 East: neighbour 101 is in-zone");
        assertEquals(1, neighborZoneIdx[0][1]);
        assertEquals(SIDE_DYNAMIC, sideKind[0][2], "cell100 South: neighbour 116 is in-zone");
        assertEquals(2, neighborZoneIdx[0][2]);
        assertEquals(FILLER, sideKind[0][3], "cell100 West: fixed neighbour colour");

        // cell 101 (zone idx 1): N/E/S neighbours are outside the zone, W is cell100.
        assertEquals(FILLER, sideKind[1][0]);
        assertEquals(FILLER, sideKind[1][1]);
        assertEquals(FILLER, sideKind[1][2]);
        assertEquals(SIDE_DYNAMIC, sideKind[1][3]);
        assertEquals(0, neighborZoneIdx[1][3]);

        // cell 116 (zone idx 2): N neighbour is cell100, E/S/W are outside the zone.
        assertEquals(SIDE_DYNAMIC, sideKind[2][0]);
        assertEquals(0, neighborZoneIdx[2][0]);
        assertEquals(FILLER, sideKind[2][1]);
        assertEquals(FILLER, sideKind[2][2]);
        assertEquals(FILLER, sideKind[2][3]);
    }

    @Test
    void testConstructorBuildsPoolGreysAndCellGreys() throws Exception {
        ExactZoneSolver solver = newSolver(buildBoardWithSignatureRegion(), zoneOf(100, 101, 116));

        int[] poolGreys = (int[]) getField(solver, "poolGreys");
        int[] cellGreys = (int[]) getField(solver, "cellGreys");

        assertArrayEquals(new int[]{0, 0, 0}, poolGreys, "All three pool pieces are pure interior (0 grey edges)");
        assertArrayEquals(new int[]{0, 0, 0}, cellGreys, "All three zone cells demand interior-class pieces");
    }

    @Test
    void testConstructorBuildsPoolRotationsFromOriginalBoardPieces() throws Exception {
        int[] board = buildBoardWithSignatureRegion();
        ExactZoneSolver solver = newSolver(board, zoneOf(100, 101, 116));

        int[] poolRot = (int[]) getField(solver, "poolRot");

        assertEquals(board[100], poolRot[0], "pool piece 0, rotation 0, must be the original board piece at 100");
        assertEquals(PieceUtils.rotate(board[100]), poolRot[1]);
        assertEquals(board[101], poolRot[4], "pool piece 1, rotation 0");
        assertEquals(board[116], poolRot[8], "pool piece 2, rotation 0");
    }

    @Test
    void testConstructorHandlesEmptyZoneWithoutError() throws Exception {
        ExactZoneSolver solver = newSolver(buildUniformBoard(), new boolean[256]);

        int[] zoneCells = (int[]) getField(solver, "zoneCells");
        assertEquals(0, zoneCells.length);
    }

    // ------------------------------------------------------------------
    // hasConflict (private static)
    // ------------------------------------------------------------------

    private static boolean invokeHasConflict(int[] board, int pos) throws Exception {
        Method m = ExactZoneSolver.class.getDeclaredMethod("hasConflict", int[].class, int.class);
        m.setAccessible(true);
        return (boolean) m.invoke(null, board, pos);
    }

    @Test
    void testHasConflictOnCleanBoardIsAlwaysFalse() throws Exception {
        int[] board = buildBoardWithSignatureRegion();
        for (int pos = 0; pos < 256; pos++) {
            assertFalse(invokeHasConflict(board, pos), "cell " + pos + " must be conflict-free on a valid board");
        }
    }

    @Test
    void testHasConflictDetectsMismatchAtBothEndpoints() throws Exception {
        int[] board = buildCorruptedBoard();

        assertTrue(invokeHasConflict(board, 100));
        assertTrue(invokeHasConflict(board, 101), "East/West mismatch must be visible from cell 101 too");
        assertTrue(invokeHasConflict(board, 116), "South/North mismatch must be visible from cell 116 too");
        assertFalse(invokeHasConflict(board, 99), "Cell not touching the corruption must stay clean");
    }

    // ------------------------------------------------------------------
    // zoneFromConflicts / zoneFromRows (private static)
    // ------------------------------------------------------------------

    private static boolean[] invokeZoneFromConflicts(int[] board, int radius, Set<Integer> locked) throws Exception {
        Method m = ExactZoneSolver.class.getDeclaredMethod("zoneFromConflicts", int[].class, int.class, Set.class);
        m.setAccessible(true);
        return (boolean[]) m.invoke(null, board, radius, locked);
    }

    @Test
    void testZoneFromConflictsWithZeroRadiusIncludesOnlyConflictedCells() throws Exception {
        int[] board = buildCorruptedBoard();
        boolean[] zone = invokeZoneFromConflicts(board, 0, Set.of());

        Set<Integer> expected = Set.of(100, 101, 116);
        for (int i = 0; i < 256; i++) {
            assertEquals(expected.contains(i), zone[i], "cell " + i);
        }
    }

    @Test
    void testZoneFromConflictsExcludesLockedCells() throws Exception {
        int[] board = buildCorruptedBoard();
        boolean[] zone = invokeZoneFromConflicts(board, 0, Set.of(101));

        assertTrue(zone[100]);
        assertFalse(zone[101], "Locked cells must never enter the zone even if conflicted");
        assertTrue(zone[116]);
    }

    @Test
    void testZoneFromConflictsCleanBoardProducesEmptyZone() throws Exception {
        boolean[] zone = invokeZoneFromConflicts(buildBoardWithSignatureRegion(), 2, Set.of());
        for (boolean inZone : zone) assertFalse(inZone);
    }

    private static boolean[] invokeZoneFromRows(int rowFrom, int rowTo, Set<Integer> locked) throws Exception {
        Method m = ExactZoneSolver.class.getDeclaredMethod("zoneFromRows", int.class, int.class, Set.class);
        m.setAccessible(true);
        return (boolean[]) m.invoke(null, rowFrom, rowTo, locked);
    }

    @Test
    void testZoneFromRowsIncludesFullRowBand() throws Exception {
        boolean[] zone = invokeZoneFromRows(2, 3, Set.of());
        for (int pos = 0; pos < 256; pos++) {
            int row = pos / 16;
            assertEquals(row == 2 || row == 3, zone[pos], "cell " + pos);
        }
    }

    @Test
    void testZoneFromRowsExcludesLockedCells() throws Exception {
        boolean[] zone = invokeZoneFromRows(4, 4, Set.of(4 * 16 + 7));

        assertFalse(zone[4 * 16 + 7]);
        assertTrue(zone[4 * 16 + 0]);
        assertTrue(zone[4 * 16 + 15]);
    }

    // ------------------------------------------------------------------
    // incumbentZoneCost / rebuildFullBoard (private instance methods)
    // ------------------------------------------------------------------

    @Test
    void testIncumbentZoneCostIsZeroOnCleanBoard() throws Exception {
        int[] board = buildBoardWithSignatureRegion();
        ExactZoneSolver solver = newSolver(board, zoneOf(100, 101, 116));
        Method m = ExactZoneSolver.class.getDeclaredMethod("incumbentZoneCost", int[].class);
        m.setAccessible(true);

        assertEquals(0, (int) m.invoke(solver, (Object) board));
    }

    @Test
    void testIncumbentZoneCostCountsEachMismatchOnce() throws Exception {
        int[] board = buildCorruptedBoard();
        ExactZoneSolver solver = newSolver(board, zoneOf(100, 101, 116));
        Method m = ExactZoneSolver.class.getDeclaredMethod("incumbentZoneCost", int[].class);
        m.setAccessible(true);

        assertEquals(2, (int) m.invoke(solver, (Object) board),
                "Exactly the 100-101 and 100-116 mismatches, each counted once");
    }

    @Test
    void testRebuildFullBoardOverlaysSolutionOntoFixedBoard() throws Exception {
        int[] board = buildBoardWithSignatureRegion();
        ExactZoneSolver solver = newSolver(board, zoneOf(100, 101));

        Field solutionField = ExactZoneSolver.class.getDeclaredField("solution");
        solutionField.setAccessible(true);
        solutionField.set(solver, new int[]{999, 888});

        Method m = ExactZoneSolver.class.getDeclaredMethod("rebuildFullBoard");
        m.setAccessible(true);
        int[] full = (int[]) m.invoke(solver);

        assertEquals(999, full[100]);
        assertEquals(888, full[101]);
        assertEquals(board[50], full[50], "Cells outside the zone must be unchanged from the original board");
    }

    // ------------------------------------------------------------------
    // Worker (private inner class, reached via reflection)
    // ------------------------------------------------------------------

    private static Class<?> workerClass() throws Exception {
        return Class.forName("dk.puzzle.tools.ExactZoneSolver$Worker");
    }

    private static Object newWorker(ExactZoneSolver solver, int budget, long deadline) throws Exception {
        Constructor<?> ctor = workerClass().getDeclaredConstructor(ExactZoneSolver.class, int.class, long.class);
        ctor.setAccessible(true);
        return ctor.newInstance(solver, budget, deadline);
    }

    private static long farFutureDeadline() {
        return System.nanoTime() + TimeUnit.MINUTES.toNanos(10);
    }

    @Test
    void testWorkerPickCellChoosesMostConstrainedZoneCell() throws Exception {
        ExactZoneSolver solver = newSolver(buildBoardWithSignatureRegion(), zoneOf(100, 101, 116));
        Object worker = newWorker(solver, 4, farFutureDeadline());

        Method pickCell = workerClass().getDeclaredMethod("pickCell");
        pickCell.setAccessible(true);
        int cell = (int) pickCell.invoke(worker);

        // cell100 has 2 fixed/border sides known (N,W) — its other two sides
        // are dynamic (in-zone) neighbours. cell101 and cell116 each have 3
        // known sides. pickCell keeps the FIRST cell reaching the max (strict
        // >), so scanning zoneCells in order [100,101,116] lands on 101.
        assertEquals(101, cell);
    }

    @Test
    void testWorkerPlacementCostHardGreyViolationAndMismatchCounting() throws Exception {
        int[] board = buildUniformBoard();
        ExactZoneSolver solver = newSolver(board, zoneOf(0)); // corner cell: N and W are hard frame greys
        Object worker = newWorker(solver, 4, farFutureDeadline());

        Method placementCost = workerClass().getDeclaredMethod("placementCost", int.class, int.class);
        placementCost.setAccessible(true);

        int correctPiece = board[0]; // pack(BORDER, FILLER, FILLER, BORDER)
        assertEquals(0, (int) placementCost.invoke(worker, 0, correctPiece));

        int hardViolation = PieceUtils.pack(5, FILLER, FILLER, PieceUtils.BORDER_COLOR); // North must be border
        assertEquals(Integer.MAX_VALUE, (int) placementCost.invoke(worker, 0, hardViolation),
                "A non-grey edge facing off-board is a hard (frame) violation");

        int oneMismatch = PieceUtils.pack(PieceUtils.BORDER_COLOR, 7, FILLER, PieceUtils.BORDER_COLOR); // East wrong
        assertEquals(1, (int) placementCost.invoke(worker, 0, oneMismatch),
                "A single wrong soft (fixed-neighbour) edge costs exactly 1");
    }

    @Test
    void testWorkerScanCandidatesIsBucketSortedByCostAndFindsAZeroCostFit() throws Exception {
        ExactZoneSolver solver = newSolver(buildBoardWithSignatureRegion(), zoneOf(100, 101, 116));
        Object worker = newWorker(solver, 4, farFutureDeadline());

        Method scanCandidates = workerClass().getDeclaredMethod("scanCandidates",
                int.class, int.class, int.class, int.class);
        scanCandidates.setAccessible(true);
        int count = (int) scanCandidates.invoke(worker, 101, 4, 0, 0);

        assertTrue(count > 0, "The pool must offer at least one candidate for cell 101");

        Field candCostField = workerClass().getDeclaredField("candCost");
        candCostField.setAccessible(true);
        int[][] candCost = (int[][]) candCostField.get(worker);

        assertEquals(0, candCost[0][0],
                "Piece 101's own original orientation matches all 3 fixed neighbours, so a 0-cost fit must exist");
        for (int i = 1; i < count; i++) {
            assertTrue(candCost[0][i - 1] <= candCost[0][i], "Bucket sort must be non-decreasing by cost");
        }
    }

    @Test
    void testWorkerPlaceAndUnplaceRoundTrip() throws Exception {
        ExactZoneSolver solver = newSolver(buildBoardWithSignatureRegion(), zoneOf(100, 101, 116));
        Object worker = newWorker(solver, 4, farFutureDeadline());

        Method place = workerClass().getDeclaredMethod("place", int.class, int.class, int.class);
        place.setAccessible(true);
        Method unplace = workerClass().getDeclaredMethod("unplace", int.class, int.class);
        unplace.setAccessible(true);
        Field boardField = workerClass().getDeclaredField("board");
        boardField.setAccessible(true);
        Field usedField = workerClass().getDeclaredField("used");
        usedField.setAccessible(true);

        place.invoke(worker, 101, 12345, 1); // cell=101, piece=12345, poolIdx=1 (pool slot for cell 101)

        assertEquals(12345, ((int[]) boardField.get(worker))[101]);
        assertTrue(((boolean[]) usedField.get(worker))[1]);

        unplace.invoke(worker, 101, 1);

        assertEquals(-1, ((int[]) boardField.get(worker))[101]);
        assertFalse(((boolean[]) usedField.get(worker))[1]);
    }
}

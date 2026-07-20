package dk.puzzle.ai;

import dk.puzzle.util.PieceUtils;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for ConflictAnnealer's deterministic building blocks: conflict/
 * border counting, board validation, class-normalisation and frame-orientation
 * repair, and the incremental cost/energy helpers that drive the simulated
 * annealer.
 *
 * <p>ConflictAnnealer has no {@code main(String[])} — it is a library class
 * consumed by other solving code, not a standalone CLI. Its constructor is
 * private (invoked here via reflection) and it has no PieceInventory
 * dependency: it operates purely on raw packed-piece board arrays via
 * {@link PieceUtils}.</p>
 *
 * <p>The stochastic core of the annealer (the multi-threaded {@code runWalker}
 * SA loop, {@code rotationMove}/{@code swapMove}, {@code lnsKick}, and the
 * branch-and-bound DFS inside {@code bnbWindowSearch}) is intentionally not
 * exercised move-by-move here: those methods are randomised local search
 * whose interesting behaviour only emerges over many thousands of moves, and
 * asserting on their outcome without either a fixed RNG contract or a very
 * long run would be flaky or slow. Instead, the tests below (a) verify every
 * deterministic helper those loops are built from (energy/cost accounting,
 * acceptance criterion, adjacency/class classification, window cost, class
 * normalisation, frame orientation) directly, and (b) verify the public
 * {@link ConflictAnnealer#optimize} entry point's documented contracts
 * (input validation, and the already-optimal short-circuit which is fast and
 * fully deterministic) end-to-end.</p>
 */
class ConflictAnnealerTest {

    private static final int W = 16;

    // -----------------------------------------------------------------------
    // Perfect-board test fixture
    // -----------------------------------------------------------------------

    // Internal edge colour ranges kept disjoint from each other and from
    // PieceUtils.BORDER_COLOR (0), and safely within the 8-bit edge field.
    private static final int H_BASE = 10;  // horizontal (row-boundary) edges -> [10,109]
    private static final int V_BASE = 130; // vertical (col-boundary) edges  -> [130,229]

    private static int hEdge(int r, int c) {
        return H_BASE + ((r * 16 + c) % 100);
    }

    private static int vEdge(int r, int c) {
        return V_BASE + ((r * 16 + c) % 100);
    }

    private static int pieceAt(int r, int c) {
        int n = (r == 0) ? PieceUtils.BORDER_COLOR : hEdge(r - 1, c);
        int s = (r == 15) ? PieceUtils.BORDER_COLOR : hEdge(r, c);
        int w = (c == 0) ? PieceUtils.BORDER_COLOR : vEdge(r, c - 1);
        int e = (c == 15) ? PieceUtils.BORDER_COLOR : vEdge(r, c);
        return PieceUtils.pack(n, e, s, w);
    }

    /**
     * Builds a fully self-consistent 256-cell board: every internal edge
     * matches its neighbour by construction, every frame edge is
     * PieceUtils.BORDER_COLOR, and every cell's grey-edge count matches its
     * geometric class (corner=2, border=1, interior=0). Zero internal
     * conflicts and zero border violations, guaranteed independent of the
     * exact (non-zero, disjoint) colour ids chosen above.
     */
    private static int[] buildPerfectBoard() {
        int[] board = new int[256];
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                board[r * 16 + c] = pieceAt(r, c);
            }
        }
        return board;
    }

    // -----------------------------------------------------------------------
    // Reflection helpers
    // -----------------------------------------------------------------------

    private ConflictAnnealer newAnnealer(Set<Integer> lockedCells) throws Exception {
        Constructor<ConflictAnnealer> ctor = ConflictAnnealer.class.getDeclaredConstructor(Set.class);
        ctor.setAccessible(true);
        return ctor.newInstance(lockedCells);
    }

    private Method privateMethod(String name, Class<?>... types) throws Exception {
        Method m = ConflictAnnealer.class.getDeclaredMethod(name, types);
        m.setAccessible(true);
        return m;
    }

    private int cellClass(int pos) throws Exception {
        return (int) privateMethod("cellClass", int.class).invoke(null, pos);
    }

    private boolean areAdjacent(int a, int b) throws Exception {
        return (boolean) privateMethod("areAdjacent", int.class, int.class).invoke(null, a, b);
    }

    // -----------------------------------------------------------------------
    // countInternalConflicts / countBorderViolations (public static)
    // -----------------------------------------------------------------------

    @Test
    void testCountInternalConflictsZeroOnPerfectBoard() {
        assertEquals(0, ConflictAnnealer.countInternalConflicts(buildPerfectBoard()));
    }

    @Test
    void testCountInternalConflictsDetectsHorizontalAndVerticalMismatches() {
        int[] board = buildPerfectBoard();
        // Corrupt cell 0's East and South edges: breaks the shared edge with
        // its East neighbour (cell 1) and its South neighbour (cell 16).
        board[0] = PieceUtils.pack(PieceUtils.getNorth(board[0]), 250, 251, PieceUtils.getWest(board[0]));
        assertEquals(2, ConflictAnnealer.countInternalConflicts(board));
    }

    @Test
    void testCountInternalConflictsIgnoresHoles() {
        int[] board = buildPerfectBoard();
        board[1] = -1;
        assertEquals(0, ConflictAnnealer.countInternalConflicts(board),
                "Cells adjacent to a hole must not be scored as conflicts");
    }

    @Test
    void testCountBorderViolationsZeroOnPerfectBoard() {
        assertEquals(0, ConflictAnnealer.countBorderViolations(buildPerfectBoard()));
    }

    @Test
    void testCountBorderViolationsDetectsNonGreyFrameEdges() {
        int[] board = buildPerfectBoard();
        int corner = board[0]; // (0,0): North and West must be border-facing greys
        board[0] = PieceUtils.pack(77, PieceUtils.getEast(corner), PieceUtils.getSouth(corner), PieceUtils.getWest(corner));
        assertEquals(1, ConflictAnnealer.countBorderViolations(board));
    }

    // -----------------------------------------------------------------------
    // optimize (public static) - validation and deterministic short-circuit
    // -----------------------------------------------------------------------

    @Test
    void testOptimizeThrowsOnBoardContainingHole() {
        int[] board = buildPerfectBoard();
        board[5] = -1;
        assertThrows(IllegalArgumentException.class,
                () -> ConflictAnnealer.optimize(board, null, 10, 1, null));
    }

    @Test
    void testOptimizeThrowsOnBoardContainingSurgeonHole() {
        int[] board = buildPerfectBoard();
        board[5] = -2;
        assertThrows(IllegalArgumentException.class,
                () -> ConflictAnnealer.optimize(board, null, 10, 1, null));
    }

    @Test
    void testOptimizeOnAlreadyPerfectBoardReturnsImmediatelyWithZeroConflicts() {
        int[] board = buildPerfectBoard();

        ConflictAnnealer.Result result = ConflictAnnealer.optimize(board, null, 200, 1, null);

        assertEquals(0, result.internalConflicts());
        assertEquals(0, result.borderConflicts());
        assertArrayEquals(board, result.board(), "A board that is already conflict-free must come back unchanged");
        assertEquals(0, result.moves(), "No SA cycle should run when the board starts at zero energy");
        assertEquals(0, result.lnsKicks());
        assertEquals(0, result.lnsWins());
    }

    @Test
    void testOptimizeOnAlreadyPerfectBoardConvergesWithMultipleWalkers() {
        int[] board = buildPerfectBoard();

        ConflictAnnealer.Result result = ConflictAnnealer.optimize(board, null, 200, 4, null);

        assertEquals(0, result.internalConflicts());
        assertEquals(0, result.borderConflicts());
    }

    @Test
    void testOptimizeAcceptsLockedCellsWithoutError() {
        int[] board = buildPerfectBoard();
        Set<Integer> locked = new HashSet<>(List.of(0, 255));

        ConflictAnnealer.Result result = ConflictAnnealer.optimize(board, locked, 50, 1, null);

        assertEquals(0, result.internalConflicts());
    }

    // -----------------------------------------------------------------------
    // normalize (public static) - class repair + frame orientation
    // -----------------------------------------------------------------------

    @Test
    void testNormalizeRepairsCornerPieceStrandedInInterior() {
        int[] board = buildPerfectBoard();
        int cornerPos = 0;    // (0,0), a corner cell
        int interiorPos = 85; // (5,5), an interior cell
        int tmp = board[cornerPos];
        board[cornerPos] = board[interiorPos];
        board[interiorPos] = tmp;

        ConflictAnnealer.normalize(board, null);

        assertArrayEquals(buildPerfectBoard(), board,
                "normalize must restore the original class-clean board for a single stranded corner/interior swap");
    }

    @Test
    void testNormalizeRestoresGreyOutwardOrientationOfARotatedFramePiece() {
        int[] board = buildPerfectBoard();
        int original = board[0];
        board[0] = PieceUtils.rotate(original); // misorients the corner piece (class unchanged)

        ConflictAnnealer.normalize(board, null);

        assertEquals(original, board[0], "orientFrame must find the rotation whose greys face the two off-board sides");
    }

    // -----------------------------------------------------------------------
    // cellClass / areAdjacent (private static)
    // -----------------------------------------------------------------------

    @Test
    void testCellClassIdentifiesCornersBordersAndInterior() throws Exception {
        assertEquals(0, cellClass(0), "Position 0 (0,0) is a corner");
        assertEquals(0, cellClass(255), "Position 255 (15,15) is a corner");
        assertEquals(1, cellClass(5), "Position 5 (0,5) is a border cell");
        assertEquals(2, cellClass(85), "Position 85 (5,5) is an interior cell");
    }

    @Test
    void testAreAdjacentDetectsHorizontalAndVerticalNeighboursOnly() throws Exception {
        assertTrue(areAdjacent(0, 1), "Same-row neighbours are adjacent");
        assertTrue(areAdjacent(0, 16), "Same-column neighbours one row apart are adjacent");
        assertFalse(areAdjacent(0, 17), "Diagonal cells are not adjacent");
        assertFalse(areAdjacent(15, 16), "Wrap-around across a row boundary must not count as adjacent");
    }

    // -----------------------------------------------------------------------
    // energy / cellCost / pairCost (private instance)
    // -----------------------------------------------------------------------

    @Test
    void testEnergyCombinesInternalAndWeightedBorderConflicts() throws Exception {
        ConflictAnnealer annealer = newAnnealer(null);
        int[] board = buildPerfectBoard();
        board[0] = PieceUtils.pack(77, PieceUtils.getEast(board[0]), PieceUtils.getSouth(board[0]), PieceUtils.getWest(board[0]));

        int e = (int) privateMethod("energy", int[].class).invoke(annealer, (Object) board);

        assertEquals(2, e, "energy = internalConflicts(0) + BORDER_WEIGHT(2) * borderConflicts(1)");
    }

    @Test
    void testCellCostCountsBorderWeightedAndInternalMismatches() throws Exception {
        ConflictAnnealer annealer = newAnnealer(null);
        int[] board = buildPerfectBoard();
        int corner = board[0];
        // Break both the North border edge and the East internal edge of cell 0.
        board[0] = PieceUtils.pack(77, 250, PieceUtils.getSouth(corner), PieceUtils.getWest(corner));

        int cost = (int) privateMethod("cellCost", int[].class, int.class).invoke(annealer, board, 0);

        assertEquals(3, cost, "1 broken border edge (weight 2) + 1 broken internal edge (weight 1) = 3");
    }

    @Test
    void testPairCostSubtractsSharedEdgeOnceWhenCellsAreAdjacent() throws Exception {
        ConflictAnnealer annealer = newAnnealer(null);
        int[] board = buildPerfectBoard();
        // Corrupt the shared East/West edge between adjacent cells 0 and 1.
        board[0] = PieceUtils.pack(PieceUtils.getNorth(board[0]), 250, PieceUtils.getSouth(board[0]), PieceUtils.getWest(board[0]));

        Method cellCostM = privateMethod("cellCost", int[].class, int.class);
        Method pairCostM = privateMethod("pairCost", int[].class, int.class, int.class, boolean.class);

        int cost0 = (int) cellCostM.invoke(annealer, board, 0); // counts the broken edge from the East side
        int cost1 = (int) cellCostM.invoke(annealer, board, 1); // counts the SAME broken edge from the West side
        int pairAdjacent = (int) pairCostM.invoke(annealer, board, 0, 1, true);

        assertEquals(cost0 + cost1 - 1, pairAdjacent,
                "The shared broken edge is double-counted by the two cellCost calls and must be subtracted once");
    }

    // -----------------------------------------------------------------------
    // accept (private instance) - SA acceptance criterion
    // -----------------------------------------------------------------------

    @Test
    void testAcceptAlwaysTakesNonPositiveDeltaMoves() throws Exception {
        ConflictAnnealer annealer = newAnnealer(null);
        Method m = privateMethod("accept", int.class, double.class, Random.class);
        assertTrue((boolean) m.invoke(annealer, 0, 0.5, new Random(1)));
        assertTrue((boolean) m.invoke(annealer, -3, 0.5, new Random(1)));
    }

    @Test
    void testAcceptRejectsLargeUphillMovesAtLowTemperature() throws Exception {
        ConflictAnnealer annealer = newAnnealer(null);
        Method m = privateMethod("accept", int.class, double.class, Random.class);
        // exp(-50/0.001) is astronomically small; any Random draw in [0,1) rejects it.
        assertFalse((boolean) m.invoke(annealer, 50, 0.001, new Random(1)));
    }

    // -----------------------------------------------------------------------
    // requiredColor (private instance)
    // -----------------------------------------------------------------------

    @Test
    void testRequiredColorReturnsBorderColorOffBoard() throws Exception {
        ConflictAnnealer annealer = newAnnealer(null);
        int[] board = buildPerfectBoard();
        int result = (int) privateMethod("requiredColor", int[].class, int.class, int.class)
                .invoke(annealer, board, 0, -W); // cell 0's north offset is off-board

        assertEquals(PieceUtils.BORDER_COLOR, result);
    }

    @Test
    void testRequiredColorReturnsNeighboursOpposingColourWhenPlaced() throws Exception {
        ConflictAnnealer annealer = newAnnealer(null);
        int[] board = buildPerfectBoard();
        // Cell 17's north neighbour is cell 1; requiredColor toward north must equal cell 1's South edge.
        int result = (int) privateMethod("requiredColor", int[].class, int.class, int.class)
                .invoke(annealer, board, 17, -W);

        assertEquals(PieceUtils.getSouth(board[1]), result);
    }

    @Test
    void testRequiredColorReturnsWildcardForUnfilledNeighbour() throws Exception {
        ConflictAnnealer annealer = newAnnealer(null);
        int[] board = buildPerfectBoard();
        board[1] = -1; // simulate an unfilled window cell north of 17
        int result = (int) privateMethod("requiredColor", int[].class, int.class, int.class)
                .invoke(annealer, board, 17, -W);

        assertEquals(-1, result);
    }

    // -----------------------------------------------------------------------
    // windowArrangementCost (private instance)
    // -----------------------------------------------------------------------

    @Test
    void testWindowArrangementCostZeroForCleanPerfectSubwindow() throws Exception {
        ConflictAnnealer annealer = newAnnealer(null);
        int[] board = buildPerfectBoard();
        int[] window = {0, 1, 16, 17}; // a clean 2x2 block, fully surrounded by matching pieces

        int cost = (int) privateMethod("windowArrangementCost", int[].class, int[].class).invoke(annealer, board, window);

        assertEquals(0, cost);
    }

    @Test
    void testWindowArrangementCostDetectsBoundaryMismatch() throws Exception {
        ConflictAnnealer annealer = newAnnealer(null);
        int[] board = buildPerfectBoard();
        int[] window = {0, 1, 16, 17};
        // Break the edge between window cell 17 and its out-of-window East neighbour (18).
        board[17] = PieceUtils.pack(PieceUtils.getNorth(board[17]), 250, PieceUtils.getSouth(board[17]), PieceUtils.getWest(board[17]));

        int cost = (int) privateMethod("windowArrangementCost", int[].class, int[].class).invoke(annealer, board, window);

        assertEquals(1, cost);
    }

    // -----------------------------------------------------------------------
    // collectConflictCells (private instance)
    // -----------------------------------------------------------------------

    @Test
    void testCollectConflictCellsFindsExactlyTheCorruptedCells() throws Exception {
        ConflictAnnealer annealer = newAnnealer(null);
        int[] board = buildPerfectBoard();
        board[0] = PieceUtils.pack(PieceUtils.getNorth(board[0]), 250, PieceUtils.getSouth(board[0]), PieceUtils.getWest(board[0]));

        int[] out = new int[256];
        int n = (int) privateMethod("collectConflictCells", int[].class, int[].class).invoke(annealer, board, out);

        Set<Integer> found = new HashSet<>();
        for (int i = 0; i < n; i++) found.add(out[i]);
        assertEquals(Set.of(0, 1), found, "Both sides of the broken East/West edge must be reported as conflicted");
    }

    // -----------------------------------------------------------------------
    // bnbWindowSearch (private instance) - already-optimal short-circuit
    // -----------------------------------------------------------------------

    @Test
    void testBnbWindowSearchReturnsFalseWhenWindowAlreadyOptimal() throws Exception {
        ConflictAnnealer annealer = newAnnealer(null);
        int[] board = buildPerfectBoard();
        int[] window = {0, 1, 16, 17};
        int[] before = Arrays.copyOf(board, 256);

        boolean improved = (boolean) privateMethod("bnbWindowSearch", int[].class, int[].class, Random.class, long.class)
                .invoke(annealer, board, window, new Random(1), System.nanoTime() + 1_000_000_000L);

        assertFalse(improved, "A window that is already conflict-free cannot be strictly improved");
        assertArrayEquals(before, board, "bnbWindowSearch must leave an already-optimal window untouched");
    }

    // -----------------------------------------------------------------------
    // ruinAndRecreate (private instance) - permutation invariant
    // -----------------------------------------------------------------------

    @Test
    void testRuinAndRecreatePreservesPieceMultisetAndFillsEveryZoneCell() throws Exception {
        ConflictAnnealer annealer = newAnnealer(null);
        int[] board = buildPerfectBoard();
        // Introduce one conflict so ruinAndRecreate has a zone to rebuild.
        board[85] = PieceUtils.pack(PieceUtils.getNorth(board[85]), 250, PieceUtils.getSouth(board[85]), PieceUtils.getWest(board[85]));

        int[] before = Arrays.copyOf(board, 256);
        privateMethod("ruinAndRecreate", int[].class, Random.class).invoke(annealer, board, new Random(3));

        int[] sortedBefore = Arrays.copyOf(before, 256);
        int[] sortedAfter = Arrays.copyOf(board, 256);
        Arrays.sort(sortedBefore);
        Arrays.sort(sortedAfter);
        assertArrayEquals(sortedBefore, sortedAfter,
                "ruinAndRecreate only rearranges existing pieces; it must not create, drop, or duplicate any");

        for (int pos = 0; pos < 256; pos++) {
            assertNotEquals(-1, board[pos], "Every cell must end up filled after the ruin-and-recreate refill");
        }
    }
}

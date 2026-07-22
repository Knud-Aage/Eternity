package dk.puzzle.ai;

import dk.puzzle.model.PieceInventory;
import dk.puzzle.util.PieceUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;

/**
 * Unit tests for ConflictReducer's local-search conflict-repair logic.
 *
 * <p>The PieceInventory dependency is a Mockito mock whose allOrientations/
 * physicalMapping arrays are populated by hand (the same pattern used in
 * EternitySolverTest), starting from an all-zero "filler" piece for every
 * physical id and then overwriting specific physical ids with real,
 * distinguishable colours via {@link #setPhysicalPiece} where a test needs
 * meaningful rotation/matching behaviour.</p>
 */
class ConflictReducerTest {

    private PieceInventory mockInventory;

    @BeforeEach
    void setUp() {
        mockInventory = mock(PieceInventory.class);
        mockInventory.allOrientations = new int[1024];
        mockInventory.physicalMapping = new int[1024];
        Arrays.fill(mockInventory.allOrientations, 0); // all-default "empty" filler pieces
        for (int i = 0; i < 1024; i++) {
            mockInventory.physicalMapping[i] = i / 4;
        }
    }

    /** Overwrites physical piece {@code physId}'s 4 orientation slots with the 4 rotations of {@code base}. */
    private void setPhysicalPiece(int physId, int base) {
        int oriented = base;
        for (int r = 0; r < 4; r++) {
            mockInventory.allOrientations[physId * 4 + r] = oriented;
            mockInventory.physicalMapping[physId * 4 + r] = physId;
            oriented = PieceUtils.rotate(oriented);
        }
    }

    private int[] emptyBoard() {
        int[] board = new int[256];
        Arrays.fill(board, -1);
        return board;
    }

    // -----------------------------------------------------------------------
    // countConflicts
    // -----------------------------------------------------------------------

    @Test
    void testCountConflictsEmptyBoardIsZero() {
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);
        assertEquals(0, reducer.countConflicts(emptyBoard()));
    }

    @Test
    void testCountConflictsMatchingAdjacentPiecesIsZero() {
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);
        int[] board = emptyBoard();
        // Interior cells (row1/col1, row1/col2, row2/col1) so the frame-violation
        // check below can't contribute — this test is about internal mismatches only.
        board[17] = PieceUtils.pack(1, 2, 3, 4);
        board[18] = PieceUtils.pack(5, 6, 7, 2);   // West=2 matches East of index 17
        board[33] = PieceUtils.pack(3, 8, 9, 10);  // North=3 matches South of index 17
        assertEquals(0, reducer.countConflicts(board));
    }

    @Test
    void testCountConflictsCountsHorizontalAndVerticalMismatches() {
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);
        int[] board = emptyBoard();
        board[17] = PieceUtils.pack(1, 2, 3, 4);
        board[18] = PieceUtils.pack(5, 6, 7, 9);   // West=9 mismatches East=2 -> 1 conflict
        board[33] = PieceUtils.pack(9, 8, 9, 10);  // North=9 mismatches South=3 -> 1 conflict
        assertEquals(2, reducer.countConflicts(board));
    }

    @Test
    void testCountConflictsIgnoresSurgeonHoles() {
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);
        int[] board = emptyBoard();
        board[17] = PieceUtils.pack(1, 2, 3, 4);
        board[18] = -2; // surgeon hole; would mismatch (West unknown) but must be skipped
        assertEquals(0, reducer.countConflicts(board));
    }

    @Test
    void testCountConflictsDetectsFrameViolationOnBorderCell() {
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);
        int[] board = emptyBoard();
        // Row 0: North edge must be BORDER_COLOR (0). This piece's North=1 is a
        // frame violation even though it has no neighbor to mismatch against.
        board[5] = PieceUtils.pack(1, 0, 0, 0);
        assertEquals(1, reducer.countConflicts(board),
                "A non-border-colored outward edge on a border cell must count as a conflict");
    }

    @Test
    void testCountConflictsAcceptsCorrectBorderColorOnBorderCell() {
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);
        int[] board = emptyBoard();
        board[5] = PieceUtils.pack(0, 7, 8, 9); // North=0 (BORDER_COLOR) -> no frame violation
        assertEquals(0, reducer.countConflicts(board));
    }

    @Test
    void testCountConflictsDetectsBothFrameViolationsAtACorner() {
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);
        int[] board = emptyBoard();
        // Index 0 is row0/col0: both North and West must be BORDER_COLOR.
        board[0] = PieceUtils.pack(1, 5, 6, 4);
        assertEquals(2, reducer.countConflicts(board),
                "A corner cell violating both outward edges must count as 2 conflicts");
    }

    // -----------------------------------------------------------------------
    // scoreConflicts (private)
    // -----------------------------------------------------------------------

    @Test
    void testScoreConflictsAssignsScoreToBothSidesOfAMismatch() throws Exception {
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);
        int[] board = emptyBoard();
        board[17] = PieceUtils.pack(1, 2, 3, 4);    // East=2
        board[18] = PieceUtils.pack(5, 6, 7, 9);    // West=9 mismatches East(17)=2
        board[19] = PieceUtils.pack(20, 21, 22, 6); // West=6 matches East(18)=6 -> no conflict here

        Method m = ConflictReducer.class.getDeclaredMethod("scoreConflicts", int[].class);
        m.setAccessible(true);
        int[] score = (int[]) m.invoke(reducer, (Object) board);

        assertEquals(1, score[17], "Position 17 must be credited with its East mismatch");
        assertEquals(1, score[18], "Position 18 must be credited with its West mismatch");
        assertEquals(0, score[19], "Position 19 has no mismatches and must score 0");
    }

    @Test
    void testScoreConflictsCreditsFrameViolationToItsOwnCell() throws Exception {
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);
        int[] board = emptyBoard();
        board[5] = PieceUtils.pack(1, 0, 0, 0); // row0: North=1 is a frame violation

        Method m = ConflictReducer.class.getDeclaredMethod("scoreConflicts", int[].class);
        m.setAccessible(true);
        int[] score = (int[]) m.invoke(reducer, (Object) board);

        assertEquals(1, score[5], "The bordering cell itself must be credited with its own frame violation");
    }

    // -----------------------------------------------------------------------
    // isLocked (private)
    // -----------------------------------------------------------------------

    @Test
    void testIsLockedTrueForCenterAndHintsWhenLockCenterEnabled() throws Exception {
        ConflictReducer reducer = new ConflictReducer(mockInventory, true);
        Method m = ConflictReducer.class.getDeclaredMethod("isLocked", int.class);
        m.setAccessible(true);
        for (int pos : new int[]{135, 221, 45, 210, 34}) {
            assertTrue((boolean) m.invoke(reducer, pos), "Position " + pos + " must be locked");
        }
        assertFalse((boolean) m.invoke(reducer, 0), "Unrelated position must not be locked");
    }

    @Test
    void testIsLockedAlwaysFalseWhenLockCenterDisabled() throws Exception {
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);
        Method m = ConflictReducer.class.getDeclaredMethod("isLocked", int.class);
        m.setAccessible(true);
        for (int pos : new int[]{135, 221, 45, 210, 34, 0}) {
            assertFalse((boolean) m.invoke(reducer, pos), "Position " + pos + " must not be locked when lockCenter=false");
        }
    }

    // -----------------------------------------------------------------------
    // isPlaced / countPieces (private)
    // -----------------------------------------------------------------------

    @Test
    void testIsPlacedHelper() throws Exception {
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);
        Method m = ConflictReducer.class.getDeclaredMethod("isPlaced", int.class);
        m.setAccessible(true);
        assertFalse((boolean) m.invoke(reducer, -1));
        assertFalse((boolean) m.invoke(reducer, -2));
        assertTrue((boolean) m.invoke(reducer, PieceUtils.pack(1, 2, 3, 4)));
    }

    @Test
    void testCountPiecesPrivateHelperIgnoresHolesAndSurgeonHoles() throws Exception {
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);
        Method m = ConflictReducer.class.getDeclaredMethod("countPieces", int[].class);
        m.setAccessible(true);
        int[] board = emptyBoard();
        board[10] = 500;
        board[20] = 600;
        board[30] = -2;
        assertEquals(2, (int) m.invoke(reducer, (Object) board));
    }

    // -----------------------------------------------------------------------
    // getPhysId (private)
    // -----------------------------------------------------------------------

    @Test
    void testGetPhysIdReturnsMappedPhysicalIdForKnownOrientation() throws Exception {
        setPhysicalPiece(7, PieceUtils.pack(11, 22, 33, 44));
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);
        Method m = ConflictReducer.class.getDeclaredMethod("getPhysId", int.class);
        m.setAccessible(true);
        int orientedValue = mockInventory.allOrientations[7 * 4 + 2]; // one of its rotations
        assertEquals(7, (int) m.invoke(reducer, orientedValue));
    }

    @Test
    void testGetPhysIdReturnsNegativeOneForUnknownPiece() throws Exception {
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);
        Method m = ConflictReducer.class.getDeclaredMethod("getPhysId", int.class);
        m.setAccessible(true);
        // A value that appears nowhere in allOrientations (default entries are all 0)
        int unknown = PieceUtils.pack(99, 98, 97, 96);
        assertEquals(-1, (int) m.invoke(reducer, unknown));
    }

    // -----------------------------------------------------------------------
    // rotationPass (private) - Strategy 1
    // -----------------------------------------------------------------------

    @Test
    void testRotationPassChoosesOrientationThatMinimisesConflicts() throws Exception {
        int base = PieceUtils.pack(1, 2, 3, 4); // N=1,E=2,S=3,W=4
        setPhysicalPiece(3, base);
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);

        int[] board = emptyBoard();
        // Interior cells: none of this piece's rotations ever expose a
        // BORDER_COLOR edge, so testing this on an actual border cell would
        // make the conflict unresolvable by rotation alone regardless of
        // the East/West match this test is actually about.
        board[17] = mockInventory.allOrientations[3 * 4 + 1]; // non-ideal rotation (East != 2)
        board[18] = PieceUtils.pack(50, 51, 52, 2);            // requires East(17) == 2 to match

        Method m = ConflictReducer.class.getDeclaredMethod("rotationPass", int[].class);
        m.setAccessible(true);
        int after = (int) m.invoke(reducer, (Object) board);

        assertEquals(0, after, "rotationPass should find the orientation whose East==2, eliminating the conflict");
        assertEquals(2, PieceUtils.getEast(board[17]), "Winning orientation's East edge must equal 2 to match the neighbour's West");
    }

    @Test
    void testRotationPassSkipsLockedPositions() throws Exception {
        int base = PieceUtils.pack(1, 2, 3, 4);
        setPhysicalPiece(3, base);
        ConflictReducer reducer = new ConflictReducer(mockInventory, true); // lockCenter=true locks position 135

        int[] board = emptyBoard();
        board[135] = mockInventory.allOrientations[3 * 4 + 1]; // non-ideal rotation, mismatched
        board[136] = PieceUtils.pack(50, 51, 52, 2);            // would match base's East==2 if rotation were allowed

        Method m = ConflictReducer.class.getDeclaredMethod("rotationPass", int[].class);
        m.setAccessible(true);
        int before = reducer.countConflicts(board);
        int after = (int) m.invoke(reducer, (Object) board);

        assertEquals(before, after, "Locked position's conflict must remain unresolved");
        assertEquals(mockInventory.allOrientations[3 * 4 + 1], board[135], "Locked position's piece must not be rotated");
    }

    // -----------------------------------------------------------------------
    // swapPass (private) - Strategy 2
    // -----------------------------------------------------------------------

    @Test
    void testSwapPassSwapsPiecesToReduceConflicts() throws Exception {
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);
        int[] board = emptyBoard();
        // Interior cells: at a border cell, swapping these two specific pieces
        // would fix the East/West mismatch but introduce a new frame violation
        // (net conflicts unchanged), so swapPass would correctly reject the
        // swap — this test is about the swap mechanics, not border interplay.
        board[17] = PieceUtils.pack(0, 9, 0, 0); // East=9
        board[18] = PieceUtils.pack(0, 0, 0, 7); // West=7 -> mismatches East=9

        Method m = ConflictReducer.class.getDeclaredMethod("swapPass", int[].class);
        m.setAccessible(true);
        int before = reducer.countConflicts(board);
        int after = (int) m.invoke(reducer, (Object) board);

        assertEquals(1, before);
        assertEquals(0, after, "swapPass should find a swap that eliminates the conflict");
        assertEquals(PieceUtils.pack(0, 0, 0, 7), board[17]);
        assertEquals(PieceUtils.pack(0, 9, 0, 0), board[18]);
    }

    @Test
    void testSwapPassSkipsLockedPositions() throws Exception {
        ConflictReducer reducer = new ConflictReducer(mockInventory, true); // lockCenter=true
        int[] board = emptyBoard();
        board[135] = PieceUtils.pack(0, 9, 0, 0);
        board[136] = PieceUtils.pack(0, 0, 0, 7);

        Method m = ConflictReducer.class.getDeclaredMethod("swapPass", int[].class);
        m.setAccessible(true);
        int before = reducer.countConflicts(board);
        int after = (int) m.invoke(reducer, (Object) board);

        assertEquals(before, after, "Locked positions must not be swapped away from their conflict");
        assertEquals(PieceUtils.pack(0, 9, 0, 0), board[135]);
        assertEquals(PieceUtils.pack(0, 0, 0, 7), board[136]);
    }

    // -----------------------------------------------------------------------
    // reduceLive / reducePostProcess (public)
    // -----------------------------------------------------------------------

    @Test
    void testReduceLiveShortCircuitsWhenBoardHasNoConflicts() {
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);
        int[] board = emptyBoard();
        board[17] = PieceUtils.pack(1, 2, 3, 4);
        board[18] = PieceUtils.pack(5, 6, 7, 2); // matches, no conflicts
        int[] before = Arrays.copyOf(board, 256);

        int result = reducer.reduceLive(board, 3);

        assertEquals(0, result);
        assertArrayEquals(before, board, "A conflict-free board must be left untouched");
    }

    @Test
    void testReduceLiveResolvesConflictViaRotation() {
        int base = PieceUtils.pack(1, 2, 3, 4);
        setPhysicalPiece(9, base);
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);

        int[] board = emptyBoard();
        board[17] = mockInventory.allOrientations[9 * 4 + 1]; // mismatched rotation
        board[18] = PieceUtils.pack(50, 51, 52, 2);

        int result = reducer.reduceLive(board, 2);

        assertEquals(0, result);
        assertEquals(0, reducer.countConflicts(board));
    }

    @Test
    void testReducePostProcessAlsoResolvesConflictAndStopsEarly() {
        int base = PieceUtils.pack(1, 2, 3, 4);
        setPhysicalPiece(9, base);
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);

        int[] board = emptyBoard();
        board[17] = mockInventory.allOrientations[9 * 4 + 1];
        board[18] = PieceUtils.pack(50, 51, 52, 2);

        int result = reducer.reducePostProcess(board, 20);

        assertEquals(0, result);
    }

    // -----------------------------------------------------------------------
    // randomRestartFill / mcvRestartFill (public)
    // -----------------------------------------------------------------------

    @Test
    void testMcvRestartFillReturnsUnmodifiedCopyWhenNoHoles() {
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);
        int[] board = new int[256];
        Arrays.fill(board, PieceUtils.pack(0, 0, 0, 0));

        int[] result = reducer.mcvRestartFill(board, 5);

        assertNotSame(board, result, "Result must be a fresh array, not the same reference");
        assertArrayEquals(board, result);
    }

    @Test
    void testRandomRestartFillReturnsUnmodifiedCopyWhenNoHoles() {
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);
        int[] board = new int[256];
        Arrays.fill(board, PieceUtils.pack(0, 0, 0, 0));

        int[] result = reducer.randomRestartFill(board, 5);

        assertNotSame(board, result);
        assertArrayEquals(board, result);
    }

    @Test
    void testMcvRestartFillFillsAllHolesWithUnusedPieces() {
        // Physical piece 5 sits at position 0 already; its exact orientation must be
        // resolvable via getPhysId so it's excluded from the fill pool. Every other
        // physical id is the default all-zero filler piece.
        int placedBase = PieceUtils.pack(1, 2, 3, 4);
        setPhysicalPiece(5, placedBase);
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);

        int[] board = emptyBoard();
        board[0] = placedBase; // physId 5, used
        board[10] = -1;        // hole
        board[20] = -2;        // surgeon hole

        int[] original = Arrays.copyOf(board, 256);
        int[] result = reducer.mcvRestartFill(board, 3);

        assertArrayEquals(original, board, "Input board must not be mutated");
        // Every unused physical id (all but 5) is the identical all-zero filler
        // piece, so the greedy fill's outcome is fully deterministic here.
        assertEquals(0, result[10], "Hole must be filled with the only available (all-default) filler piece");
        assertEquals(0, result[20], "Surgeon hole must be filled too");
        assertEquals(placedBase, result[0], "Pre-placed piece must be untouched by the fill");
    }

    @Test
    void testRandomRestartFillFillsAllHolesWithUnusedPieces() {
        int placedBase = PieceUtils.pack(1, 2, 3, 4);
        setPhysicalPiece(5, placedBase);
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);

        int[] board = emptyBoard();
        board[0] = placedBase;
        board[10] = -1;
        board[20] = -2;

        int[] original = Arrays.copyOf(board, 256);
        int[] result = reducer.randomRestartFill(board, 3);

        assertArrayEquals(original, board, "Input board must not be mutated");
        assertEquals(0, result[10], "Hole must be filled with the only available (all-default) filler piece");
        assertEquals(0, result[20], "Surgeon hole must be filled too");
        assertEquals(placedBase, result[0]);
    }

    // -----------------------------------------------------------------------
    // pickMostConstrainedHole (private) - the documented MCV contract
    // -----------------------------------------------------------------------

    @Test
    void testPickMostConstrainedHoleChoosesHoleWithMostKnownNeighbours() throws Exception {
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);
        int[] trial = emptyBoard();
        // Hole at 17 (row1,col1): north(1), west(16) and south(33) neighbours known -> 3 known edges.
        trial[1] = PieceUtils.pack(1, 2, 3, 4);
        trial[16] = PieceUtils.pack(1, 2, 3, 4);
        trial[33] = PieceUtils.pack(1, 2, 3, 4);
        // Hole at 200 (row12,col8): only its north neighbour (184) is known -> 1 known edge.
        trial[184] = PieceUtils.pack(1, 2, 3, 4);

        Set<Integer> holes = new HashSet<>(Arrays.asList(17, 200));

        Method m = ConflictReducer.class.getDeclaredMethod("pickMostConstrainedHole", int[].class, Set.class, Random.class);
        m.setAccessible(true);
        int chosen = (int) m.invoke(reducer, trial, holes, new Random(1));

        assertEquals(17, chosen, "Hole with 3 known neighbour edges must be chosen over one with only 1");
    }

    // -----------------------------------------------------------------------
    // fillOnce / mcvFillOnce (private) - greedy candidate selection
    // -----------------------------------------------------------------------

    @Test
    void testMcvFillOnceChoosesZeroViolationCandidateWhenAvailable() throws Exception {
        // physId 1: perfect match for the hole's one real constraint (North==9).
        // physId 2: guaranteed mismatch on that constraint, regardless of rotation.
        setPhysicalPiece(1, PieceUtils.pack(9, 0, 0, 0));
        setPhysicalPiece(2, PieceUtils.pack(1, 1, 1, 1));
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);

        int[] board = emptyBoard();
        // Hole at 17 (row1,col1); its only known neighbour is north (pos 1), whose
        // South edge is 9, so the hole's North edge must equal 9 to match.
        board[1] = PieceUtils.pack(0, 0, 9, 0);

        List<Integer> holePositions = new ArrayList<>(List.of(17));
        List<Integer> unusedPhysIds = new ArrayList<>(List.of(1, 2));

        Method m = ConflictReducer.class.getDeclaredMethod("mcvFillOnce", int[].class, List.class, List.class, Random.class);
        m.setAccessible(true);
        int[] result = (int[]) m.invoke(reducer, board, holePositions, unusedPhysIds, new Random(7));

        assertEquals(PieceUtils.pack(9, 0, 0, 0), result[17],
                "The zero-violation candidate (physId 1, rotation 0) must be chosen over the guaranteed mismatch");
    }

    @Test
    void testFillOnceChoosesZeroViolationCandidateWhenAvailable() throws Exception {
        setPhysicalPiece(1, PieceUtils.pack(9, 0, 0, 0));
        setPhysicalPiece(2, PieceUtils.pack(1, 1, 1, 1));
        ConflictReducer reducer = new ConflictReducer(mockInventory, false);

        int[] board = emptyBoard();
        board[1] = PieceUtils.pack(0, 0, 9, 0);

        List<Integer> holePositions = new ArrayList<>(List.of(17));
        List<Integer> unusedPhysIds = new ArrayList<>(List.of(1, 2));

        Method m = ConflictReducer.class.getDeclaredMethod("fillOnce", int[].class, List.class, List.class, Random.class);
        m.setAccessible(true);
        int[] result = (int[]) m.invoke(reducer, board, holePositions, unusedPhysIds, new Random(7));

        assertEquals(PieceUtils.pack(9, 0, 0, 0), result[17]);
    }
}
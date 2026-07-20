package dk.puzzle.ai;

import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for SurgeonHeuristics, constructing it directly (not via
 * EternitySolver) and exercising its own public contract: locking behavior,
 * setTargetedHolesPercentage, punchHoles, and excavateFrontier.
 *
 * <p>Both punchHoles and excavateFrontier use an internal, unseeded
 * {@code new Random()}, so tests here assert structural invariants (counts,
 * locked/tabu exclusion, board independence) rather than exact chosen
 * indices - except where a fixture is deliberately sized so the outcome is
 * forced regardless of the random draw (see
 * testExcavateFrontierNeverRemovesLockedCenterEvenWithFullBoardExhaustion).</p>
 */
class SurgeonHeuristicsTest {

    private boolean[] getIsLockedField(SurgeonHeuristics surgeon) throws Exception {
        Field field = SurgeonHeuristics.class.getDeclaredField("isLocked");
        field.setAccessible(true);
        return (boolean[]) field.get(surgeon);
    }

    private double getTargetedHolesPercentageField(SurgeonHeuristics surgeon) throws Exception {
        Field field = SurgeonHeuristics.class.getDeclaredField("targetedHolesPercentage");
        field.setAccessible(true);
        return (double) field.get(surgeon);
    }

    private int[] buildFullyPlacedBoard() {
        int[] board = new int[256];
        for (int i = 0; i < 256; i++) {
            board[i] = 1000 + i;
        }
        return board;
    }

    private int[] identityBuildOrder() {
        int[] order = new int[256];
        for (int i = 0; i < 256; i++) order[i] = i;
        return order;
    }

    private int countValue(int[] board, int value) {
        int count = 0;
        for (int v : board) {
            if (v == value) count++;
        }
        return count;
    }

    // ── Constructor / locking ──

    @Test
    void testConstructorLocksCenterWhenRequested() throws Exception {
        SurgeonHeuristics surgeon = new SurgeonHeuristics(true, 0.5);
        boolean[] locked = getIsLockedField(surgeon);

        assertTrue(locked[135], "Center position 135 must be locked when lockCenter=true");
        assertFalse(locked[0], "Unrelated positions must remain unlocked");
    }

    @Test
    void testConstructorDoesNotLockCenterWhenFalse() throws Exception {
        SurgeonHeuristics surgeon = new SurgeonHeuristics(false, 0.5);
        boolean[] locked = getIsLockedField(surgeon);

        assertFalse(locked[135], "Center position must stay unlocked when lockCenter=false");
    }

    @Test
    void testConstructorLocksHintPositions() throws Exception {
        SurgeonHeuristics surgeon = new SurgeonHeuristics(false, 0.5, 10, 20, 30);
        boolean[] locked = getIsLockedField(surgeon);

        assertTrue(locked[10], "Hint position 10 must be locked");
        assertTrue(locked[20], "Hint position 20 must be locked");
        assertTrue(locked[30], "Hint position 30 must be locked");
        assertFalse(locked[135], "Center must stay unlocked since lockCenter=false, even with hints supplied");
        assertFalse(locked[11], "Positions not listed as hints must remain unlocked");
    }

    @Test
    void testConstructorHandlesNullHintPositionsVarargsGracefully() throws Exception {
        SurgeonHeuristics surgeon = new SurgeonHeuristics(true, 0.5, (int[]) null);
        boolean[] locked = getIsLockedField(surgeon);

        assertTrue(locked[135], "Center should still be locked from lockCenter=true");
        assertFalse(locked[0], "No exception should be thrown for a null hintPositions varargs array");
    }

    // ── setTargetedHolesPercentage ──

    @Test
    void testSetTargetedHolesPercentageUpdatesField() throws Exception {
        SurgeonHeuristics surgeon = new SurgeonHeuristics(false, 0.3);
        surgeon.setTargetedHolesPercentage(0.9);

        assertEquals(0.9, getTargetedHolesPercentageField(surgeon), 0.0001);
    }

    @Test
    void testSetTargetedHolesPercentageAcceptsBoundaryValuesZeroAndOne() throws Exception {
        SurgeonHeuristics surgeon = new SurgeonHeuristics(false, 0.5);

        surgeon.setTargetedHolesPercentage(0.0);
        assertEquals(0.0, getTargetedHolesPercentageField(surgeon), 0.0001);

        surgeon.setTargetedHolesPercentage(1.0);
        assertEquals(1.0, getTargetedHolesPercentageField(surgeon), 0.0001);
    }

    // ── punchHoles ──

    @Test
    void testPunchHolesReturnsRequestedNumberOfClones() {
        SurgeonHeuristics surgeon = new SurgeonHeuristics(false, 0.5);
        int[] board = buildFullyPlacedBoard();
        int[] tabuTenure = new int[256];

        List<int[]> clones = surgeon.punchHoles(board, 5, 10, tabuTenure, 0, 256, identityBuildOrder());

        assertEquals(5, clones.size(), "punchHoles must produce exactly numClones board variations");
    }

    @Test
    void testPunchHolesEachCloneHasRequestedNumberOfHoles() {
        SurgeonHeuristics surgeon = new SurgeonHeuristics(false, 0.5);
        int[] board = buildFullyPlacedBoard();
        int[] tabuTenure = new int[256];

        // currentHighScore=256 skips the extra forced frontier hole, isolating
        // the count contributed purely by the targeted+random hole selection.
        List<int[]> clones = surgeon.punchHoles(board, 4, 20, tabuTenure, 0, 256, identityBuildOrder());

        for (int[] clone : clones) {
            assertEquals(256, clone.length);
            assertEquals(20, countValue(clone, -2),
                    "Each clone should have exactly numHoles pieces punched when enough placed pieces are available");
        }
    }

    @Test
    void testPunchHolesNeverRemovesLockedCenterOrHintPositions() {
        SurgeonHeuristics surgeon = new SurgeonHeuristics(true, 0.5, 5, 10);
        int[] board = buildFullyPlacedBoard();
        int[] tabuTenure = new int[256];

        // Request almost every placeable piece as a hole to maximize the
        // chance a naive implementation would have swept up a locked index.
        List<int[]> clones = surgeon.punchHoles(board, 3, 200, tabuTenure, 0, 256, identityBuildOrder());

        for (int[] clone : clones) {
            assertNotEquals(-2, clone[135], "Locked center piece must never be punched out");
            assertNotEquals(-2, clone[5], "Locked hint position must never be punched out");
            assertNotEquals(-2, clone[10], "Locked hint position must never be punched out");
        }
    }

    @Test
    void testPunchHolesRespectsTabuTenure() {
        SurgeonHeuristics surgeon = new SurgeonHeuristics(false, 0.5);
        int[] board = buildFullyPlacedBoard();

        int[] freeIndices = {3, 40, 90, 150, 200};
        Set<Integer> freeSet = new HashSet<>();
        for (int idx : freeIndices) freeSet.add(idx);

        int[] tabuTenure = new int[256];
        Arrays.fill(tabuTenure, 100); // tenure(100) > currentIteration(0) -> blocked
        for (int idx : freeIndices) {
            tabuTenure[idx] = 0; // tenure(0) <= currentIteration(0) -> free
        }

        List<int[]> clones = surgeon.punchHoles(board, 3, 3, tabuTenure, 0, 256, identityBuildOrder());

        for (int[] clone : clones) {
            for (int i = 0; i < 256; i++) {
                if (!freeSet.contains(i)) {
                    assertNotEquals(-2, clone[i], "Position " + i + " is under tabu tenure and must not be punched");
                }
            }
        }
    }

    @Test
    void testPunchHolesActualHolesClampedToAvailablePlacedCount() {
        SurgeonHeuristics surgeon = new SurgeonHeuristics(false, 0.5);
        int[] board = new int[256];
        Arrays.fill(board, -1);
        board[1] = 1001;
        board[2] = 1002;
        board[3] = 1003; // only 3 placeable pieces exist on this board

        int[] tabuTenure = new int[256];

        List<int[]> clones = surgeon.punchHoles(board, 2, 50, tabuTenure, 0, 256, identityBuildOrder());

        for (int[] clone : clones) {
            assertEquals(3, countValue(clone, -2),
                    "actualHoles must be clamped to the number of available placed pieces (3), not the requested 50");
        }
    }

    @Test
    void testPunchHolesSetsForcedHoleAtBuildOrderFrontierWhenBelow256() {
        SurgeonHeuristics surgeon = new SurgeonHeuristics(false, 0.5);
        int[] board = buildFullyPlacedBoard();
        int[] tabuTenure = new int[256];
        int[] buildOrder = identityBuildOrder();

        // numHoles=0 removes all randomness from the hole count, isolating
        // the unconditional "clear the next frontier slot" behavior.
        List<int[]> clones = surgeon.punchHoles(board, 3, 0, tabuTenure, 0, 100, buildOrder);

        for (int[] clone : clones) {
            assertEquals(-2, clone[buildOrder[100]],
                    "The next frontier slot must always be explicitly punched, even when numHoles is 0");
            assertEquals(1, countValue(clone, -2),
                    "With numHoles=0 the only hole should be the forced frontier slot");
        }
    }

    @Test
    void testPunchHolesForcedFrontierHoleSkipsLockedPosition() {
        SurgeonHeuristics surgeon = new SurgeonHeuristics(true, 0.5); // locks index 135
        int[] board = buildFullyPlacedBoard();
        int originalCenterValue = board[135];
        int[] tabuTenure = new int[256];
        int[] buildOrder = identityBuildOrder();

        // buildOrder[135] == 135, which is locked. numHoles=0 isolates the
        // forced frontier-hole behavior from the regular targeted/random
        // hole selection (which already excludes locked positions).
        List<int[]> clones = surgeon.punchHoles(board, 3, 0, tabuTenure, 0, 135, buildOrder);

        for (int[] clone : clones) {
            assertEquals(originalCenterValue, clone[135],
                    "The forced frontier hole must not overwrite a locked position");
            assertEquals(0, countValue(clone, -2),
                    "No hole should be punched at all when the only candidate (the frontier slot) is locked");
        }
    }

    @Test
    void testPunchHolesReturnsIndependentBoardCopiesAndDoesNotMutateSource() {
        SurgeonHeuristics surgeon = new SurgeonHeuristics(false, 0.5);
        int[] board = buildFullyPlacedBoard();
        int[] snapshot = Arrays.copyOf(board, 256);
        int[] tabuTenure = new int[256];

        List<int[]> clones = surgeon.punchHoles(board, 2, 5, tabuTenure, 0, 256, identityBuildOrder());

        assertArrayEquals(snapshot, board, "punchHoles must not mutate the source board");
        assertNotSame(clones.get(0), clones.get(1), "Each clone must be an independent array instance");
        assertNotSame(board, clones.get(0), "Clones must not alias the source board array");
    }

    // ── excavateFrontier ──

    @Test
    void testExcavateFrontierReturnsEmptyListWhenCurrentHighScoreIsAtOrBeyond256() {
        SurgeonHeuristics surgeon = new SurgeonHeuristics(false, 0.5);
        int[] board = buildFullyPlacedBoard();

        List<int[]> clones = surgeon.excavateFrontier(board, 5, 10, 0, 256, identityBuildOrder());

        assertTrue(clones.isEmpty(), "No stuck index exists once the board is fully built (currentHighScore >= 256)");
    }

    @Test
    void testExcavateFrontierReturnsRequestedNumberOfClones() {
        SurgeonHeuristics surgeon = new SurgeonHeuristics(false, 0.5);
        int[] board = buildFullyPlacedBoard();

        List<int[]> clones = surgeon.excavateFrontier(board, 4, 10, 0, 50, identityBuildOrder());

        assertEquals(4, clones.size());
    }

    @Test
    void testExcavateFrontierNeverRemovesLockedCenterEvenWithFullBoardExhaustion() {
        SurgeonHeuristics surgeon = new SurgeonHeuristics(true, 0.5); // locks index 135
        int[] board = buildFullyPlacedBoard();
        int originalCenterValue = board[135];
        int[] buildOrder = identityBuildOrder();

        // numHoles=255 equals the total number of non-locked cells on a
        // fully-placed, fully-connected 16x16 grid. This forces the random
        // BFS crater to keep growing until every non-locked cell has been
        // visited and evaluated, making the outcome deterministic despite
        // excavateFrontier's internal unseeded Random: the locked center is
        // the only cell that can possibly survive.
        List<int[]> clones = surgeon.excavateFrontier(board, 1, 255, 0, 0, buildOrder);

        assertEquals(1, clones.size());
        int[] clone = clones.get(0);

        assertEquals(originalCenterValue, clone[135],
                "Locked center must survive even when the crater exhausts the entire reachable board");
        assertEquals(-2, clone[0], "The stuck index (buildOrder[currentHighScore]) must end up marked as the active hole");
        assertEquals(254, countValue(clone, -1),
                "255 non-locked cells get removed; one (the stuck index) is then overwritten to -2, leaving 254 as -1");
        assertEquals(1, countValue(clone, -2));
    }

    @Test
    void testExcavateFrontierHoleCountNeverExceedsRequested() {
        SurgeonHeuristics surgeon = new SurgeonHeuristics(false, 0.5);
        int[] board = buildFullyPlacedBoard();

        List<int[]> clones = surgeon.excavateFrontier(board, 5, 5, 0, 30, identityBuildOrder());

        for (int[] clone : clones) {
            int removed = countValue(clone, -1);
            assertTrue(removed <= 5, "The crater must never remove more than the requested numHoles cells");
        }
    }

    @Test
    void testExcavateFrontierForcedStuckIndexSkipsLockedPosition() {
        SurgeonHeuristics surgeon = new SurgeonHeuristics(true, 0.5); // locks index 135
        int[] board = buildFullyPlacedBoard();
        int originalCenterValue = board[135];
        int[] buildOrder = identityBuildOrder();

        // buildOrder[135] == 135, so the stuck index itself is the locked center.
        List<int[]> clones = surgeon.excavateFrontier(board, 3, 5, 0, 135, buildOrder);

        for (int[] clone : clones) {
            assertEquals(originalCenterValue, clone[135],
                    "The forced stuck-index hole must not overwrite a locked position");
            assertNotEquals(-2, clone[135]);
        }
    }

    @Test
    void testExcavateFrontierAlwaysMarksStuckIndexAsHole() {
        SurgeonHeuristics surgeon = new SurgeonHeuristics(false, 0.5);
        int[] board = buildFullyPlacedBoard();
        int[] buildOrder = identityBuildOrder();

        List<int[]> clones = surgeon.excavateFrontier(board, 2, 5, 0, 50, buildOrder);

        for (int[] clone : clones) {
            assertEquals(-2, clone[buildOrder[50]],
                    "The frontier slot (the exact point of failure) must always be marked as the active hole");
        }
    }
}

package dk.puzzle.core;

import dk.puzzle.io.CheckpointManager;
import dk.puzzle.io.RecordManager;
import dk.puzzle.model.PieceInventory;
import dk.puzzle.util.PieceUtils;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.MockedStatic;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;

/**
 * Unit tests for EternitySolver core logic.
 * Mocks static I/O and GUI dependencies to allow headless testing.
 */
class EternitySolverTest {

    private PieceInventory mockInventory;
    private MockedStatic<CheckpointManager> checkpointMock;
    private MockedStatic<RecordManager> recordMock;
    private MockedStatic<Eternity> eternityMock;

    @BeforeEach
    void setUp() {
        mockInventory = mock(PieceInventory.class);
        mockInventory.allOrientations = new int[1024];
        mockInventory.physicalMapping = new int[1024]; // This array also needs valid data
        
        // Fill allOrientations with a valid "empty" piece (all sides 0)
        Arrays.fill(mockInventory.allOrientations, 0); 
        // Fill physicalMapping with dummy physical IDs (0-255)
        for (int i = 0; i < 1024; i++) {
            mockInventory.physicalMapping[i] = i / 4; // Maps 4 orientations to each physical piece
        }

        // Prevent disk access and GUI updates during tests
        checkpointMock = mockStatic(CheckpointManager.class);
        recordMock = mockStatic(RecordManager.class);
        eternityMock = mockStatic(Eternity.class);

        checkpointMock.when(() -> CheckpointManager.loadSmartCheckpoint(anyString())).thenReturn(null);
    }

    @AfterEach
    void tearDown() {
        checkpointMock.close();
        recordMock.close();
        eternityMock.close();
    }

    private double getTargetedHolesPercentage(dk.puzzle.ai.SurgeonHeuristics surgeon) throws Exception {
        java.lang.reflect.Field field = dk.puzzle.ai.SurgeonHeuristics.class.getDeclaredField("targetedHolesPercentage");
        field.setAccessible(true);
        return (double) field.get(surgeon);
    }

    private int[] getHintPositions() throws Exception {
        java.lang.reflect.Field field = EternitySolver.class.getDeclaredField("HINT_POSITIONS");
        field.setAccessible(true);
        return (int[]) field.get(null);
    }

    private int getAtomicIntField(EternitySolver solver, String fieldName) throws Exception {
        java.lang.reflect.Field field = EternitySolver.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        java.util.concurrent.atomic.AtomicInteger atomic =
                (java.util.concurrent.atomic.AtomicInteger) field.get(solver);
        return atomic.get();
    }

    @Test
    void testConstructorBuildOrderTypewriter() {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        for (int i = 0; i < 256; i++) {
            assertEquals(i, solver.buildOrder[i], "Typewriter build order should be sequential [0..255]");
        }
    }

    @Test
    void testConstructorBuildOrderSpiral() {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.SPIRAL, false);

        // Spiral for 16x16 starts at 0, 1... 15, then moves down the right edge (31, 47...)
        assertEquals(0, solver.buildOrder[0]);
        assertEquals(1, solver.buildOrder[1]);
        assertEquals(15, solver.buildOrder[15]);
        assertEquals(31, solver.buildOrder[16]);
        assertNotEquals(16, solver.buildOrder[16], "Spiral build order should wrap edges, not follow row-index sequentially");
    }

    @Test
    void testCountPiecesIgnoringHoles() {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        int[] board = new int[256];
        Arrays.fill(board, -1);
        board[10] = 500;
        board[20] = 600;
        board[30] = -2; // Surgeon Hole should not be counted as a placed piece

        assertEquals(2, solver.countPieces(board), "countPieces should only count valid bit-packed pieces");
    }

    @Test
    void testBuildDisplayBoardCoordinates() {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        int[] source = new int[256];
        Arrays.fill(source, -1);
        source[17] = 1234; // Row 1, Column 1

        int[][] display = solver.buildDisplayBoard(source);
        assertEquals(1234, display[1][1], "Flat board index 17 should map to 2D array [1][1]");
    }

    @Test
    void testRetreatResetsBoardState() {
        int centerPiece = 9999;
        EternitySolver solver = new EternitySolver(
                mockInventory, centerPiece, false, EternitySolver.BuildStrategy.TYPEWRITER, true);

        Arrays.fill(solver.bestBoard, 1);
        solver.bestBoard[135] = centerPiece;

        solver.retreat(100, "Test retreat");

        assertEquals(100, solver.deepestStep);
        assertEquals(-1, solver.bestBoard[100], "Pieces at or after the retreat depth should be cleared");
        assertEquals(centerPiece, solver.bestBoard[135], "Locked center piece at index 135 must be preserved");
    }

    @Test
    void testConfigurationSetters() throws Exception {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        // Test setStagnationLimit
        solver.setStagnationLimit(45);
        assertEquals(45, solver.stagnationLimitMinutes);

        // Test setBatchSizeOverride
        solver.setBatchSizeOverride(1000);
        assertEquals(1000, solver.userBatchSizeOverride);

        // Test setExtinctionThreshold
        solver.setExtinctionThreshold(0.85);
        assertEquals(0.85, solver.extinctionThreshold, 0.001);

        // Test setTargetedHolesPercentage
        solver.setTargetedHolesPercentage(0.95);
        double val = getTargetedHolesPercentage(solver.surgeon);
        assertEquals(0.95, val, 0.001);
    }

    @Test
    void testTriggerManualOverride() {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        assertFalse(solver.manualOverrideRequested);
        assertEquals(0, solver.manualBaseCampTarget);

        solver.triggerManualOverride(150);
        assertTrue(solver.manualOverrideRequested);
        assertEquals(150, solver.manualBaseCampTarget);
    }

    @Test
    void testVerifyBoardStrictValidEmpty() {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        int[] board = new int[256];
        Arrays.fill(board, -1); // empty board is mathematically flawless

        assertTrue(solver.verifyBoardStrict(board));
    }

    @Test
    void testVerifyBoardStrictValidMatching() {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        int[] board = new int[256];
        Arrays.fill(board, -1);

        // Pack two matching pieces horizontally:
        // Piece at index 0 (row 0, col 0): North=1, East=2, South=3, West=4
        // Piece at index 1 (row 0, col 1): North=5, East=6, South=7, West=2
        // East of index 0 (2) matches West of index 1 (2).
        board[0] = PieceUtils.pack(1, 2, 3, 4);
        board[1] = PieceUtils.pack(5, 6, 7, 2);

        // Pack two matching pieces vertically:
        // Piece at index 16 (row 1, col 0): North=3, East=8, South=9, West=10
        // South of index 0 (3) matches North of index 16 (3).
        board[16] = PieceUtils.pack(3, 8, 9, 10);

        assertTrue(solver.verifyBoardStrict(board), "Board with all matching adjacent edges should be valid");
    }

    @Test
    void testVerifyBoardStrictEastWestMismatch() {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        int[] board = new int[256];
        Arrays.fill(board, -1);

        // East of index 0 (2) mismatches West of index 1 (9)
        board[0] = PieceUtils.pack(1, 2, 3, 4);
        board[1] = PieceUtils.pack(5, 6, 7, 9);

        assertFalse(solver.verifyBoardStrict(board), "Board with horizontal conflict must be invalid");
    }

    @Test
    void testVerifyBoardStrictSouthNorthMismatch() {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        int[] board = new int[256];
        Arrays.fill(board, -1);

        // South of index 0 (3) mismatches North of index 16 (9)
        board[0] = PieceUtils.pack(1, 2, 3, 4);
        board[16] = PieceUtils.pack(9, 6, 7, 8);

        assertFalse(solver.verifyBoardStrict(board), "Board with vertical conflict must be invalid");
    }

    @Test
    void testTopBoardRegistry() {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);
        EternitySolver.TopBoardRegistry registry = solver.topBoards;

        registry.clear();
        assertNull(registry.nextForRepair());

        // Offer some unique boards
        int[] board1 = new int[256];
        board1[0] = 100;
        int[] board2 = new int[256];
        board2[0] = 200;

        registry.offer(board1, 10);
        registry.offer(board2, 12);

        // Offer a duplicate board
        int[] board1Duplicate = new int[256];
        board1Duplicate[0] = 100;
        registry.offer(board1Duplicate, 10);

        // Retrieve round-robin and check they are unique
        int[] first = registry.nextForRepair();
        int[] second = registry.nextForRepair();
        int[] third = registry.nextForRepair();

        assertNotNull(first);
        assertNotNull(second);
        assertNotNull(third);

        // First and third should be identical (since we have size 2 and round-robin)
        assertArrayEquals(first, third);
        assertFalse(Arrays.equals(first, second));

        // Test capacity limit of 20
        registry.clear();
        for (int i = 0; i < 30; i++) {
            int[] temp = new int[256];
            temp[0] = i;
            registry.offer(temp, i);
        }

        // Retrieve 20 elements, they must be unique, and retrieving 21st element should be one of the first 20.
        Set<Integer> uniqueValues = new HashSet<>();
        for (int i = 0; i < 20; i++) {
            int[] b = registry.nextForRepair();
            assertNotNull(b);
            uniqueValues.add(b[0]);
        }
        assertEquals(20, uniqueValues.size(), "Registry should contain exactly 20 unique boards");

        // 21st call should return a board whose val is in uniqueValues (no new boards beyond capacity of 20)
        int[] b21 = registry.nextForRepair();
        assertTrue(uniqueValues.contains(b21[0]), "Capped registry should not add boards beyond capacity");
    }

    @Test
    void testLockCenterLocksCenterAndHintPositions() throws Exception {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, true);

        assertNotEquals(-1, solver.bestBoard[135], "Center position must be locked when lockCenter=true");
        for (int hPos : getHintPositions()) {
            assertNotEquals(-1, solver.bestBoard[hPos], "Hint position " + hPos + " must be locked when lockCenter=true");
        }
    }

    @Test
    void testUnlockedConstructorLeavesHintsAndCenterEmpty() throws Exception {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        assertEquals(-1, solver.bestBoard[135], "Center must stay empty when lockCenter=false");
        for (int hPos : getHintPositions()) {
            assertEquals(-1, solver.bestBoard[hPos], "Hint position " + hPos + " must stay empty when lockCenter=false");
        }
    }

    @Test
    void testRetreatWithLockCenterRestoresHintPositions() throws Exception {
        int centerPiece = 8888;
        EternitySolver solver = new EternitySolver(
                mockInventory, centerPiece, false, EternitySolver.BuildStrategy.TYPEWRITER, true);

        int[] hintPositions = getHintPositions();
        int[] lockedHintValues = new int[hintPositions.length];
        for (int i = 0; i < hintPositions.length; i++) {
            lockedHintValues[i] = solver.bestBoard[hintPositions[i]];
        }

        Arrays.fill(solver.bestBoard, 1);
        solver.retreat(0, null);

        assertEquals(centerPiece, solver.bestBoard[135], "Retreat must restore the locked center piece");
        for (int i = 0; i < hintPositions.length; i++) {
            assertEquals(lockedHintValues[i], solver.bestBoard[hintPositions[i]],
                    "Retreat must restore locked hint values even when retreating to depth 0");
        }
    }

    @Test
    void testGetDynamicBatchSizeThresholds() throws Exception {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);
        java.lang.reflect.Method method = EternitySolver.class.getDeclaredMethod("getDynamicBatchSize");
        method.setAccessible(true);

        solver.deepestStep = 0;
        assertEquals(5000, (int) method.invoke(solver), "Below 50: small board still uses a large batch");

        solver.deepestStep = 75;
        assertEquals(15000, (int) method.invoke(solver), "Mid-range depth uses the largest default batch");

        solver.deepestStep = 150;
        assertEquals(5000, (int) method.invoke(solver), ">=100 depth throttles the batch size down");

        solver.deepestStep = 190;
        assertEquals(1000, (int) method.invoke(solver), ">=180 depth throttles further for precision");

        solver.deepestStep = 220;
        assertEquals(250, (int) method.invoke(solver), ">=200 depth uses the smallest endgame batch");
    }

    @Test
    void testBoardHashDeterministicAndPositionSensitive() throws Exception {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);
        java.lang.reflect.Method method = EternitySolver.class.getDeclaredMethod("boardHash", int[].class);
        method.setAccessible(true);

        int[] empty = new int[256];
        Arrays.fill(empty, -1);
        assertEquals(0, (int) method.invoke(solver, (Object) empty), "An all-empty board must hash to 0");

        int[] boardA = new int[256];
        Arrays.fill(boardA, -1);
        boardA[10] = 500;

        int[] boardB = new int[256];
        Arrays.fill(boardB, -1);
        boardB[20] = 500; // same piece value, different position

        int hashA1 = (int) method.invoke(solver, (Object) boardA);
        int hashA2 = (int) method.invoke(solver, (Object) boardA);
        int hashB = (int) method.invoke(solver, (Object) boardB);

        assertEquals(hashA1, hashA2, "boardHash must be deterministic for the same board contents");
        assertNotEquals(hashA1, hashB, "boardHash must be position-sensitive, not just content-sensitive");
    }

    @Test
    void testGenerateHashForBoardNullAndValue() throws Exception {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);
        java.lang.reflect.Method method = EternitySolver.class.getDeclaredMethod("generateHashForBoard", int[].class);
        method.setAccessible(true);

        assertEquals("EMPTY_BOARD", method.invoke(solver, new Object[]{null}));

        int[] board = new int[256];
        Arrays.fill(board, -1);
        board[5] = 42;

        String expected = Integer.toHexString(Arrays.hashCode(board)).toUpperCase();
        assertEquals(expected, method.invoke(solver, (Object) board));
    }

    @Test
    void testRescueBoardClearsConflictingNeighborsAndReturnsValidCount() throws Exception {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);
        java.lang.reflect.Method method = EternitySolver.class.getDeclaredMethod("rescueBoard", int[].class);
        method.setAccessible(true);

        int[] board = new int[256];
        Arrays.fill(board, -1);
        board[0] = PieceUtils.pack(1, 2, 3, 4);     // North=1 East=2 South=3 West=4
        board[1] = PieceUtils.pack(5, 6, 7, 9);     // West=9 conflicts with East=2 of index 0
        board[16] = PieceUtils.pack(3, 8, 9, 10);   // caught in the conflict's erasure radius (south of index 0)
        board[2] = PieceUtils.pack(20, 21, 22, 23); // isolated piece outside the conflict radius

        int validPieces = (int) method.invoke(solver, (Object) board);

        assertEquals(-1, board[0], "Conflicting piece itself must be erased");
        assertEquals(-1, board[1], "Conflicting neighbor must be erased");
        assertEquals(-1, board[16], "Neighbor within the erasure radius must be erased even without its own conflict");
        assertEquals(PieceUtils.pack(20, 21, 22, 23), board[2], "Untouched piece outside the conflict radius must survive");
        assertEquals(1, validPieces, "Only the untouched piece should remain valid");
        assertTrue(solver.verifyBoardStrict(board), "Board must be conflict-free after rescue");
    }

    @Test
    void testSetVariantAndConflictSaveThresholds() throws Exception {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        solver.setVariantSaveThreshold(230);
        solver.setConflictSaveThreshold(15);

        assertEquals(230, getAtomicIntField(solver, "variantSaveThreshold"));
        assertEquals(15, getAtomicIntField(solver, "conflictSaveThreshold"));
    }

    @Test
    void testVerifyBoardStrictTreatsSurgeonHolesAsEmpty() {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        int[] board = new int[256];
        Arrays.fill(board, -1);

        // Index 1's West would conflict with index 0's East if it were a real
        // piece, but -2 marks a surgeon hole and must be skipped, not compared.
        board[0] = PieceUtils.pack(1, 2, 3, 4);
        board[1] = -2;

        assertTrue(solver.verifyBoardStrict(board), "Surgeon holes (-2) must not be treated as conflicting pieces");
    }

    private int computeValidPrefixLength(EternitySolver solver, int[] board) throws Exception {
        java.lang.reflect.Method method = EternitySolver.class.getDeclaredMethod("computeValidPrefixLength", int[].class);
        method.setAccessible(true);
        return (int) method.invoke(solver, (Object) board);
    }

    /** Fills a full 16x16 board with border-correct, mutually-matching pieces (zero conflicts anywhere). */
    private int[] buildCleanFullBoard() {
        int[] board = new int[256];
        for (int idx = 0; idx < 256; idx++) {
            int row = idx / 16, col = idx % 16;
            int north = (row == 0) ? PieceUtils.BORDER_COLOR : (100 + idx - 16);
            int west = (col == 0) ? PieceUtils.BORDER_COLOR : (100 + idx - 1);
            int east = (col == 15) ? PieceUtils.BORDER_COLOR : (100 + idx);
            int south = (row == 15) ? PieceUtils.BORDER_COLOR : (100 + idx);
            board[idx] = PieceUtils.pack(north, east, south, west);
        }
        return board;
    }

    @Test
    void testComputeValidPrefixLengthFullyCleanBoardCountsAllPieces() throws Exception {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        assertEquals(256, computeValidPrefixLength(solver, buildCleanFullBoard()));
    }

    @Test
    void testComputeValidPrefixLengthStopsAtNorthMismatch() throws Exception {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        int[] board = buildCleanFullBoard();
        // idx 20's north edge no longer matches idx 4's south - this is the only kind of
        // edge the typewriter method can actually check at placement time (the neighbor
        // above is already placed), so it must be caught immediately, right at idx 20.
        int p = board[20];
        board[20] = PieceUtils.pack(9999, PieceUtils.getEast(p), PieceUtils.getSouth(p), PieceUtils.getWest(p));

        assertEquals(20, computeValidPrefixLength(solver, board));
    }

    @Test
    void testComputeValidPrefixLengthStopsAtWestMismatch() throws Exception {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        int[] board = buildCleanFullBoard();
        // idx 21's west edge no longer matches idx 20's east.
        int p = board[21];
        board[21] = PieceUtils.pack(PieceUtils.getNorth(p), PieceUtils.getEast(p), PieceUtils.getSouth(p), 9999);

        assertEquals(21, computeValidPrefixLength(solver, board));
    }

    @Test
    void testComputeValidPrefixLengthDoesNotCheckSouthAgainstUnplacedNeighbor() throws Exception {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        int[] board = buildCleanFullBoard();
        // idx 40 (interior) gets a bad south edge. The typewriter method never checks
        // south against a neighbor that isn't placed yet, so the walk must sail straight
        // past idx 40 - it only notices once it reaches idx 56 and checks *that* tile's
        // north edge (the same physical edge, seen from the other, already-placed side).
        int p = board[40];
        board[40] = PieceUtils.pack(PieceUtils.getNorth(p), PieceUtils.getEast(p), 9999, PieceUtils.getWest(p));

        assertEquals(56, computeValidPrefixLength(solver, board),
                "South edges must only be caught later via the neighbor's own north check, never at the tile itself");
    }

    @Test
    void testComputeValidPrefixLengthDoesNotCheckEastAgainstUnplacedNeighbor() throws Exception {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        int[] board = buildCleanFullBoard();
        // idx 40 (interior) gets a bad east edge - only caught later at idx 41's west check.
        int p = board[40];
        board[40] = PieceUtils.pack(PieceUtils.getNorth(p), 9999, PieceUtils.getSouth(p), PieceUtils.getWest(p));

        assertEquals(41, computeValidPrefixLength(solver, board),
                "East edges must only be caught later via the neighbor's own west check, never at the tile itself");
    }

    @Test
    void testComputeValidPrefixLengthStopsAtTopRowWrongOutwardEdge() throws Exception {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        int[] board = buildCleanFullBoard();
        // idx 5 is on the top row (row 0); give it a non-grey north edge.
        int p = board[5];
        board[5] = PieceUtils.pack(9999, PieceUtils.getEast(p), PieceUtils.getSouth(p), PieceUtils.getWest(p));

        assertEquals(5, computeValidPrefixLength(solver, board));
    }

    @Test
    void testComputeValidPrefixLengthStopsAtBottomRowWrongOutwardEdge() throws Exception {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        int[] board = buildCleanFullBoard();
        // idx 241 is on the bottom row (row 15); its south border color is an intrinsic
        // property of the piece, knowable immediately - no neighbor needed to check it.
        int p = board[241];
        board[241] = PieceUtils.pack(PieceUtils.getNorth(p), PieceUtils.getEast(p), 9999, PieceUtils.getWest(p));

        assertEquals(241, computeValidPrefixLength(solver, board));
    }

    @Test
    void testComputeValidPrefixLengthStopsAtRightColumnWrongOutwardEdge() throws Exception {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        int[] board = buildCleanFullBoard();
        // idx 47 is on the right column (col 15); its east border color is likewise
        // intrinsic and checked immediately, unlike an actual neighbor comparison.
        int p = board[47];
        board[47] = PieceUtils.pack(PieceUtils.getNorth(p), 9999, PieceUtils.getSouth(p), PieceUtils.getWest(p));

        assertEquals(47, computeValidPrefixLength(solver, board));
    }

    @Test
    void testComputeValidPrefixLengthStopsAtEmptyCell() throws Exception {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        int[] board = buildCleanFullBoard();
        board[30] = -1;

        assertEquals(30, computeValidPrefixLength(solver, board));
    }

    @Test
    void testComputeValidPrefixLengthStopsAtSurgeonHole() throws Exception {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);

        int[] board = buildCleanFullBoard();
        board[60] = -2;

        assertEquals(60, computeValidPrefixLength(solver, board));
    }

    @Test
    void testComputeValidPrefixLengthIgnoresBuildOrderForLockedProfile() throws Exception {
        // lockCenter=true reorders buildOrder to visit the center hint (135) and
        // HINT_POSITIONS ({221, 45, 210, 34}) before anything else - a real bug
        // (Base 1 reported on a board that was genuinely Base 194+) happened
        // because the old implementation walked buildOrder and treated whatever
        // came earlier IN THAT ORDER as "already validated", when idx 221's true
        // geometric north neighbor (idx 205) was never actually checked first.
        // "Base" must reflect pure top-left-to-bottom-right geometric reading
        // order regardless of the solver's internal build/search order.
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, true);

        int[] board = buildCleanFullBoard();
        // Break idx 221's north edge against its true geometric neighbor (idx 205).
        // Under the old buildOrder-walk this surfaces at step 1 (right after the
        // center hint) -> validCount=1. Under pure geometric order, idx 0-220 are
        // all still genuinely clean, so the walk must reach all the way to 221.
        int p = board[221];
        board[221] = PieceUtils.pack(9999, PieceUtils.getEast(p), PieceUtils.getSouth(p), PieceUtils.getWest(p));

        assertEquals(221, computeValidPrefixLength(solver, board),
                "Base must reflect geometric reading order, not the locked profile's hint-first internal build order");
    }

    // ── promoteToGlobalRecordIfHigher ──

    private void promoteToGlobalRecordIfHigher(EternitySolver solver, int[] board, int displayScore) throws Exception {
        java.lang.reflect.Method method = EternitySolver.class.getDeclaredMethod(
                "promoteToGlobalRecordIfHigher", int[].class, int.class);
        method.setAccessible(true);
        method.invoke(solver, board, displayScore);
    }

    private void setSaveProfile(EternitySolver solver, String path) throws Exception {
        java.lang.reflect.Field field = EternitySolver.class.getDeclaredField("saveProfile");
        field.setAccessible(true);
        field.set(solver, path);
    }

    @Test
    void testPromoteToGlobalRecordIfHigherSavesAndUpdatesStateWhenGenuinelyHigher(@TempDir Path tempDir) throws Exception {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);
        setSaveProfile(solver, tempDir.resolve("profile").toString());

        int[] board = buildCleanFullBoard();
        assertEquals(0, solver.absoluteHighScore, "Freshly constructed solver should start with no record");

        promoteToGlobalRecordIfHigher(solver, board, 194);

        assertEquals(194, solver.absoluteHighScore, "A genuinely higher geometric prefix must become the new record");
        for (int i = 0; i < 194; i++) {
            assertEquals(board[i], solver.globalBestBoard[i], "First 194 cells of globalBestBoard must match the source board");
        }
        for (int i = 194; i < 256; i++) {
            assertEquals(-1, solver.globalBestBoard[i], "Cells beyond the valid geometric prefix must be cleared, not carried over with (possibly conflicting) content");
        }
        recordMock.verify(() -> RecordManager.saveRecord(any(), eq(194), anyString()));
    }

    @Test
    void testPromoteToGlobalRecordIfHigherDoesNothingWhenNotHigherThanCurrentRecord(@TempDir Path tempDir) throws Exception {
        EternitySolver solver = new EternitySolver(
                mockInventory, 0, false, EternitySolver.BuildStrategy.TYPEWRITER, false);
        setSaveProfile(solver, tempDir.resolve("profile").toString());
        solver.absoluteHighScore = 205;

        int[] board = buildCleanFullBoard();
        promoteToGlobalRecordIfHigher(solver, board, 205);

        assertEquals(205, solver.absoluteHighScore, "A tied (not strictly higher) score must not be treated as a new record");
        recordMock.verifyNoInteractions();
        assertFalse(Files.exists(tempDir.resolve("profile").resolve("bucas_link_205.txt")),
                "No record file should be written when the score does not beat the existing record");
    }

}
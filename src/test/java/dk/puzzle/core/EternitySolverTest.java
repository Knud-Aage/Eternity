package dk.puzzle.core;

import dk.puzzle.io.CheckpointManager;
import dk.puzzle.io.RecordManager;
import dk.puzzle.model.PieceInventory;
import dk.puzzle.util.PieceUtils;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.MockedStatic;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyString;
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

}
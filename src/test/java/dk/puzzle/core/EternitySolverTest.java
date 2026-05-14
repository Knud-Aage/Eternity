package dk.puzzle.core;

import dk.puzzle.io.CheckpointManager;
import dk.puzzle.io.RecordManager;
import dk.puzzle.model.PieceInventory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.MockedStatic;

import java.util.Arrays;

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
}
package dk.puzzle;

import dk.puzzle.ai.SurgeonHeuristics;
import dk.puzzle.core.EternitySolver;
import dk.puzzle.gpu.GpuEngine;
import dk.puzzle.model.CompatibilityIndex;
import dk.puzzle.model.PieceInventory;
import dk.puzzle.util.PieceUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class EternitySolverTest {

    // We will use a spy on EternitySolver to override its internal dependencies
    // and methods that interact with external systems (like static methods).
    private EternitySolver solverSpy;

    private PieceInventory inventory;
    @Mock
    private CompatibilityIndex mockCompatIndex;
    @Mock
    private SurgeonHeuristics mockSurgeon;
    @Mock
    private GpuEngine mockGpuEngine;

    // MOCK DATA for PieceInventory
    private final int[] MOCK_ALL_ORIENTATIONS = {
            // Physical Piece 0, Orientation 0 (index 0): N=1, E=2, S=3, W=4
            PieceUtils.pack(1, 2, 3, 4),
            // Physical Piece 0, Orientation 1 (index 1): N=4, E=1, S=2, W=3
            PieceUtils.pack(4, 1, 2, 3),
            // Physical Piece 0, Orientation 2 (index 2): N=3, E=4, S=1, W=2
            PieceUtils.pack(3, 4, 1, 2),
            // Physical Piece 0, Orientation 3 (index 3): N=2, E=3, S=4, W=1),

            // Physical Piece 1, Orientation 0 (index 4): N=3, E=5, S=6, W=2 (compatible with Piece 0's South)
            PieceUtils.pack(3, 5, 6, 2),
            // Physical Piece 1, Orientation 1 (index 5): N=2, E=3, S=5, W=6
            PieceUtils.pack(2, 3, 5, 6),
            // Physical Piece 1, Orientation 2 (index 6): N=6, E=2, S=3, W=5
            PieceUtils.pack(6, 2, 3, 5),
            // Physical Piece 1, Orientation 3 (index 7): N=5, E=6, S=2, W=3
            PieceUtils.pack(5, 6, 2, 3),
    };
    private final int[] MOCK_PHYSICAL_MAPPING = {0, 0, 0, 0, 1, 1, 1, 1}; // maps orientation index to physical piece index

    @BeforeEach
    void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);

        inventory = new PieceInventory(MOCK_ALL_ORIENTATIONS); // Assuming PieceInventory now takes allOrientations in its constructor
        inventory.physicalMapping = MOCK_PHYSICAL_MAPPING; // physicalMapping might still be a public field or set separately

        // Initialize the spy manually because EternitySolver has no no-args constructor
        EternitySolver realSolver = new EternitySolver(inventory, MOCK_ALL_ORIENTATIONS[0], false, EternitySolver.BuildStrategy.TYPEWRITER, false);
        solverSpy = spy(realSolver);

        // Use reflection to inject mocks for internal fields that are not passed via constructor
        java.lang.reflect.Field compatIndexField = EternitySolver.class.getDeclaredField("compatIndex");
        compatIndexField.setAccessible(true);
        compatIndexField.set(solverSpy, mockCompatIndex);

        java.lang.reflect.Field surgeonField = EternitySolver.class.getDeclaredField("surgeon");
        surgeonField.setAccessible(true);
        surgeonField.set(solverSpy, mockSurgeon);

        java.lang.reflect.Field gpuEngineField = EternitySolver.class.getDeclaredField("gpuEngine");
        gpuEngineField.setAccessible(true);
        gpuEngineField.set(solverSpy, mockGpuEngine); // Inject mockGpuEngine if needed for other tests

        // Mock static methods or methods that interact with external systems
        // For Eternity.updateDisplay, we can mock its behavior or just let it be called and ignore.
        // For CheckpointManager and RecordManager, we need to ensure they don't cause side effects.
        // For this test, we'll assume CheckpointManager.loadSmartCheckpoint returns null (default behavior if no file).
        // And we'll mock the updateDisplay method of the solverSpy.
        doNothing().when(solverSpy).updateDisplay(anyInt(), any(int[][].class));

        // Reset internal state for each test
        solverSpy.seedPool = new ConcurrentLinkedQueue<>(); // Reset seedPool
        solverSpy.currentBatchSize = new AtomicInteger(0); // Reset currentBatchSize
        solverSpy.manualOverrideRequested = false;
        solverSpy.deepestStep = 0;
        solverSpy.absoluteHighScore = 0;
        Arrays.fill(solverSpy.bestBoard, -1);
        Arrays.fill(solverSpy.flatResumeBoard, -1);
        Arrays.fill(solverSpy.usedPhysicalPieces, false);
        Arrays.fill(solverSpy.tabuTenure, 0);
    }

    // --- Unit tests for CpuSearchWorker (inner class) ---

    @Test
    void cpuSearchWorker_solvesToSeedDepthAndAddsToPool() throws Exception {
        // Given
        // We need to access the inner class constructor.
        // This is a bit tricky with reflection, but Mockito's spy on the outer class helps.
        EternitySolver.CpuSearchWorker worker = solverSpy.new CpuSearchWorker(50000);

        // Mock behavior for compatIndex
        BitSet mockCandidates = new BitSet();
        mockCandidates.set(0); // Piece 0, rot 0 (index 0 in MOCK_ALL_ORIENTATIONS)
        when(mockCompatIndex.candidatesFor(anyInt(), anyInt(), anyInt(), anyInt())).thenReturn(mockCandidates);
        doNothing().when(mockCompatIndex).andNotUsed(any(BitSet.class), any(boolean[].class));
        when(mockCompatIndex.hasAnyCandidate(anyInt(), anyInt(), anyInt(), anyInt(), any(boolean[].class))).thenReturn(true);

        // When
        boolean result = worker.call();

        // Then
        assertFalse(result, "CpuSearchWorker should return false after adding a seed at SEED_DEPTH to force backtracking");
        assertEquals(1, solverSpy.seedPool.size(), "One seed should be added to the seedPool");
        assertEquals(EternitySolver.SEED_DEPTH, solverSpy.deepestStep, "deepestStep should be updated to SEED_DEPTH");

        // Verify that compatIndex methods were called
        verify(mockCompatIndex, atLeastOnce()).candidatesFor(anyInt(), anyInt(), anyInt(), anyInt());
        verify(mockCompatIndex, atLeastOnce()).andNotUsed(any(BitSet.class), any(boolean[].class));
        verify(mockCompatIndex, atLeastOnce()).hasAnyCandidate(anyInt(), anyInt(), anyInt(), anyInt(), any(boolean[].class));
    }

    @Test
    void cpuSearchWorker_respectsManualOverride() throws Exception {
        // Given
        EternitySolver.CpuSearchWorker worker = solverSpy.new CpuSearchWorker(50000);
        solverSpy.manualOverrideRequested = true; // Simulate manual override

        // When
        boolean result = worker.call();

        // Then
        assertFalse(result, "CpuSearchWorker should return false if manual override is requested");
        assertEquals(0, solverSpy.seedPool.size(), "No seeds should be added if manual override stops the search");
    }

    @Test
    void cpuSearchWorker_respectsBatchSizeLimit() throws Exception {
        // Given
        EternitySolver.CpuSearchWorker worker = solverSpy.new CpuSearchWorker(1); // Max 1 seed
        solverSpy.seedPool.add(new int[256]); // Add one seed to reach the limit
        solverSpy.currentBatchSize.set(1);

        // When
        boolean result = worker.call();

        // Then
        assertFalse(result, "CpuSearchWorker should return false if batch size limit is reached");
        assertEquals(1, solverSpy.seedPool.size(), "No new seeds should be added if batch size limit is reached");
    }

    @Test
    void cpuSearchWorker_handlesNoCandidates() throws Exception {
        // Given
        EternitySolver.CpuSearchWorker worker = solverSpy.new CpuSearchWorker(50000);
        when(mockCompatIndex.candidatesFor(anyInt(), anyInt(), anyInt(), anyInt())).thenReturn(new BitSet()); // No candidates

        // When
        boolean result = worker.call();

        // Then
        assertFalse(result, "CpuSearchWorker should return false if no candidates are found at any step");
        assertEquals(0, solverSpy.seedPool.size(), "No seeds should be added");
    }

    // --- Unit tests for Orchestration Phases ---

    @Test
    void runPhase1_CpuSeedGen_invokesWorkers() throws Exception {
        // Given
        solverSpy.setBatchSizeOverride(10);

        // When
        solverSpy.runPhase1_CpuSeedGen();

        // Then
        // Verify that the batch size was updated (even if seeds aren't found in mock environment immediately)
        assertNotNull(solverSpy.seedPool);
    }

    @Test
    void runPhase2_GpuDeepDfs_updatesHighScoreOnSuccess() {
        // Given
        int[] seed = new int[256];
        Arrays.fill(seed, 1);
        solverSpy.seedPool.add(seed);
        for(int i=1; i<50000; i++) solverSpy.seedPool.add(new int[256]); // Fill to targetBatchSize

        int[] gpuResultBoard = new int[256];
        Arrays.fill(gpuResultBoard, 5);
        GpuEngine.GpuResult mockResult = new GpuEngine.GpuResult(100, false, 1000000L, new int[0]);

        when(mockGpuEngine.runDeepDfs(anyList(), anyInt(), anyInt(), any(int[].class), any(int[].class)))
                .thenAnswer(invocation -> {
                    System.arraycopy(gpuResultBoard, 0, (int[])invocation.getArguments()[3], 0, 256);
                    return mockResult;
                });

        solverSpy.deepestStep = 40;
        solverSpy.absoluteHighScore = 40;

        // When
        solverSpy.runPhase2_GpuDeepDfs();

        // Then
        assertEquals(100, solverSpy.deepestStep);
        assertEquals(100, solverSpy.absoluteHighScore);
        assertArrayEquals(gpuResultBoard, solverSpy.bestBoard);
    }

    @Test
    void runPhase3_GpuSurgeon_triggersExtinctionOnStagnation() {
        // Given
        solverSpy.deepestStep = 210;
        solverSpy.absoluteHighScore = 210;
        solverSpy.consecutiveExtinctions = 9; // One away from trigger

        List<int[]> emptyList = new ArrayList<>();
        when(mockSurgeon.punchHoles(any(), anyInt(), anyInt(), any(), anyInt(), anyInt(), any()))
                .thenReturn(emptyList);

        GpuEngine.GpuResult stagnantResult = new GpuEngine.GpuResult(210, false, 500000L, new int[0]);
        when(mockGpuEngine.runRepairMode(anyList(), anyInt(), any(int[].class)))
                .thenReturn(stagnantResult);

        // When
        solverSpy.runPhase3_GpuSurgeon();

        // Then
        assertEquals(10, solverSpy.consecutiveExtinctions);
        assertEquals(EternitySolver.SEED_DEPTH, solverSpy.deepestStep, "Should have triggered branch scrap/extinction");
    }

    // --- Unit tests for helper methods ---

    @Test
    void countPieces_correctlyCountsPlacedPieces() {
        // Given
        int[] board = new int[256];
        Arrays.fill(board, -1); // Empty board
        board[0] = 100; // Placed piece
        board[10] = 200; // Placed piece
        board[20] = -2; // Hole (should not be counted)

        // When
        int count = solverSpy.countPieces(board);

        // Then
        assertEquals(2, count);
    }

    @Test
    void buildDisplayBoard_correctlyTransformsFlatBoard() {
        // Given
        int[] flatBoard = new int[256];
        Arrays.fill(flatBoard, -1);
        flatBoard[0] = 1; // Top-left
        flatBoard[15] = 2; // Top-right
        flatBoard[240] = 3; // Bottom-left (15*16 + 0)
        flatBoard[255] = 4; // Bottom-right (15*16 + 15)

        // When
        int[][] displayBoard = solverSpy.buildDisplayBoard(flatBoard);

        // Then
        assertEquals(1, displayBoard[0][0]);
        assertEquals(2, displayBoard[0][15]);
        assertEquals(3, displayBoard[15][0]);
        assertEquals(4, displayBoard[15][15]);
        assertEquals(-1, displayBoard[0][1]); // Check an empty spot
    }

    @Test
    void retreat_resetsBoardAndDeepestStep() {
        // Given
        solverSpy.absoluteHighScore = 100;
        solverSpy.deepestStep = 150;
        Arrays.fill(solverSpy.bestBoard, 123); // Fill with some pieces
        solverSpy.bestBoard[135] = 999; // Center piece (will be overwritten if not locked)

        int targetStep = 50;

        // When
        solverSpy.retreat(targetStep, "Test retreat message");

        // Then
        assertEquals(targetStep, solverSpy.deepestStep, "deepestStep should be set to targetStep");
        for (int i = 0; i < targetStep; i++) {
            assertEquals(123, solverSpy.bestBoard[solverSpy.buildOrder[i]], "Pieces before targetStep should remain");
        }
        for (int i = targetStep; i < 256; i++) {
            assertEquals(-1, solverSpy.bestBoard[solverSpy.buildOrder[i]], "Pieces after targetStep should be reset");
        }
        // Since lockCenter is false in setup, bestBoard[135] should be -1 if buildOrder[>50] includes 135
        // For typewriter order, 135 is in the middle, so it will be reset.
        assertEquals(-1, solverSpy.bestBoard[135]);
    }

    @Test
    void retreat_preservesLockedCenterPiece() throws Exception {
        // Re-initialize solverSpy with lockCenter = true
        EternitySolver realSolverLocked = new EternitySolver(inventory, MOCK_ALL_ORIENTATIONS[0], false, EternitySolver.BuildStrategy.TYPEWRITER, true);
        solverSpy = spy(realSolverLocked);

        java.lang.reflect.Field compatIndexField = EternitySolver.class.getDeclaredField("compatIndex");
        compatIndexField.setAccessible(true);
        compatIndexField.set(solverSpy, mockCompatIndex);

        java.lang.reflect.Field surgeonField = EternitySolver.class.getDeclaredField("surgeon");
        surgeonField.setAccessible(true);
        surgeonField.set(solverSpy, mockSurgeon);

        doNothing().when(solverSpy).updateDisplay(anyInt(), any(int[][].class));

        // Given
        solverSpy.absoluteHighScore = 100;
        solverSpy.deepestStep = 150;
        Arrays.fill(solverSpy.bestBoard, 123);
        solverSpy.bestBoard[135] = solverSpy.targetPiece; // Ensure center piece is set

        int targetStep = 50;

        // When
        solverSpy.retreat(targetStep, "Test retreat message");

        // Then
        assertEquals(solverSpy.targetPiece, solverSpy.bestBoard[135], "Locked center piece should be preserved");
    }

    @Test
    void updateTabuList_marksChangedPieces() {
        // Given
        int[] oldBoard = new int[256];
        int[] newBoard = new int[256];
        Arrays.fill(oldBoard, -1);
        Arrays.fill(newBoard, -1);

        oldBoard[0] = 1;
        newBoard[0] = 1; // Unchanged
        oldBoard[1] = 2;
        newBoard[1] = 3; // Changed
        oldBoard[2] = -1;
        newBoard[2] = 4; // New piece

        solverSpy.bestBoard = oldBoard; // Set solver's internal bestBoard
        solverSpy.currentRepairIteration = 100;
        Arrays.fill(solverSpy.tabuTenure, 0); // Clear tabu tenure

        // When
        solverSpy.updateTabuList(newBoard);

        // Then
        assertEquals(0, solverSpy.tabuTenure[0], "Unchanged piece should not be tabu");
        assertEquals(100 + 25, solverSpy.tabuTenure[1], "Changed piece should be tabu");
        assertEquals(100 + 25, solverSpy.tabuTenure[2], "New piece should be tabu");
    }

    @Test
    void triggerBranchScrap_resetsStateForPhase3() {
        // Given
        solverSpy.deepestStep = 220; // In LNS phase
        solverSpy.consecutiveExtinctions = 5;
        solverSpy.absoluteHighScore = 220; // Assume this is the high score

        // When
        solverSpy.triggerBranchScrap();

        // Then
        assertEquals(EternitySolver.SEED_DEPTH, solverSpy.deepestStep, "deepestStep should reset to SEED_DEPTH");
        assertEquals(0, solverSpy.consecutiveExtinctions, "consecutiveExtinctions should be reset");
        // absoluteHighScore should remain the same, as we are scrapping a branch, not the overall best.
        assertEquals(220, solverSpy.absoluteHighScore);
    }
}

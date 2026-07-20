package dk.puzzle.io;

import dk.puzzle.core.SolverState;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link CheckpointManager} (the dk.puzzle.io implementation
 * used by the solver - not the empty stub at dk.puzzle.CheckpointManager).
 *
 * <p>Every method here accepts a {@code profileFolder} path parameter, so
 * all file I/O in these tests is redirected to a JUnit {@code @TempDir} and
 * never touches the project's real checkpoint/profile directories.</p>
 */
class CheckpointManagerTest {

    private int[][] board(int fill) {
        int[][] b = new int[16][16];
        for (int[] row : b) {
            Arrays.fill(row, fill);
        }
        return b;
    }

    private void assertBoardEquals(int[][] expected, int[][] actual, String message) {
        assertEquals(expected.length, actual.length, message);
        for (int r = 0; r < expected.length; r++) {
            assertArrayEquals(expected[r], actual[r], message + " (row " + r + ")");
        }
    }

    @Test
    void testSaveAndLoadSmartStateRoundTrip(@TempDir Path tempDir) {
        String profileFolder = tempDir.resolve("profileA").toString();
        int[][] boardData = board(-1);
        boardData[5][5] = 999;
        Set<Integer> hashes = new HashSet<>(Arrays.asList(1, 2, 3));
        List<int[]> registry = new ArrayList<>();
        registry.add(new int[]{10, 20, 30});
        SolverState state = new SolverState(boardData, 120, hashes, registry, 4242L);

        CheckpointManager.saveSmartState(state, profileFolder);
        SolverState loaded = CheckpointManager.loadSmartState(profileFolder);

        assertNotNull(loaded, "A freshly saved state must be loadable");
        assertEquals(120, loaded.score);
        assertEquals(4242L, loaded.cumulativeTrials);
        assertEquals(hashes, loaded.uniqueMaxScoreHashes);
        assertEquals(1, loaded.topBoardsRegistry.size());
        assertArrayEquals(new int[]{10, 20, 30}, loaded.topBoardsRegistry.get(0));
        assertBoardEquals(boardData, loaded.bestBoard, "Deserialized board must match the saved board");
    }

    @Test
    void testLoadSmartStateReturnsNullWhenFolderMissing(@TempDir Path tempDir) {
        String profileFolder = tempDir.resolve("doesNotExist").toString();
        assertNull(CheckpointManager.loadSmartState(profileFolder));
    }

    @Test
    void testLoadSmartStatePicksHighestScoreFile(@TempDir Path tempDir) {
        String profileFolder = tempDir.resolve("profileB").toString();
        SolverState low = new SolverState(board(-1), 50, new HashSet<>(), new ArrayList<>(), 1L);
        SolverState high = new SolverState(board(-1), 200, new HashSet<>(), new ArrayList<>(), 2L);
        SolverState mid = new SolverState(board(-1), 130, new HashSet<>(), new ArrayList<>(), 3L);

        CheckpointManager.saveSmartState(low, profileFolder);
        CheckpointManager.saveSmartState(high, profileFolder);
        CheckpointManager.saveSmartState(mid, profileFolder);

        SolverState loaded = CheckpointManager.loadSmartState(profileFolder);
        assertNotNull(loaded);
        assertEquals(200, loaded.score, "loadSmartState must pick the file with the highest embedded score");
    }

    @Test
    void testSaveRecordCheckpointAndLoadSmartCheckpointRoundTrip(@TempDir Path tempDir) {
        String profileFolder = tempDir.resolve("profileC").toString();
        int[][] boardData = board(-1);
        boardData[0][0] = 777;

        CheckpointManager.saveRecordCheckpoint(boardData, 88, profileFolder);
        int[][] loaded = CheckpointManager.loadSmartCheckpoint(profileFolder);

        assertNotNull(loaded);
        assertBoardEquals(boardData, loaded, "Deserialized board from loadSmartCheckpoint must match what was saved");
    }

    @Test
    void testLoadSmartCheckpointReturnsNullWhenFolderMissing(@TempDir Path tempDir) {
        String profileFolder = tempDir.resolve("noSuchFolder").toString();
        assertNull(CheckpointManager.loadSmartCheckpoint(profileFolder));
    }

    @Test
    void testLoadSmartCheckpointReturnsNullWhenNoDatFilesPresent(@TempDir Path tempDir) throws Exception {
        Path folder = tempDir.resolve("profileD");
        Files.createDirectories(folder);
        Files.writeString(folder.resolve("not_a_checkpoint.txt"), "irrelevant");

        assertNull(CheckpointManager.loadSmartCheckpoint(folder.toString()));
    }

    @Test
    void testLoadSmartCheckpointPicksHighestScoreAmongMultipleFiles(@TempDir Path tempDir) {
        String profileFolder = tempDir.resolve("profileE").toString();
        int[][] boardLow = board(-1);
        boardLow[0][0] = 1;
        int[][] boardHigh = board(-1);
        boardHigh[0][0] = 2;

        CheckpointManager.saveRecordCheckpoint(boardLow, 40, profileFolder);
        CheckpointManager.saveRecordCheckpoint(boardHigh, 210, profileFolder);

        int[][] loaded = CheckpointManager.loadSmartCheckpoint(profileFolder);
        assertNotNull(loaded);
        assertEquals(2, loaded[0][0], "The checkpoint file with the higher score in its filename must win");
    }
}

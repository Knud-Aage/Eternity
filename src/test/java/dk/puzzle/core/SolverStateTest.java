package dk.puzzle.core;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for SolverState, a plain Serializable data holder used to
 * checkpoint solver progress. Currently unused/dead code (part of an
 * in-progress extraction from the EternitySolver monolith) but its
 * constructor wiring and serializability are deterministic and worth
 * locking down. All fields are public, so no reflection is needed here.
 */
class SolverStateTest {

    private int[][] sampleBestBoard() {
        return new int[][]{{1, 2}, {3, 4}};
    }

    @Test
    void testConstructorAssignsAllFieldsVerbatim() {
        int[][] bestBoard = sampleBestBoard();
        Set<Integer> hashes = new HashSet<>(Arrays.asList(10, 20, 30));
        List<int[]> registry = new ArrayList<>();
        registry.add(new int[]{5, 6});

        SolverState state = new SolverState(bestBoard, 187, hashes, registry, 12345L);

        assertSame(bestBoard, state.bestBoard, "Constructor must not defensively copy bestBoard");
        assertEquals(187, state.score);
        assertSame(hashes, state.uniqueMaxScoreHashes, "Constructor must not defensively copy the hash set");
        assertSame(registry, state.topBoardsRegistry, "Constructor must not defensively copy the registry list");
        assertEquals(12345L, state.cumulativeTrials);
    }

    @Test
    void testBestBoardFieldAliasesCallerArray() {
        int[][] bestBoard = sampleBestBoard();
        SolverState state = new SolverState(bestBoard, 100, new HashSet<>(), new ArrayList<>(), 0L);

        bestBoard[0][0] = 999; // mutate the caller's array after construction

        assertEquals(999, state.bestBoard[0][0], "bestBoard is stored by reference, not copied");
    }

    @Test
    void testCumulativeTrialsIsMutableAfterConstruction() {
        SolverState state = new SolverState(sampleBestBoard(), 1, new HashSet<>(), new ArrayList<>(), 0L);

        state.cumulativeTrials = 999L;

        assertEquals(999L, state.cumulativeTrials, "cumulativeTrials is not final and must be reassignable");
    }

    @Test
    void testConstructorAllowsNullCollectionsWithoutValidation() {
        SolverState state = new SolverState(null, 0, null, null, 0L);

        assertNull(state.bestBoard);
        assertNull(state.uniqueMaxScoreHashes);
        assertNull(state.topBoardsRegistry);
    }

    @Test
    void testInstanceIsFullySerializableAndRoundTrips() throws Exception {
        int[][] bestBoard = sampleBestBoard();
        Set<Integer> hashes = new HashSet<>(Arrays.asList(1, 2, 3));
        List<int[]> registry = new ArrayList<>();
        registry.add(new int[]{7, 8, 9});
        SolverState original = new SolverState(bestBoard, 250, hashes, registry, 42L);

        ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
        try (ObjectOutputStream out = new ObjectOutputStream(byteStream)) {
            out.writeObject(original);
        }

        SolverState restored;
        try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(byteStream.toByteArray()))) {
            restored = (SolverState) in.readObject();
        }

        assertEquals(original.bestBoard.length, restored.bestBoard.length);
        for (int i = 0; i < original.bestBoard.length; i++) {
            assertArrayEquals(original.bestBoard[i], restored.bestBoard[i],
                    "Row " + i + " of bestBoard must survive round-trip serialization");
        }
        assertEquals(original.score, restored.score);
        assertEquals(original.cumulativeTrials, restored.cumulativeTrials);
        assertEquals(original.uniqueMaxScoreHashes, restored.uniqueMaxScoreHashes);

        assertEquals(original.topBoardsRegistry.size(), restored.topBoardsRegistry.size());
        for (int i = 0; i < original.topBoardsRegistry.size(); i++) {
            assertArrayEquals(original.topBoardsRegistry.get(i), restored.topBoardsRegistry.get(i),
                    "Each board in topBoardsRegistry must survive round-trip serialization");
        }
    }
}

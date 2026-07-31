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

        SolverState state = new SolverState(bestBoard, 187, hashes, registry, 12345L, 42);

        assertSame(bestBoard, state.bestBoard, "Constructor must not defensively copy bestBoard");
        assertEquals(187, state.score);
        assertSame(hashes, state.uniqueMaxScoreHashes, "Constructor must not defensively copy the hash set");
        assertSame(registry, state.topBoardsRegistry, "Constructor must not defensively copy the registry list");
        assertEquals(12345L, state.cumulativeTrials);
        assertEquals(42, state.lowestConflicts);
    }

    @Test
    void testBestBoardFieldAliasesCallerArray() {
        int[][] bestBoard = sampleBestBoard();
        SolverState state = new SolverState(bestBoard, 100, new HashSet<>(), new ArrayList<>(), 0L, null);

        bestBoard[0][0] = 999; // mutate the caller's array after construction

        assertEquals(999, state.bestBoard[0][0], "bestBoard is stored by reference, not copied");
    }

    @Test
    void testCumulativeTrialsIsMutableAfterConstruction() {
        SolverState state = new SolverState(sampleBestBoard(), 1, new HashSet<>(), new ArrayList<>(), 0L, null);

        state.cumulativeTrials = 999L;

        assertEquals(999L, state.cumulativeTrials, "cumulativeTrials is not final and must be reassignable");
    }

    @Test
    void testConstructorAllowsNullCollectionsWithoutValidation() {
        SolverState state = new SolverState(null, 0, null, null, 0L, null);

        assertNull(state.bestBoard);
        assertNull(state.uniqueMaxScoreHashes);
        assertNull(state.topBoardsRegistry);
        assertNull(state.lowestConflicts);
    }

    @Test
    void testInstanceIsFullySerializableAndRoundTrips() throws Exception {
        int[][] bestBoard = sampleBestBoard();
        Set<Integer> hashes = new HashSet<>(Arrays.asList(1, 2, 3));
        List<int[]> registry = new ArrayList<>();
        registry.add(new int[]{7, 8, 9});
        SolverState original = new SolverState(bestBoard, 250, hashes, registry, 42L, 17);

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
        assertEquals(original.lowestConflicts, restored.lowestConflicts);

        assertEquals(original.topBoardsRegistry.size(), restored.topBoardsRegistry.size());
        for (int i = 0; i < original.topBoardsRegistry.size(); i++) {
            assertArrayEquals(original.topBoardsRegistry.get(i), restored.topBoardsRegistry.get(i),
                    "Each board in topBoardsRegistry must survive round-trip serialization");
        }
    }

    @Test
    void testNullLowestConflictsRoundTripsAsNull() throws Exception {
        // lowestConflicts is boxed specifically so that checkpoints written
        // before this field existed deserialize it as null rather than 0 --
        // 0 would misread as "an unbeatable conflict record" and permanently
        // block the conflict-record feature from ever firing again. null is
        // what a legacy stream actually produces for an absent object field,
        // so exercising it here stands in for that backward-compat path.
        SolverState original = new SolverState(sampleBestBoard(), 250, new HashSet<>(), new ArrayList<>(), 42L, null);

        ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
        try (ObjectOutputStream out = new ObjectOutputStream(byteStream)) {
            out.writeObject(original);
        }

        SolverState restored;
        try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(byteStream.toByteArray()))) {
            restored = (SolverState) in.readObject();
        }

        assertNull(restored.lowestConflicts, "Absent/legacy conflict record must deserialize as null, not 0");
    }
}

package dk.puzzle.core; // Adjust package to your structure

import java.io.Serializable;
import java.util.List;
import java.util.Set;

public class SolverState implements Serializable {
    private static final long serialVersionUID = 1L; // Required for serialization

    public final int[][] bestBoard;
    public final int score;
    public long cumulativeTrials;

    // The "Memory" of the solver
//    public final int[] tabuTenure;
    public final Set<Integer> uniqueMaxScoreHashes;
    public final List<int[]> topBoardsRegistry;
    // All-time-low edge-conflict count, the counterpart to `score` (the depth
    // record) -- see EternitySolver.lowestConflictsEver. Boxed rather than a
    // primitive int so that checkpoints written before this field existed
    // deserialize it as null (Java's default handling for an object-typed
    // field absent from the stream) instead of silently defaulting to 0,
    // which would look like a genuine (and unbeatable) record.
    public final Integer lowestConflicts;

    public SolverState(int[][] bestBoard, int score, Set<Integer> uniqueHashes, List<int[]> registry, long cumulativeTrials, Integer lowestConflicts) {
        this.bestBoard = bestBoard;
        this.score = score;
//        this.tabuTenure = tabuTenure;
        this.uniqueMaxScoreHashes = uniqueHashes;
        this.topBoardsRegistry = registry;
        this.cumulativeTrials = cumulativeTrials;
        this.lowestConflicts = lowestConflicts;
    }
}
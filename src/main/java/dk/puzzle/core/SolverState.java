package dk.puzzle.core; // Adjust package to your structure

import java.io.Serializable;
import java.util.List;
import java.util.Set;

public class SolverState implements Serializable {
    private static final long serialVersionUID = 1L; // Required for serialization

    public final int[][] bestBoard;
    public final int score;

    // The "Memory" of the solver
    public final int[] tabuTenure;
    public final Set<Integer> uniqueMaxScoreHashes;
    public final List<int[]> topBoardsRegistry;

    public SolverState(int[][] bestBoard, int score, int[] tabuTenure, Set<Integer> uniqueHashes, List<int[]> registry) {
        this.bestBoard = bestBoard;
        this.score = score;
        this.tabuTenure = tabuTenure;
        this.uniqueMaxScoreHashes = uniqueHashes;
        this.topBoardsRegistry = registry;
    }
}
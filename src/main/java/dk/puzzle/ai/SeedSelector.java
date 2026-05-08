package dk.puzzle.ai;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Selects and evolves board configurations (seeds) to be used by the puzzle solver.
 * 
 * <p>This class implements a heuristic scoring system and a selection mechanism 
 * resembling a genetic algorithm, prioritizing boards with more pieces, better 
 * placement distribution, and fewer "trapped" empty slots.</p>
 */
public class SeedSelector {

    private static final int[] POSITION_WEIGHT = buildPositionWeights();

    private static int[] buildPositionWeights() {
        int[] w = new int[256];
        for (int i = 0; i < 256; i++) {
            int row = i / 16;
            int col = i % 16;
            int distFromEdge = Math.min(Math.min(row, 15 - row), Math.min(col, 15 - col));
            w[i] = distFromEdge + 1;
        }
        return w;
    }

    /**
     * Evaluates the "fitness" of a given board state.
     * 
     * <p>The score is calculated based on:
     * <ul>
     *     <li><b>Depth:</b> The number of pieces successfully placed (heavily weighted).</li>
     *     <li><b>Position:</b> Favoring pieces placed further from the edges.</li>
     *     <li><b>Danger Penalty:</b> Penalizing empty slots that have 3 or more neighbors, as these are difficult to fill.</li>
     * </ul></p>
     *
     * @param board An array of 256 integers representing the current board state.
     * @param depthReached The number of pieces placed or the search depth attained for this specific board.
     * @return A heuristic score where higher values indicate a more promising board state.
     */
    public static int scoreBoard(int[] board, int depthReached) {
        int placed = 0;
        int posScore = 0;
        int dangerPenalty = 0;

        for (int i = 0; i < 256; i++) {
            if (board[i] != -1) {
                placed++;
                posScore += POSITION_WEIGHT[i];
            }
        }

        for (int i = 0; i < 256; i++) {
            if (board[i] == -1) {
                int row = i / 16;
                int col = i % 16;
                int filledNeighbours = 0;
                if (row > 0 && board[i - 16] != -1) filledNeighbours++;
                if (row < 15 && board[i + 16] != -1) filledNeighbours++;
                if (col > 0 && board[i - 1] != -1) filledNeighbours++;
                if (col < 15 && board[i + 1] != -1) filledNeighbours++;
                if (filledNeighbours >= 3) dangerPenalty += (filledNeighbours - 2) * 5;
            }
        }

        return depthReached * 100 + posScore * 2 - dangerPenalty;
    }

    /**
     * Selects a subset of boards from a population, applying elitism and mutation.
     * 
     * <p>The selection strategy produces the {@code targetCount} by:
     * 1. Selecting the top 10% (Elites) as exact copies.
     * 2. Selecting and heavily mutating the top 40% (Exploration).
     * 3. Filling the remainder with lightly mutated versions of the top 50% (Refinement).</p>
     *
     * @param allBoards A list of board configurations to choose from.
     * @param threadDepths An array where each index corresponds to the depth reached by the board at the same index in {@code allBoards}.
     * @param targetCount The desired number of boards to return.
     * @param random A {@link Random} instance for mutation and selection.
     * @return A new list of selected and potentially mutated board configurations.
     */
    public static List<int[]> selectBest(
            List<int[]> allBoards,
            int[] threadDepths,
            int targetCount,
            Random random) {
        int n = allBoards.size();
        if (n == 0) return new ArrayList<>();

        int[] scores = new int[n];
        for (int i = 0; i < n; i++) {
            scores[i] = scoreBoard(allBoards.get(i), i < threadDepths.length ? threadDepths[i] : 0);
        }

        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> scores[b] - scores[a]);

        int eliteCount = Math.max(1, targetCount / 10);
        int mutateCount = Math.max(1, targetCount * 4 / 10);
        int fillCount = targetCount - eliteCount - mutateCount;

        List<int[]> result = new ArrayList<>(targetCount);

        for (int i = 0; i < eliteCount && i < n; i++) {
            result.add(Arrays.copyOf(allBoards.get(indices[i]), 256));
        }

        for (int i = 0; i < mutateCount && i < n; i++) {
            int[] mutated = Arrays.copyOf(allBoards.get(indices[i]), 256);
            mutate(mutated, random, 2 + random.nextInt(2));
            result.add(mutated);
        }

        int top50 = Math.max(1, n / 2);
        for (int i = 0; i < fillCount; i++) {
            int srcIdx = indices[random.nextInt(top50)];
            int[] copy = Arrays.copyOf(allBoards.get(srcIdx), 256);
            mutate(copy, random, 1);
            result.add(copy);
        }

        return result;
    }

    private static void mutate(int[] board, Random random, int swaps) {
        List<Integer> innerPlaced = new ArrayList<>();
        for (int i = 0; i < 256; i++) {
            if (board[i] != -1) {
                int row = i / 16;
                int col = i % 16;
                if (row > 0 && row < 15 && col > 0 && col < 15) {
                    innerPlaced.add(i);
                }
            }
        }

        for (int s = 0; s < swaps && innerPlaced.size() >= 2; s++) {
            int idxA = random.nextInt(innerPlaced.size());
            int idxB = random.nextInt(innerPlaced.size());
            if (idxA == idxB) continue;
            int posA = innerPlaced.get(idxA);
            int posB = innerPlaced.get(idxB);
            // Swap
            int tmp = board[posA];
            board[posA] = board[posB];
            board[posB] = tmp;
        }
    }
}
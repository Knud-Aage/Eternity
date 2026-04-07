package dk.roleplay;

import java.util.List;

/**
 * Utility class for sorting macro-tile candidates based on a heuristic score.
 * The heuristic favors candidates that use more common colors, preserving rare colors
 * for parts of the board where they might be strictly required, thus improving
 * backtracking efficiency.
 */
public class HeuristicSorter {
    /**
     * Sorts a list of macro-tile candidates based on color frequency heuristics.
     *
     * @param candidates  The list of 16-piece macro-tiles to be sorted.
     * @param frequencies An array containing the global frequency of each color ID.
     */
    public static void sort(List<int[]> candidates, int[] frequencies) {
        candidates.sort((a, b) -> Double.compare(score(a, frequencies), score(b, frequencies)));
    }

    private static double score(int[] tile, int[] freq) {
        double score = 0;
        for (int p : tile) {
            score += 1.0 / (freq[PieceUtils.getNorth(p)] + 1);
            score += 1.0 / (freq[PieceUtils.getEast(p)] + 1);
            score += 1.0 / (freq[PieceUtils.getSouth(p)] + 1);
            score += 1.0 / (freq[PieceUtils.getWest(p)] + 1);
        }
        return score;
    }
}

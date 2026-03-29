package dk.roleplay;

import java.util.List;

public class HeuristicSorter {
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

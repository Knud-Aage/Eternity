package dk.roleplay;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * CPU-based implementation of the CandidateValidator.
 * Uses parallel streams to perform edge-match validation on batches of candidates.
 */
public class CpuValidator implements CandidateValidator {

    /**
     * Validates a batch of candidates using CPU parallel processing.
     * 
     * @param candidates Flattened array of 16-piece sets
     * @param numPermutations Number of 4x4 sets in the candidates array
     * @return List of internally valid 4x4 macro-tiles
     */
    @Override
    public List<int[]> validate(int[] candidates, int numPermutations) {
        return IntStream.range(0, numPermutations)
            .parallel()
            .filter(i -> isInternallyValid(candidates, i * 16))
            .mapToObj(i -> {
                int[] tile = new int[16];
                System.arraycopy(candidates, i * 16, tile, 0, 16);
                return tile;
            })
            .collect(Collectors.toList());
    }

    private boolean isInternallyValid(int[] candidates, int base) {
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 3; c++) {
                int leftIdx = base + (r * 4) + c;
                int rightIdx = leftIdx + 1;
                if (PieceUtils.getEast(candidates[leftIdx]) != PieceUtils.getWest(candidates[rightIdx])) return false;
            }
        }

        for (int c = 0; c < 4; c++) {
            for (int r = 0; r < 3; r++) {
                int topIdx = base + (r * 4) + c;
                int bottomIdx = topIdx + 4;
                if (PieceUtils.getSouth(candidates[topIdx]) != PieceUtils.getNorth(candidates[bottomIdx])) return false;
            }
        }
        return true;
    }
}

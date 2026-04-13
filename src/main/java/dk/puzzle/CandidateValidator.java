package dk.puzzle;

import java.util.List;

/**
 * Interface for validating batches of 4x4 macro-tile candidates.
 * Implementations can provide hardware-specific validation logic (e.g., GPU or CPU).
 */
public interface CandidateValidator {
    /**
     * Validates a batch of candidate macro-tiles to ensure internal edge consistency.
     *
     * @param candidateBatch  Flattened 1D array of 32-bit packed integers
     * @param numPermutations Total number of 16-piece candidates in the batch
     * @return A list of validated 16-piece arrays that are internally consistent
     */
    List<int[]> validate(int[] candidateBatch, int numPermutations);
}

package dk.roleplay;

import java.util.List;

public interface CandidateValidator {
    List<int[]> validate(int[] candidateBatch, int numPermutations);
}

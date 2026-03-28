package dk.roleplay;

import java.util.List;

public interface CandidateValidator {
    /**
     * @param candidates Flattened array of 16-piece sets (N*16 integers)
     * @return List of internally valid 16-piece sets
     */
    List<int[]> validate(int[] candidates);
}

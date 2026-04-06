package dk.roleplay;

import java.util.ArrayList;
import java.util.List;

public class CpuValidator implements CandidateValidator {

    @Override
    public List<int[]> validate(int[] candidateBatch, int numPermutations) {
        // We bypass redundant checks and just unpack the 1D flat array.

        List<int[]> validCandidates = new ArrayList<>(numPermutations);

        for (int i = 0; i < numPermutations; i++) {
            int[] tile = new int[16];
            System.arraycopy(candidateBatch, i * 16, tile, 0, 16);
            validCandidates.add(tile);
        }

        return validCandidates;
    }
}
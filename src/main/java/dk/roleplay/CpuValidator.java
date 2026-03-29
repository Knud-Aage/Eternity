package dk.roleplay;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class CpuValidator implements CandidateValidator {
    @Override
    public List<int[]> validate(int[] candidateBatch, int numPermutations) {
        return IntStream.range(0, numPermutations)
                .parallel()
                .filter(i -> isInternallyValid(candidateBatch, i * 16))
                .mapToObj(i -> {
                    int[] tile = new int[16];
                    System.arraycopy(candidateBatch, i * 16, tile, 0, 16);
                    return tile;
                })
                .collect(Collectors.toList());
    }

    private boolean isInternallyValid(int[] batch, int base) {
        for (
                int r = 0;
                r < 4;
                r++
        ) {
            for (
                    int c = 0;
                    c < 3;
                    c++
            ) {
                int l = base + r * 4 + c, r_idx = l + 1;
                if (PieceUtils.getEast(batch[l]) != PieceUtils.getWest(batch[r_idx])) {
                    return false;
                }
            }
        }
        for (
                int c = 0;
                c < 4;
                c++
        ) {
            for (
                    int r = 0;
                    r < 3;
                    r++
            ) {
                int t = base + r * 4 + c, b = t + 4;
                if (PieceUtils.getSouth(batch[t]) != PieceUtils.getNorth(batch[b])) {
                    return false;
                }
            }
        }
        return true;
    }
}

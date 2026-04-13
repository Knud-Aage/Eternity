package dk.puzzle;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Optimized generator for 4x4 macro-tiles in the Eternity II puzzle.
 * This class uses a backtracking algorithm to find combinations of 16 pieces
 * that satisfy a given set of outer boundary constraints and internal adjacency rules.
 */
public class PermutationGenerator {
    private final PieceInventory inventory;
    private final boolean[] physicalUsed;
    private final Random rnd = new Random();
    private int deepestPos = 0;
    private int centerPhysicalIdx = -1;

    /**
     * Constructs a new PermutationGenerator.
     *
     * @param inventory The inventory of puzzle pieces and their orientations.
     * @param used      A shared boolean array tracking which physical pieces are currently in use.
     */
    public PermutationGenerator(PieceInventory inventory, boolean[] used) {
        this.inventory = inventory;
        this.physicalUsed = used;
        // Find the physical ID of the centerpiece so we can protect it
        int targetPiece = PieceUtils.pack(18, 12, 18, 3);
        for (int i = 0; i < 1024; i++) {
            if (inventory.allOrientations[i] == targetPiece) {
                this.centerPhysicalIdx = inventory.physicalMapping[i];
                break;
            }
        }
    }

    /**
     * Generates a list of internally valid 4x4 macro-tiles that satisfy the specified constraints.
     *
     * @param mIdx             The index of the macro-tile being generated.
     * @param macroConstraints A 16-element array representing the boundary constraints for each slot.
     * @param limit            The maximum number of valid macro-tiles to generate.
     * @return A list of integer arrays, each representing a valid 4x4 macro-tile.
     */
    public List<int[]> generate(int mIdx, int[] macroConstraints, int limit) {
        List<int[]> results = new ArrayList<>();
        deepestPos = 0;

        // Use -1 to represent empty spaces so we can safely build out of order
        int[] current = new int[16];
        Arrays.fill(current, -1);

        // Define solving order: Macro 5 builds backwards from the centerpiece!
        int[] order = new int[16];
        if (mIdx == 5) {
            for (int i = 0; i < 16; i++) order[i] = 15 - i; // 15, 14, 13... 0
        } else {
            for (int i = 0; i < 16; i++) order[i] = i;      // 0, 1, 2... 15
        }

        backtrack(0, mIdx, current, macroConstraints, results, limit, order);

        return results;
    }

    private void backtrack(int step, int mIdx, int[] current, int[] constraints, List<int[]> results, int limit,
                           int[] order) {
        if (results.size() >= limit) {
            return;
        }
        if (step > deepestPos) {
            deepestPos = step;
        }

        if (step == 16) {
            results.add(current.clone());
            return;
        }

        // Fetch the actual position we are working on based on our custom order
        int pos = order[step];
        int r = pos / 4, c = pos % 4;

        int packed = constraints[pos];
        int n_req = PieceUtils.getNorth(packed);
        int e_req = PieceUtils.getEast(packed);
        int s_req = PieceUtils.getSouth(packed);
        int w_req = PieceUtils.getWest(packed);

        // --- DYNAMIC ORDER-INDEPENDENT CONSTRAINTS ---
        // Look at neighboring spaces. If a piece is already there (!= -1), we MUST connect to it!
        if (r > 0 && current[pos - 4] != -1) {
            n_req = PieceUtils.getSouth(current[pos - 4]);
        }
        if (c < 3 && current[pos + 1] != -1) {
            e_req = PieceUtils.getWest(current[pos + 1]);
        }
        if (r < 3 && current[pos + 4] != -1) {
            s_req = PieceUtils.getNorth(current[pos + 4]);
        }
        if (c > 0 && current[pos - 1] != -1) {
            w_req = PieceUtils.getEast(current[pos - 1]);
        }

        int mRow = mIdx / 4, mCol = mIdx % 4;
        int gR = mRow * 4 + r, gC = mCol * 4 + c;
        int b_req = 0;
        if (gR == 0 || gR == 15) {
            b_req++;
        }
        if (gC == 0 || gC == 15) {
            b_req++;
        }

        List<Integer> pool;
        if (b_req == 2) {
            pool = inventory.corners;
        } else if (b_req == 1) {
            pool = inventory.edges;
        } else {
            pool = inventory.interior;
        }

        if (pool.isEmpty()) {
            return;
        }

        int size = pool.size();
        int offset = rnd.nextInt(size);

        for (int i = 0; i < size; i++) {
            int orientationIdx = pool.get((i + offset) % size);
            int p = inventory.allOrientations[orientationIdx];

            int physicalIdx = inventory.physicalMapping[orientationIdx];

            // Protect the centerpiece from being accidentally used in the wrong spot!
            if (mIdx == 5 && pos != 15 && physicalIdx == centerPhysicalIdx) {
                continue;
            }

            if (physicalUsed[physicalIdx]) {
                continue;
            }

            if (matches(p, n_req, e_req, s_req, w_req)) {
                current[pos] = p;
                physicalUsed[physicalIdx] = true;

                // Move to the next step in the order array
                backtrack(step + 1, mIdx, current, constraints, results, limit, order);

                physicalUsed[physicalIdx] = false;
                current[pos] = -1; // Reset to empty for backtracking

                if (results.size() >= limit) {
                    return;
                }
            }
        }
    }

    private boolean matches(int p, int n, int e, int s, int w) {
        if (n != PieceUtils.WILDCARD && PieceUtils.getNorth(p) != n) {
            return false;
        }
        if (e != PieceUtils.WILDCARD && PieceUtils.getEast(p) != e) {
            return false;
        }
        if (s != PieceUtils.WILDCARD && PieceUtils.getSouth(p) != s) {
            return false;
        }
        return w == PieceUtils.WILDCARD || PieceUtils.getWest(p) == w;
    }
}

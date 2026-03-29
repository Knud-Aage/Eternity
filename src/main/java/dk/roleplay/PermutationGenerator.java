package dk.roleplay;

import java.util.*;

/**
 * Optimized Generator for 4x4 Macro-Tiles.
 * It uses internal backtracking to ensure that every 16-piece candidate
 * produced is INTERNALLY VALID (all internal edges match).
 */
public class PermutationGenerator {
    private final PieceInventory inventory;
    private final boolean[] physicalUsed;

    public PermutationGenerator(PieceInventory inventory, boolean[] used) {
        this.inventory = inventory;
        this.physicalUsed = used;
    }

    public List<int[]> generate(int mIdx, int[] macroConstraints, int limit) {
        List<int[]> results = new ArrayList<>();
        backtrack(0, mIdx, new int[16], macroConstraints, results, limit);
        return results;
    }

    private void backtrack(int pos, int mIdx, int[] current, int[] constraints, List<int[]> results, int limit) {
        if (results.size() >= limit) return;
        
        if (pos == 16) {
            results.add(current.clone());
            return;
        }

        int r = pos / 4;
        int c = pos % 4;
        
        // Outer constraints from the board
        int outer = constraints[pos];
        int n_req = PieceUtils.getNorth(outer);
        int e_req = PieceUtils.getEast(outer);
        int s_req = PieceUtils.getSouth(outer);
        int w_req = PieceUtils.getWest(outer);

        // INTERNAL CONSTRAINTS: Match with already placed pieces in THIS macro-tile
        // If not on the top row of the macro-tile, must match piece above
        if (r > 0) {
            n_req = PieceUtils.getSouth(current[pos - 4]);
        }
        // If not on the left col of the macro-tile, must match piece to the left
        if (c > 0) {
            w_req = PieceUtils.getEast(current[pos - 1]);
        }

        // Determine which pool to draw from (Corner/Edge/Interior piece)
        List<Integer> pool = inventory.getPoolFor(mIdx, pos);
        
        // Use compatibility lookup if we have a North or West constraint
        List<Integer> candidates;
        if (n_req != PieceUtils.WILDCARD) {
            candidates = inventory.compatibility[0][n_req];
        } else if (w_req != PieceUtils.WILDCARD) {
            candidates = inventory.compatibility[3][w_req];
        } else {
            candidates = pool;
        }

        for (int orientationIdx : candidates) {
            // Must belong to the correct pool (Corner/Edge/Interior)
            // Note: compatibility lists contain all pieces, so we must filter by pool
            if (!pool.contains(orientationIdx)) continue;

            int physicalIdx = inventory.physicalMapping[orientationIdx];
            if (physicalUsed[physicalIdx]) continue;

            int p = inventory.allOrientations[orientationIdx];
            
            // Final check against all 4 potential constraints
            if (matches(p, n_req, e_req, s_req, w_req)) {
                current[pos] = p;
                physicalUsed[physicalIdx] = true;
                backtrack(pos + 1, mIdx, current, constraints, results, limit);
                physicalUsed[physicalIdx] = false;
                if (results.size() >= limit) return;
            }
        }
    }

    private boolean matches(int p, int n, int e, int s, int w) {
        if (n != PieceUtils.WILDCARD && PieceUtils.getNorth(p) != n) return false;
        if (e != PieceUtils.WILDCARD && PieceUtils.getEast(p) != e) return false;
        if (s != PieceUtils.WILDCARD && PieceUtils.getSouth(p) != s) return false;
        if (w != PieceUtils.WILDCARD && PieceUtils.getWest(p) != w) return false;
        return true;
    }
}

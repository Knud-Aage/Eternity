package dk.roleplay;

import java.util.*;

/**
 * Optimized Generator for 4x4 Macro-Tiles.
 */
public class PermutationGenerator {
    private final PieceInventory inventory;
    private final boolean[] physicalUsed;
    private final Random rnd = new Random();
    private int deepestPos = 0;

    public PermutationGenerator(PieceInventory inventory, boolean[] used) {
        this.inventory = inventory;
        this.physicalUsed = used;
    }

    public List<int[]> generate(int mIdx, int[] macroConstraints, int limit) {
        List<int[]> results = new ArrayList<>();
        deepestPos = 0;

        if (mIdx == 0) {
            System.out.println("--- Macro 0 Generation Start ---");
            int cornerMatches = 0;
            for (int idx : inventory.corners) {
                int p = inventory.allOrientations[idx];
                if (PieceUtils.getNorth(p) == 0 && PieceUtils.getWest(p) == 0) cornerMatches++;
            }
            System.out.println("Oriented corners matching Top-Left (N=0, W=0): " + cornerMatches);
        }
        
        backtrack(0, mIdx, new int[16], macroConstraints, results, limit);
        
        if (mIdx == 0 && results.isEmpty()) {
            System.out.println("--- Macro 0: FAILED (No valid 4x4 configuration found) ---");
            System.out.println("DEBUG: The CPU got stuck at position " + deepestPos + " inside the 4x4 block.");
        } else if (mIdx == 0) {
            System.out.println("Found " + results.size() + " candidates for Macro 0");
        }
        return results;
    }

    private void backtrack(int pos, int mIdx, int[] current, int[] constraints, List<int[]> results, int limit) {
        if (results.size() >= limit) return;
        if (pos > deepestPos) deepestPos = pos;

        if (pos == 16) {
            results.add(current.clone());
            return;
        }

        int r = pos / 4, c = pos % 4;
        int packed = constraints[pos];
        int n_req = PieceUtils.getNorth(packed);
        int e_req = PieceUtils.getEast(packed);
        int s_req = PieceUtils.getSouth(packed);
        int w_req = PieceUtils.getWest(packed);

        if (r > 0) n_req = PieceUtils.getSouth(current[pos - 4]);
        if (c > 0) w_req = PieceUtils.getEast(current[pos - 1]);

        int mRow = mIdx / 4, mCol = mIdx % 4;
        int gR = mRow * 4 + r, gC = mCol * 4 + c;
        int b_req = 0;
        if (gR == 0 || gR == 15) b_req++;
        if (gC == 0 || gC == 15) b_req++;

        List<Integer> pool;
        if (b_req == 2) pool = inventory.corners;
        else if (b_req == 1) pool = inventory.edges;
        else pool = inventory.interior;

        List<Integer> candidates = pool;
        if (n_req != PieceUtils.WILDCARD && n_req < inventory.compatibility[0].length) {
            List<Integer> comp = inventory.compatibility[0][n_req];
            if (comp.size() < candidates.size()) candidates = comp;
        } else if (w_req != PieceUtils.WILDCARD && w_req < inventory.compatibility[3].length) {
            List<Integer> comp = inventory.compatibility[3][w_req];
            if (comp.size() < candidates.size()) candidates = comp;
        }

        if (candidates.isEmpty()) {
            if (mIdx == 0 && pos == 1) {
                System.out.println("  DEBUG: No candidates found in compatibility list for Pos 1 (N=" + n_req + ", W=" + w_req + ")");
            }
            return;
        }

        List<Integer> shuffled = new ArrayList<>(candidates);
        Collections.shuffle(shuffled, rnd);

        int matchCount = 0;
        for (int idx : shuffled) {
            int p = inventory.allOrientations[idx];
            if (PieceUtils.getBorderCount(p) != b_req) continue;

            int physicalIdx = inventory.physicalMapping[idx];
            if (physicalUsed[physicalIdx]) continue;
            
            if (matches(p, n_req, e_req, s_req, w_req)) {
                matchCount++;
                current[pos] = p;
                physicalUsed[physicalIdx] = true;
                backtrack(pos + 1, mIdx, current, constraints, results, limit);
                physicalUsed[physicalIdx] = false;
                if (results.size() >= limit) return;
            }
        }

        if (mIdx == 0 && pos == 1 && matchCount == 0 && results.isEmpty()) {
            // Log once per corner attempted for pos 0
            // System.out.println("  DEBUG: Pos 1 failed to match North=" + n_req + " and West=" + w_req);
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

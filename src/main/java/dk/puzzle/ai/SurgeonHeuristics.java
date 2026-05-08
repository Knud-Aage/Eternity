package dk.puzzle.ai;

import dk.puzzle.util.PieceUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Implements "surgeon" heuristics for the Eternity II puzzle solver, responsible for 
 * strategically removing pieces from a board to create "holes" for the solver to re-fill.
 * 
 * <p>This class is typically used in metaheuristic search algorithms like Large 
 * Neighborhood Search (LNS) or Tabu Search to escape local optima. It identifies 
 * "conflict" zones where pieces are mismatched with their neighbors and prioritizes 
 * clearing those areas while respecting pre-placed pieces (like the center) and 
 * tabu restrictions.</p>
 */
public class SurgeonHeuristics {
    private final boolean lockCenter;
    private double targetedHolesPercentage;

    /**
     * Constructs a new SurgeonHeuristics instance.
     * 
     * @param lockCenter If true, the center piece (index 135) will be protected from removal.
     * @param targetedHolesPercentage The fraction (0.0 to 1.0) of holes that should be 
     *                                 targeted specifically at conflict zones. The remaining 
     *                                 holes are chosen randomly from all placed pieces.
     */
    public SurgeonHeuristics(boolean lockCenter, double targetedHolesPercentage) {
        this.lockCenter = lockCenter;
        this.targetedHolesPercentage = targetedHolesPercentage;
    }

    /**
     * Updates the percentage of holes to be targeted at conflict zones.
     * 
     * @param percentage A value between 0.0 (all random) and 1.0 (all conflict-based) 
     *                   representing the bias towards clearing mismatched pieces.
     */
    public void setTargetedHolesPercentage(double percentage) {
        this.targetedHolesPercentage = percentage;
    }

    /**
     * Creates multiple variations of a board by removing pieces ("punching holes").
     * 
     * <p>The method identifies pieces that can be removed (excluding empty slots, 
     * locked center pieces, and pieces currently under tabu tenure). It then scores 
     * pieces based on how many of their edges conflict with neighbors. A portion 
     * of the holes is targeted at these high-conflict pieces, while the rest are 
     * selected randomly. Finally, it explicitly clears the "frontier" slot 
     * associated with the current search progress.</p>
     * 
     * @param bestBoard The source board configuration to modify.
     * @param numClones The number of modified board variations to generate.
     * @param numHoles The target number of pieces to remove from each variation.
     * @param tabuTenure An array storing iteration timestamps to prevent removing 
     *                   recently modified pieces.
     * @param currentIteration The current iteration count of the solver.
     * @param currentHighScore The current search depth or progress, used to identify 
     *                         the next slot in the build sequence.
     * @param buildOrder The sequence of indices representing the solver's placement order.
     * @return A list of cloned board arrays with pieces removed.
     */
    public List<int[]> punchHoles(int[] bestBoard, int numClones, int numHoles, int[] tabuTenure, int currentIteration, int currentHighScore, int[] buildOrder) {
        List<int[]> swissCheeseBoards = new ArrayList<>(numClones);
        Random rnd = new Random();
        int[] placedIndices = new int[256];
        int placedCount = 0;

        for (int i = 0; i < 256; i++) {
            if (bestBoard[i] != -1 && bestBoard[i] != -2) {
                if (lockCenter && i == 135) continue;
                if (tabuTenure[i] > currentIteration) continue;
                placedIndices[placedCount++] = i;
            }
        }

        int actualHoles = Math.min(numHoles, placedCount);
        int[] conflicts = scoreConflicts(bestBoard);

        Integer[] sortedByConflict = new Integer[placedCount];
        for (int i = 0; i < placedCount; i++) sortedByConflict[i] = placedIndices[i];
        Arrays.sort(sortedByConflict, (a, b) -> Integer.compare(conflicts[b], conflicts[a]));

        int hotZoneSize = Math.max(actualHoles, placedCount / 4);
        int[] hotZone = new int[hotZoneSize];
        for (int i = 0; i < hotZoneSize; i++) hotZone[i] = sortedByConflict[i];

        int targetedHoles = (int) Math.round(actualHoles * targetedHolesPercentage);
        int randomHoles = actualHoles - targetedHoles;

        for (int clone = 0; clone < numClones; clone++) {
            int[] clonedBoard = new int[256];
            System.arraycopy(bestBoard, 0, clonedBoard, 0, 256);
            boolean[] punched = new boolean[256];

            int hotPicked = 0;
            int[] hotShuffled = hotZone.clone();
            hotStuffled(rnd, targetedHoles, clonedBoard, punched, hotPicked, hotShuffled);

            int randPicked = 0;
            int[] allShuffled = Arrays.copyOf(placedIndices, placedCount);
            hotStuffled(rnd, randomHoles, clonedBoard, punched, randPicked, allShuffled);

            if (currentHighScore < 256) clonedBoard[buildOrder[currentHighScore]] = -2;
            swissCheeseBoards.add(clonedBoard);
        }
        return swissCheeseBoards;
    }

    private void hotStuffled(Random rnd, int targetedHoles, int[] clonedBoard, boolean[] punched, int hotPicked, int[] hotShuffled) {
        for (int i = 0; i < hotShuffled.length && hotPicked < targetedHoles; i++) {
            int swapIdx = i + rnd.nextInt(hotShuffled.length - i);
            int tmp = hotShuffled[i];
            hotShuffled[i] = hotShuffled[swapIdx];
            hotShuffled[swapIdx] = tmp;
            int holeIdx = hotShuffled[i];
            if (!punched[holeIdx]) {
                clonedBoard[holeIdx] = -2;
                punched[holeIdx] = true;
                hotPicked++;
            }
        }
    }

    private int[] scoreConflicts(int[] sourceBoard) {
        int[] conflicts = new int[256];
        for (int idx = 0; idx < 256; idx++) {
            int p = sourceBoard[idx];
            if (p == -1 || p == -2 || (lockCenter && idx == 135)) continue;

            int row = idx / 16;
            int col = idx % 16;
            int score = 0;

            if (row > 0 && sourceBoard[idx - 16] > 0 && PieceUtils.getSouth(sourceBoard[idx - 16]) != PieceUtils.getNorth(p)) score++;
            if (row < 15 && sourceBoard[idx + 16] > 0 && PieceUtils.getNorth(sourceBoard[idx + 16]) != PieceUtils.getSouth(p)) score++;
            if (col > 0 && sourceBoard[idx - 1] > 0 && PieceUtils.getEast(sourceBoard[idx - 1]) != PieceUtils.getWest(p)) score++;
            if (col < 15 && sourceBoard[idx + 1] > 0 && PieceUtils.getWest(sourceBoard[idx + 1]) != PieceUtils.getEast(p)) score++;

            conflicts[idx] = score;
        }
        return conflicts;
    }
}
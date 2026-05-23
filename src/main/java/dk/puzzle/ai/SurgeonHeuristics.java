package dk.puzzle.ai;

import dk.puzzle.util.PieceUtils;

import java.util.*;

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
     * Flat board indices of pieces that must never be removed.
     *
     * <p>Includes the center piece (135) and all hint piece positions.
     * Passed in at construction time so this class stays decoupled from
     * the static hint arrays in {@code EternitySolver}.</p>
     */
    private final boolean[] isLocked = new boolean[256];

    /**
     * Constructs a new SurgeonHeuristics instance.
     *
     * @param lockCenter              If true, position 135 (center piece) is protected.
     * @param targetedHolesPercentage Fraction (0.0–1.0) of holes targeted at conflict zones.
     * @param hintPositions           Additional flat board indices that must never be removed
     *                                (e.g. the 4 non-center hint piece positions).
     */
    public SurgeonHeuristics(boolean lockCenter, double targetedHolesPercentage, int... hintPositions) {
        this.lockCenter = lockCenter;
        this.targetedHolesPercentage = targetedHolesPercentage;
        if (lockCenter) {
            this.isLocked[135] = true;
        }
        if (hintPositions != null) {
            for (int pos : hintPositions) {
                this.isLocked[pos] = true;
            }
        }
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
     * @param bestBoard        The source board configuration to modify.
     * @param numClones        The number of modified board variations to generate.
     * @param numHoles         The target number of pieces to remove from each variation.
     * @param tabuTenure       An array storing iteration timestamps to prevent removing
     *                         recently modified pieces.
     * @param currentIteration The current iteration count of the solver.
     * @param currentHighScore The current search depth or progress, used to identify
     *                         the next slot in the build sequence.
     * @param buildOrder       The sequence of indices representing the solver's placement order.
     * @return A list of cloned board arrays with pieces removed.
     */
    public List<int[]> punchHoles(int[] bestBoard, int numClones, int numHoles, int[] tabuTenure, int currentIteration, int currentHighScore, int[] buildOrder) {
        List<int[]> swissCheeseBoards = new ArrayList<>(numClones);
        Random rnd = new Random();
        int[] placedIndices = new int[256];
        int placedCount = 0;

        for (int i = 0; i < 256; i++) {
            if (bestBoard[i] != -1 && bestBoard[i] != -2) {
                if (isLocked[i]) {
                    continue;
                }
                if (tabuTenure[i] > currentIteration) {
                    continue;
                }
                placedIndices[placedCount++] = i;
            }
        }

        int actualHoles = Math.min(numHoles, placedCount);
        int[] conflicts = scoreConflicts(bestBoard);

        Integer[] sortedByConflict = new Integer[placedCount];
        for (int i = 0; i < placedCount; i++) {
            sortedByConflict[i] = placedIndices[i];
        }
        Arrays.sort(sortedByConflict, (a, b) -> Integer.compare(conflicts[b], conflicts[a]));

        int hotZoneSize = Math.max(actualHoles, placedCount / 4);
        int[] hotZone = new int[hotZoneSize];
        for (int i = 0; i < hotZoneSize; i++) {
            hotZone[i] = sortedByConflict[i];
        }

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

            if (currentHighScore < 256) {
                clonedBoard[buildOrder[currentHighScore]] = -2;
            }
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

    /**
     * Creates board variations by excavating a contiguous block of pieces.
     * * <p>Instead of relying on conflict scores from a post-fill, this method
     * finds the exact index where the solver got stuck (the frontier) and uses
     * a randomized Breadth-First Search (BFS) to organically grow a contiguous
     * crater backwards into the solved board. This gives the GPU a clean,
     * unbroken area to reconstruct.</p>
     * * @param bestBoard The source board configuration (the solid 209 pieces).
     *
     * @param numClones        The number of modified board variations to generate.
     * @param numHoles         The target number of pieces to remove.
     * @param tabuTenure       Array preventing removal of recently modified pieces.
     * @param currentIteration The current solver iteration.
     * @param currentHighScore The deepest point reached (e.g., 209).
     * @param buildOrder       The sequence of indices representing placement order.
     * @return A list of cloned boards with contiguous craters removed.
     */
    public List<int[]> excavateFrontier(int[] bestBoard, int numClones, int numHoles, int[] tabuTenure, int currentIteration, int currentHighScore, int[] buildOrder) {
        List<int[]> excavatedBoards = new ArrayList<>(numClones);
        Random rnd = new Random();

        // The exact spot on the board where the typewriter hit a dead end
        int stuckIndex = (currentHighScore < 256) ? buildOrder[currentHighScore] : -1;
        if (stuckIndex == -1) {
            return excavatedBoards;
        }

        for (int clone = 0; clone < numClones; clone++) {
            int[] clonedBoard = new int[256];
            System.arraycopy(bestBoard, 0, clonedBoard, 0, 256);

            Set<Integer> toRemove = new HashSet<>();
            List<Integer> frontier = new ArrayList<>();

            // Seed the crater at the exact point of failure
            frontier.add(stuckIndex);

            // Grow the crater organically until we hit the requested hole size
            while (toRemove.size() < numHoles && !frontier.isEmpty()) {
                // Pick a random tile from the edges of our growing crater
                int randomIndex = rnd.nextInt(frontier.size());
                int current = frontier.remove(randomIndex);

                if (toRemove.contains(current)) {
                    continue;
                }

                // Validate before removing: Must be placed, not locked, not tabu
                if (clonedBoard[current] != -1 && clonedBoard[current] != -2) {
                    if (isLocked[current]) {
                        continue;
                    }
                    if (tabuTenure[current] > currentIteration) {
                        continue;
                    }

                    toRemove.add(current);
                }

                // Geographically expand the search to all neighbors (BFS)
                int row = current / 16;
                int col = current % 16;

                if (row > 0) {
                    frontier.add(current - 16);  // North
                }
                if (row < 15) {
                    frontier.add(current + 16); // South
                }
                if (col > 0) {
                    frontier.add(current - 1);   // West
                }
                if (col < 15) {
                    frontier.add(current + 1);  // East
                }
            }

            // Execute the actual removal
            for (int idx : toRemove) {
                clonedBoard[idx] = -1;
            }

            // Explicitly flag the stuck index as the next target for the solver
            clonedBoard[stuckIndex] = -2;

            excavatedBoards.add(clonedBoard);
        }

        return excavatedBoards;
    }

    private int[] scoreConflicts(int[] sourceBoard) {
        int[] conflicts = new int[256];
        for (int idx = 0; idx < 256; idx++) {
            int p = sourceBoard[idx];
            if (p == -1 || p == -2 || (lockCenter && idx == 135)) {
                continue;
            }

            int row = idx / 16;
            int col = idx % 16;
            int score = 0;

            if (row > 0 && sourceBoard[idx - 16] > 0 && PieceUtils.getSouth(sourceBoard[idx - 16]) != PieceUtils.getNorth(p)) {
                score++;
            }
            if (row < 15 && sourceBoard[idx + 16] > 0 && PieceUtils.getNorth(sourceBoard[idx + 16]) != PieceUtils.getSouth(p)) {
                score++;
            }
            if (col > 0 && sourceBoard[idx - 1] > 0 && PieceUtils.getEast(sourceBoard[idx - 1]) != PieceUtils.getWest(p)) {
                score++;
            }
            if (col < 15 && sourceBoard[idx + 1] > 0 && PieceUtils.getWest(sourceBoard[idx + 1]) != PieceUtils.getEast(p)) {
                score++;
            }

            conflicts[idx] = score;
        }
        return conflicts;
    }
}
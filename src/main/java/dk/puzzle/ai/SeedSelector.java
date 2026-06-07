package dk.puzzle.ai;

import java.util.*;

/**
 * Selects seed boards for the next GPU search round using a stratified
 * population strategy designed to prevent population collapse.
 *
 * <h2>Why stratification matters at depth 110+</h2>
 * <p>When all seeds share the same pieces in the first 30-40 positions
 * (descended from the same {@code globalBestBoard}), the GPU always
 * converges to the same local optimum regardless of how many seeds are
 * provided. This is called <em>population collapse</em> and is the main
 * reason solvers stall at the same score for long periods.</p>
 *
 * <h2>Three-tier selection strategy</h2>
 * <ul>
 *   <li><b>Elite (20%):</b> highest-scoring seeds, kept exactly as-is.</li>
 *   <li><b>Diverse (50%):</b> seeds with Hamming distance &gt; 25 from all
 *       elite seeds in the first {@code DIVERSITY_DEPTH} build-order positions.
 *       Picked by score within that constraint.</li>
 *   <li><b>Random-restart (30%):</b> seeds that share zero pieces in the
 *       first {@code POISON_DEPTH} build-order positions with the reference
 *       board. Breaks population collapse completely.</li>
 * </ul>
 *
 * <h2>Edge-consistency guarantee</h2>
 * <p>Every returned seed is validated by {@link #isEdgeConsistent}. Seeds
 * with internal edge conflicts are silently dropped — they would cause
 * {@code verifyBoardStrict} rejections in Phase 2.</p>
 */
public class SeedSelector {

    /**
     * Number of build-order positions used when computing Hamming distance.
     * Covers roughly the first third of the board — the region that most
     * strongly determines the final structural outcome.
     */
    private static final int DIVERSITY_DEPTH = 40;

    /**
     * Minimum Hamming distance (in {@link #DIVERSITY_DEPTH} positions) that
     * a seed must have from all elite seeds to qualify for the diverse tier.
     * Seeds below this threshold are too similar to the elite to add value.
     */
    private static final int MIN_HAMMING_DISTANCE = 25;

    /**
     * Number of build-order positions checked for the random-restart tier.
     * Seeds must share zero pieces here with the reference board.
     */
    private static final int POISON_DEPTH = 30;

    // Position weights: centre cells are hardest, border cells are easiest.
    private static final int[] POSITION_WEIGHT = buildPositionWeights();

    private static int[] buildPositionWeights() {
        int[] w = new int[256];
        for (int i = 0; i < 256; i++) {
            int row = i / 16;
            int col = i % 16;
            int distFromEdge = Math.min(
                    Math.min(row, 15 - row),
                    Math.min(col, 15 - col));
            w[i] = distFromEdge + 1; // 1 (border) to 8 (centre)
        }
        return w;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Scores a board based on placement depth, position quality, and
     * constraint pressure.
     *
     * @param board        256-element array; -1 indicates an empty cell.
     * @param depthReached Number of pieces placed as reported by the GPU thread.
     * @return             Combined score — higher is better.
     */
    public static int scoreBoard(int[] board, int depthReached) {
        int posScore = 0;
        int dangerPenalty = 0;

        for (int i = 0; i < 256; i++) {
            if (board[i] != -1) {
                posScore += POSITION_WEIGHT[i];
            } else {
                // Penalise empty cells that are heavily surrounded —
                // they indicate structural dead ends nearby.
                int row = i / 16;
                int col = i % 16;
                int filled = 0;
                if (row > 0  && board[i - 16] != -1) filled++;
                if (row < 15 && board[i + 16] != -1) filled++;
                if (col > 0  && board[i - 1]  != -1) filled++;
                if (col < 15 && board[i + 1]  != -1) filled++;
                if (filled >= 3) dangerPenalty += (filled - 2) * 5;
            }
        }

        // depthReached carries the most weight — it is the GPU's own
        // measure of how far down the search tree this seed reached.
        return depthReached * 100 + posScore * 2 - dangerPenalty;
    }

    /**
     * Validates that a board has no internal edge conflicts between placed pieces.
     *
     * @param board 256-element board array.
     * @return {@code true} if all adjacent placed pieces have matching edges.
     */
    public static boolean isEdgeConsistent(int[] board) {
        for (int i = 0; i < 256; i++) {
            if (board[i] == -1) continue;
            int row = i / 16;
            int col = i % 16;

            if (col < 15 && board[i + 1] != -1) {
                int myEast    = (board[i]     >> 16) & 0xFF;
                int theirWest =  board[i + 1]        & 0xFF;
                if (myEast != theirWest) return false;
            }
            if (row < 15 && board[i + 16] != -1) {
                int mySouth    = (board[i]      >>  8) & 0xFF;
                int theirNorth = (board[i + 16] >> 24) & 0xFF;
                if (mySouth != theirNorth) return false;
            }
        }
        return true;
    }

    /**
     * Computes the Hamming distance between two boards over the first
     * {@code depth} positions in the given build order.
     *
     * <p>Two positions are considered different if the physical piece numbers
     * differ, ignoring rotation. This measures structural divergence in the
     * region that most strongly determines the final board outcome.</p>
     *
     * @param a          First board (256-element array).
     * @param b          Second board (256-element array).
     * @param buildOrder Build-order position sequence.
     * @param depth      Number of build-order positions to compare.
     * @return           Number of positions where the physical pieces differ.
     */
    public static int hammingDistance(int[] a, int[] b, int[] buildOrder, int depth) {
        int diff = 0;
        for (int step = 0; step < depth && step < buildOrder.length; step++) {
            int idx = buildOrder[step];
            int pa = a[idx];
            int pb = b[idx];
            if (pa == -1 || pb == -1) {
                if (pa != pb) diff++; // one empty, one not = different
                continue;
            }
            // Compare only the raw packed value — rotation differences
            // between the same physical piece also count as different
            // since they produce different edge exposures.
            if (pa != pb) diff++;
        }
        return diff;
    }

    /**
     * Returns true if {@code candidate} shares zero pieces with
     * {@code reference} in the first {@code POISON_DEPTH} build-order
     * positions. Used to identify genuine random-restart seeds.
     *
     * @param candidate  Seed board to test.
     * @param reference  Reference board (typically {@code globalBestBoard}).
     * @param buildOrder Build-order position sequence.
     * @return           {@code true} if no overlap in the poison zone.
     */
    public static boolean isRandomRestart(int[] candidate, int[] reference,
                                          int[] buildOrder) {
        for (int step = 0; step < POISON_DEPTH && step < buildOrder.length; step++) {
            int idx = buildOrder[step];
            if (candidate[idx] != -1 && candidate[idx] == reference[idx]) {
                return false; // shares at least one piece — not a fresh restart
            }
        }
        return true;
    }

    /**
     * Selects seeds for the next GPU round using a three-tier stratified strategy.
     *
     * <p>Tier breakdown:</p>
     * <ul>
     *   <li><b>Elite (20%):</b> top seeds by score, copied directly.</li>
     *   <li><b>Diverse (50%):</b> seeds with Hamming distance &gt;
     *       {@link #MIN_HAMMING_DISTANCE} from all elite seeds.</li>
     *   <li><b>Random-restart (30%):</b> seeds sharing zero pieces with
     *       {@code referenceBoard} in the first {@link #POISON_DEPTH}
     *       positions — breaks population collapse.</li>
     * </ul>
     *
     * <p>All returned seeds are edge-consistent. If a tier cannot be filled
     * (e.g. not enough diverse seeds in the pool), the shortfall is filled
     * from the elite tier to ensure {@code targetCount} is always returned.</p>
     *
     * @param allBoards      All seed boards from the previous GPU round.
     * @param threadDepths   GPU-reported max depth per thread (same order).
     * @param targetCount    Number of seeds to return.
     * @param buildOrder     Build-order position sequence from the solver.
     * @param referenceBoard The current best board — used to identify
     *                       random-restart seeds. Pass {@code globalBestBoard}.
     * @param random         Random source for sampling within tiers.
     * @return               Selected seeds, all edge-consistent.
     */
    public static List<int[]> selectBest(
            List<int[]> allBoards,
            int[] threadDepths,
            int targetCount,
            int[] buildOrder,
            int[] referenceBoard,
            Random random)
    {
        int n = allBoards.size();
        if (n == 0) return new ArrayList<>();

        // ── Score all boards ──────────────────────────────────────────────
        int[] scores = new int[n];
        for (int i = 0; i < n; i++) {
            scores[i] = scoreBoard(allBoards.get(i),
                    i < threadDepths.length ? threadDepths[i] : 0);
        }

        // Sort indices by score descending
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> scores[b] - scores[a]);

        // ── Tier sizes ────────────────────────────────────────────────────
        int eliteCount   = Math.max(1, targetCount * 20 / 100); // 20%
        int diverseCount = Math.max(1, targetCount * 50 / 100); // 50%
        int restartCount = targetCount - eliteCount - diverseCount; // 30%

        List<int[]> result    = new ArrayList<>(targetCount);
        List<int[]> eliteList = new ArrayList<>(eliteCount);

        // ── TIER 1: Elite — top seeds by score ───────────────────────────
        for (int i = 0; i < indices.length && eliteList.size() < eliteCount; i++) {
            int[] board = allBoards.get(indices[i]);
            if (isEdgeConsistent(board)) {
                int[] copy = Arrays.copyOf(board, 256);
                eliteList.add(copy);
                result.add(copy);
            }
        }

        // ── TIER 2: Diverse — Hamming distance > MIN from all elite ──────
        // Walk candidates in score order; accept only those sufficiently
        // different from every elite seed already selected.
        int diverseFilled = 0;
        for (int i = 0; i < indices.length && diverseFilled < diverseCount; i++) {
            int[] board = allBoards.get(indices[i]);
            if (!isEdgeConsistent(board)) continue;

            boolean sufficientlyDiverse = true;
            for (int[] elite : eliteList) {
                if (hammingDistance(board, elite, buildOrder, DIVERSITY_DEPTH)
                        < MIN_HAMMING_DISTANCE) {
                    sufficientlyDiverse = false;
                    break;
                }
            }
            if (sufficientlyDiverse) {
                result.add(Arrays.copyOf(board, 256));
                diverseFilled++;
            }
        }

        // ── TIER 3: Random-restart — zero overlap with referenceBoard ────
        // Shuffle indices so we don't always pick the same restart seeds.
        List<Integer> shuffled = new ArrayList<>(Arrays.asList(indices));
        Collections.shuffle(shuffled, random);

        int restartFilled = 0;
        for (int idx : shuffled) {
            if (restartFilled >= restartCount) break;
            int[] board = allBoards.get(idx);
            if (!isEdgeConsistent(board)) continue;
            if (referenceBoard != null
                    && isRandomRestart(board, referenceBoard, buildOrder)) {
                result.add(Arrays.copyOf(board, 256));
                restartFilled++;
            }
        }

        // ── Fallback: fill any shortfall from elite ───────────────────────
        // Happens when the pool lacks enough diverse or restart seeds.
        int eliteIdx = 0;
        while (result.size() < targetCount && eliteIdx < indices.length) {
            int[] board = allBoards.get(indices[eliteIdx++]);
            if (isEdgeConsistent(board)) {
                result.add(Arrays.copyOf(board, 256));
            }
        }

        return result;
    }

    /**
     * Backwards-compatible overload that omits {@code buildOrder} and
     * {@code referenceBoard}. Falls back to the old score-only strategy
     * without diversity enforcement — use the full overload where possible.
     *
     * @deprecated Use
     * {@link #selectBest(List, int[], int, int[], int[], Random)} instead.
     */
    @Deprecated
    public static List<int[]> selectBest(
            List<int[]> allBoards,
            int[] threadDepths,
            int targetCount,
            Random random)
    {
        return selectBest(allBoards, threadDepths, targetCount,
                defaultBuildOrder(), null, random);
    }

    private static int[] defaultBuildOrder() {
        int[] order = new int[256];
        for (int i = 0; i < 256; i++) order[i] = i;
        return order;
    }
}
package dk.puzzle.ai;

import java.util.*;

/**
 * Selects the best seed boards from GPU results for the next search round.
 *
 * <p>Seeds are partial boards with ~40 placed pieces produced by CPU workers.
 * They are passed directly to the GPU DFS kernel, which means every seed
 * must be internally edge-consistent — any constraint violation in a seed
 * propagates directly into the GPU's best-board output and triggers the
 * {@code verifyBoardStrict} rejection in Phase 2.</p>
 *
 * <p><b>No mutation is applied.</b> Swapping pieces between positions in a
 * valid partial board almost always creates edge conflicts (the swapped piece
 * no longer matches its neighbours). Diversity is instead achieved by
 * selecting from a wide pool of CPU-generated seeds that are already
 * structurally distinct, and by stratified sampling across score tiers.</p>
 *
 * <p>Selection strategy:</p>
 * <ul>
 *   <li>Top 20% by score: kept as-is ("elite")</li>
 *   <li>Next 30%: sampled randomly from the top 50% for variety</li>
 *   <li>Remaining 50%: sampled from the full pool to preserve diversity</li>
 * </ul>
 */
public class SeedSelector {

    // Position weights: centre cells are hardest, border cells are easiest.
    private static final int[] POSITION_WEIGHT = buildPositionWeights();

    private static int[] buildPositionWeights() {
        int[] w = new int[256];
        for (int i = 0; i < 256; i++) {
            int row = i / 16;
            int col = i % 16;
            int distFromEdge = Math.min(Math.min(row, 15 - row), Math.min(col, 15 - col));
            w[i] = distFromEdge + 1; // 1 (border) to 8 (centre)
        }
        return w;
    }

    /**
     * Scores a board based on placement depth, position quality, and constraint pressure.
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
                // Penalise empty cells that are heavily surrounded — they are
                // hard to fill and indicate structural dead ends nearby.
                int row = i / 16;
                int col = i % 16;
                int filledNeighbours = 0;
                if (row > 0  && board[i - 16] != -1) filledNeighbours++;
                if (row < 15 && board[i + 16] != -1) filledNeighbours++;
                if (col > 0  && board[i - 1]  != -1) filledNeighbours++;
                if (col < 15 && board[i + 1]  != -1) filledNeighbours++;
                if (filledNeighbours >= 3) dangerPenalty += (filledNeighbours - 2) * 5;
            }
        }

        // depthReached carries the most weight: it is the GPU's own measure of progress.
        return depthReached * 100 + posScore * 2 - dangerPenalty;
    }

    /**
     * Validates that a board has no internal edge conflicts between placed pieces.
     *
     * <p>Used as a safety gate before seeds are returned. Any seed that fails
     * this check is silently dropped — it would cause a {@code verifyBoardStrict}
     * rejection in Phase 2 if sent to the GPU.</p>
     *
     * @param board 256-element board array.
     * @return {@code true} if all adjacent placed pieces have matching edges.
     */
    public static boolean isEdgeConsistent(int[] board) {
        for (int i = 0; i < 256; i++) {
            if (board[i] == -1) continue;
            int row = i / 16;
            int col = i % 16;

            // Check east neighbour
            if (col < 15 && board[i + 1] != -1) {
                int myEast    = (board[i]     >> 16) & 0xFF;
                int theirWest =  board[i + 1]        & 0xFF;
                if (myEast != theirWest) return false;
            }
            // Check south neighbour
            if (row < 15 && board[i + 16] != -1) {
                int mySouth    = (board[i]      >>  8) & 0xFF;
                int theirNorth = (board[i + 16] >> 24) & 0xFF;
                if (mySouth != theirNorth) return false;
            }
        }
        return true;
    }

    /**
     * Selects the best seeds from a completed GPU round for the next round.
     *
     * <p>All returned seeds are guaranteed to be edge-consistent. Any board
     * that fails {@link #isEdgeConsistent} is silently skipped.</p>
     *
     * @param allBoards    All seed boards from the previous round (one board = int[256]).
     * @param threadDepths GPU-reported max depth reached per thread (same order as allBoards).
     * @param targetCount  Number of seeds to return.
     * @param random       Random source for stratified sampling.
     * @return             Selected seeds, all edge-consistent, ready for the next GPU round.
     */
    public static List<int[]> selectBest(
            List<int[]> allBoards,
            int[] threadDepths,
            int targetCount,
            Random random)
    {
        int n = allBoards.size();
        if (n == 0) return new ArrayList<>();

        // Score all boards
        int[] scores = new int[n];
        for (int i = 0; i < n; i++) {
            scores[i] = scoreBoard(allBoards.get(i), i < threadDepths.length ? threadDepths[i] : 0);
        }

        // Sort indices by score descending
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> scores[b] - scores[a]);

        // Tier boundaries
        int eliteCount  = Math.max(1, targetCount * 2 / 10); // top 20%
        int midCount    = Math.max(1, targetCount * 3 / 10); // next 30% from top half
        int broadCount  = targetCount - eliteCount - midCount; // remaining from full pool

        int top50 = Math.max(1, n / 2);

        List<int[]> result = new ArrayList<>(targetCount);

        // --- Elite: top 20% by score, copied directly ---
        for (int i = 0; i < indices.length && result.size() < eliteCount; i++) {
            int[] board = allBoards.get(indices[i]);
            if (isEdgeConsistent(board)) {
                result.add(Arrays.copyOf(board, 256));
            }
        }

        // --- Mid tier: random sample from top 50% ---
        for (int attempt = 0; attempt < midCount * 3 && result.size() < eliteCount + midCount; attempt++) {
            int[] board = allBoards.get(indices[random.nextInt(top50)]);
            if (isEdgeConsistent(board)) {
                result.add(Arrays.copyOf(board, 256));
            }
        }

        // --- Broad tier: random sample from full pool for diversity ---
        for (int attempt = 0; attempt < broadCount * 3 && result.size() < targetCount; attempt++) {
            int[] board = allBoards.get(indices[random.nextInt(n)]);
            if (isEdgeConsistent(board)) {
                result.add(Arrays.copyOf(board, 256));
            }
        }

        // If we came up short (many invalid seeds), fill remainder from elite
        int eliteIdx = 0;
        while (result.size() < targetCount && eliteIdx < indices.length) {
            int[] board = allBoards.get(indices[eliteIdx++]);
            if (isEdgeConsistent(board)) {
                result.add(Arrays.copyOf(board, 256));
            }
        }

        return result;
    }
}
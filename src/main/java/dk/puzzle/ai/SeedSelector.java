package dk.puzzle.ai;

import java.util.*;

/**
 * Selects seed boards for the next GPU search round using a stratified
 * population strategy dynamically scaled for the deep endgame.
 */
public class SeedSelector {

    private static final int[] POSITION_WEIGHT = buildPositionWeights();

    private static int[] buildPositionWeights() {
        int[] w = new int[256];
        for (int i = 0; i < 256; i++) {
            int row = i / 16;
            int col = i % 16;
            int distFromEdge = Math.min(
                    Math.min(row, 15 - row),
                    Math.min(col, 15 - col));
            w[i] = distFromEdge + 1;
        }
        return w;
    }

    public static int scoreBoard(int[] board, int depthReached) {
        int posScore = 0;
        int dangerPenalty = 0;
        int edgeConflicts = 0;

        for (int i = 0; i < 256; i++) {
            if (board[i] != -1) {
                posScore += POSITION_WEIGHT[i];

                // Count internal edge conflicts between placed neighbours.
                // Each conflict is counted once (only check east and south
                // so we don't double-count).
                int row = i / 16;
                int col = i % 16;
                if (col < 15 && board[i + 1] != -1) {
                    int myEast    = (board[i]     >> 16) & 0xFF;
                    int theirWest =  board[i + 1]        & 0xFF;
                    if (myEast != theirWest) edgeConflicts++;
                }
                if (row < 15 && board[i + 16] != -1) {
                    int mySouth    = (board[i]      >>  8) & 0xFF;
                    int theirNorth = (board[i + 16] >> 24) & 0xFF;
                    if (mySouth != theirNorth) edgeConflicts++;
                }
            } else {
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

        // Edge conflicts are heavily penalised — a seed with internal conflicts
        // is a structural dead end; the GPU will hit the same wall from it.
        // Weight: each conflict costs as much as losing ~2 depth points.
        int conflictPenalty = edgeConflicts * 200;

        return depthReached * 100 + posScore * 2 - dangerPenalty - conflictPenalty;
    }

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
     * Computes the absolute Hamming distance across the ENTIRE board.
     */
    public static int hammingDistance(int[] a, int[] b) {
        int diff = 0;
        for (int i = 0; i < 256; i++) {
            if (a[i] != -1 && b[i] != -1 && a[i] != b[i]) {
                diff++;
            }
        }
        return diff;
    }

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

        // ── 1. Auto-Detect the Active CPU Frontier ──
        // Since all seeds come from the same CPU batch, we find how many
        // pieces are permanently locked by comparing the first and last seed.
        int[] firstSeed = allBoards.get(0);
        int[] lastSeed = allBoards.get(n - 1);
        int lockedPieces = 0;
        for (int i = 0; i < 256; i++) {
            int idx = buildOrder[i];
            if (firstSeed[idx] != -1 && firstSeed[idx] == lastSeed[idx]) {
                lockedPieces++;
            } else {
                break;
            }
        }

        // Count how many pieces the CPU actually added on top of the lock
        int piecesPlacedByCpu = 0;
        for (int i = lockedPieces; i < 256; i++) {
            if (firstSeed[buildOrder[i]] != -1) piecesPlacedByCpu++;
        }

        // Dynamic thresholds: If CPU only placed 4 pieces, min diverse distance is 2.
        int dynamicMinHamming = Math.max(1, piecesPlacedByCpu / 2);

        // ── 2. Score and Sort ──
        int[] scores = new int[n];
        for (int i = 0; i < n; i++) {
            scores[i] = scoreBoard(allBoards.get(i),
                    i < threadDepths.length ? threadDepths[i] : 0);
        }

        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> scores[b] - scores[a]);

        int eliteCount   = Math.max(1, targetCount * 15 / 100);
        int diverseCount = Math.max(1, targetCount * 55 / 100);
        int restartCount = targetCount - eliteCount - diverseCount;

        List<int[]> result    = new ArrayList<>(targetCount);
        List<int[]> eliteList = new ArrayList<>(eliteCount);

        // ── TIER 1: Elite ──
        for (int i = 0; i < indices.length && eliteList.size() < eliteCount; i++) {
            int[] board = allBoards.get(indices[i]);
            if (isEdgeConsistent(board)) {
                int[] copy = Arrays.copyOf(board, 256);
                eliteList.add(copy);
                result.add(copy);
            }
        }

        // ── TIER 2: Diverse ──
        int diverseFilled = 0;
        for (int i = 0; i < indices.length && diverseFilled < diverseCount; i++) {
            int[] board = allBoards.get(indices[i]);
            if (!isEdgeConsistent(board)) continue;

            boolean sufficientlyDiverse = true;
            for (int[] elite : eliteList) {
                if (hammingDistance(board, elite) < dynamicMinHamming) {
                    sufficientlyDiverse = false;
                    break;
                }
            }
            if (sufficientlyDiverse) {
                result.add(Arrays.copyOf(board, 256));
                diverseFilled++;
            }
        }

        // ── TIER 3: Random-restart ──
        List<Integer> shuffled = new ArrayList<>(Arrays.asList(indices));
        Collections.shuffle(shuffled, random);

        int restartFilled = 0;
        for (int idx : shuffled) {
            if (restartFilled >= restartCount) break;
            int[] board = allBoards.get(idx);
            if (!isEdgeConsistent(board)) continue;

            if (referenceBoard != null) {
                boolean isRestart = true;
                // Check ONLY the active frontier zone. A true restart must
                // disagree with the local trap right where the CPU started branching.
                for (int step = lockedPieces; step < lockedPieces + piecesPlacedByCpu && step < 256; step++) {
                    int boardIdx = buildOrder[step];
                    if (board[boardIdx] != -1 && board[boardIdx] == referenceBoard[boardIdx]) {
                        isRestart = false; // It copied the trap!
                        break;
                    }
                }
                if (isRestart) {
                    result.add(Arrays.copyOf(board, 256));
                    restartFilled++;
                }
            }
        }

        // ── Fallback ──
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
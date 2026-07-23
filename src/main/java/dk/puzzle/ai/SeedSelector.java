package dk.puzzle.ai;

import java.util.*;

/**
 * Selects seed boards for the next GPU search round using a stratified
 * population strategy dynamically scaled for the deep endgame.
 */
public class SeedSelector {

    // Population split — adjustable via UI sliders at runtime.
    // Restarts = whatever is left after elite + diverse.
    private static volatile int elitePct   = 15;  // default: lean toward diversity
    private static volatile int diversePct = 55;

    public static void setElitePct(int pct) {
        elitePct = Math.max(5, Math.min(60, pct));
    }
    public static void setDiversePct(int pct) {
        diversePct = Math.max(10, Math.min(70, pct));
    }

    /**
     * Supplied by the caller so SeedSelector needs no PieceInventory /
     * ConflictReducer dependency. Given a seed board, returns the edge-conflict
     * count of the best full board that seed fills to.
     *
     * This is the objective that actually matters: a shorter prefix that fills
     * cleanly beats a longer one that fills badly (220 -> 21 conflicts vs
     * 221 -> 23). Depth alone cannot express that.
     */
    @FunctionalInterface
    public interface ConflictEvaluator {
        int fullBoardConflicts(int[] seedBoard);
    }

    // Diagnostics from the most recent evaluated selection (for logging).
    private static volatile int lastEvaluatedCount = 0;
    private static volatile int lastBestConflicts  = -1;
    private static volatile int lastWorstConflicts = -1;

    public static int getLastEvaluatedCount() { return lastEvaluatedCount; }
    public static int getLastBestConflicts()  { return lastBestConflicts;  }
    public static int getLastWorstConflicts() { return lastWorstConflicts; }

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

        for (int i = 0; i < 256; i++) {
            if (board[i] != -1) {
                posScore += POSITION_WEIGHT[i];
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
        return depthReached * 100 + posScore * 2 - dangerPenalty;
    }

    public static boolean isEdgeConsistent(int[] board) {
        for (int i = 0; i < 256; i++) {
            if (board[i] == -1 || board[i] == -2) continue;
            int row = i / 16;
            int col = i % 16;

            if (col < 15 && board[i + 1] != -1 && board[i + 1] != -2) {
                int myEast    = (board[i]     >> 16) & 0xFF;
                int theirWest =  board[i + 1]        & 0xFF;
                if (myEast != theirWest) return false;
            }
            if (row < 15 && board[i + 16] != -1 && board[i + 16] != -2) {
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

    /** Existing signature — unchanged behaviour (depth-ranked elite). */
    public static List<int[]> selectBest(
            List<int[]> allBoards,
            int[] threadDepths,
            int targetCount,
            int[] buildOrder,
            int[] referenceBoard,
            Random random)
    {
        return selectBest(allBoards, threadDepths, targetCount,
                buildOrder, referenceBoard, random, null, 0);
    }

    /**
     * Two-stage selection.
     *
     * STAGE 1 (cheap, every board): rank by scoreBoard() as before.
     * STAGE 2 (costly, top {@code evalPoolSize} only): fill each leader and
     *   count resulting conflicts, then choose the ELITE tier by fewest
     *   conflicts instead of greatest depth.
     *
     * Diverse and Restart tiers keep the stage-1 ordering — their job is
     * spread, not quality, and re-ranking them would work against that.
     *
     * evaluator == null reproduces the original behaviour exactly.
     */
    public static List<int[]> selectBest(
            List<int[]> allBoards,
            int[] threadDepths,
            int targetCount,
            int[] buildOrder,
            int[] referenceBoard,
            Random random,
            ConflictEvaluator evaluator,
            int evalPoolSize)
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

        int eliteCount   = Math.max(1, targetCount * elitePct   / 100);
        int diverseCount = Math.max(1, targetCount * diversePct / 100);
        int restartCount = Math.max(1, targetCount - eliteCount - diverseCount);

        List<int[]> result    = new ArrayList<>(targetCount);
        List<int[]> eliteList = new ArrayList<>(eliteCount);

        // ── TIER 1: Elite ──
        // Each tier's own count floors to at least 1 (see above), so their sum
        // can exceed targetCount for small targetCount values; result.size() <
        // targetCount is checked in every tier's loop condition as the hard
        // cap so selectBest never returns more than requested.
        // STAGE 2: re-rank the stage-1 leaders by how cleanly they FILL rather
        // than how deep they GO, and draw elites from that ordering instead.
        //
        // The candidate pool is NOT a flat top-evalPoolSize by scores[] --
        // scoreBoard() is depth-dominated (depthReached*100 swamps the
        // posScore/dangerPenalty terms), so a flat top-N is effectively just
        // "the N deepest seeds" and would never let a shallower-but-cleaner
        // seed be evaluated at all -- exactly the case this feature exists to
        // catch (220->21 vs 221->23 conflicts). Instead: guarantee at least
        // one representative from every distinct depth present in the batch
        // first, then spend the remaining budget on the overall stage-1
        // ordering (still depth-leaning, but no longer an absolute gate).
        List<Integer> eliteOrder = new ArrayList<>();
        if (evaluator != null && evalPoolSize > 0) {
            Map<Integer, List<Integer>> byDepth = new LinkedHashMap<>();
            for (int idx : indices) {
                if (!isEdgeConsistent(allBoards.get(idx))) continue;
                int depth = idx < threadDepths.length ? threadDepths[idx] : 0;
                byDepth.computeIfAbsent(depth, d -> new ArrayList<>()).add(idx);
            }

            Set<Integer> pooled = new LinkedHashSet<>();
            for (List<Integer> bucket : byDepth.values()) {
                if (pooled.size() >= evalPoolSize) break;
                pooled.add(bucket.get(0)); // each bucket's own best-scoring member
            }
            for (int idx : indices) {
                if (pooled.size() >= evalPoolSize) break;
                if (isEdgeConsistent(allBoards.get(idx))) pooled.add(idx);
            }

            List<Integer> pool = new ArrayList<>(pooled);
            if (!pool.isEmpty()) {
                final int[] conflicts = new int[n];
                Arrays.fill(conflicts, Integer.MAX_VALUE);

                // Each evaluation is an independent MCV fill + polish on its own
                // board copy (no shared mutable state), so run the pool in
                // parallel -- on the caller's thread, evalPoolSize evaluations
                // otherwise cost evalPoolSize times a single evaluation's
                // wall-clock time, which was blocking the next GPU batch launch
                // for several seconds per round.
                pool.parallelStream().forEach(idx ->
                        conflicts[idx] = evaluator.fullBoardConflicts(allBoards.get(idx)));

                int best = Integer.MAX_VALUE, worst = -1;
                for (int idx : pool) {
                    int cf = conflicts[idx];
                    if (cf < best)  best  = cf;
                    if (cf > worst) worst = cf;
                }
                lastEvaluatedCount = pool.size();
                lastBestConflicts  = best;
                lastWorstConflicts = worst;

                // Fewest conflicts first; ties broken by stage-1 score, so among
                // equally clean fills the deeper board still wins.
                pool.sort((a, b) -> conflicts[a] != conflicts[b]
                        ? conflicts[a] - conflicts[b]
                        : scores[b] - scores[a]);
                eliteOrder = pool;
            }
        }

        for (int i = 0; i < eliteOrder.size() && eliteList.size() < eliteCount && result.size() < targetCount; i++) {
            int[] copy = Arrays.copyOf(allBoards.get(eliteOrder.get(i)), 256);
            boolean dup = false;
            for (int[] e : eliteList) if (Arrays.equals(e, copy)) { dup = true; break; }
            if (dup) continue;
            eliteList.add(copy);
            result.add(copy);
        }
        // Top up from the stage-1 ordering when the evaluated pool was too
        // small, or when no evaluator was supplied (the original path).
        for (int i = 0; i < indices.length && eliteList.size() < eliteCount && result.size() < targetCount; i++) {
            int[] board = allBoards.get(indices[i]);
            if (!isEdgeConsistent(board)) continue;
            int[] copy = Arrays.copyOf(board, 256);
            boolean dup = false;
            for (int[] e : eliteList) if (Arrays.equals(e, copy)) { dup = true; break; }
            if (dup) continue;
            eliteList.add(copy);
            result.add(copy);
        }

        // ── TIER 2: Diverse ──
        int diverseFilled = 0;
        for (int i = 0; i < indices.length && diverseFilled < diverseCount && result.size() < targetCount; i++) {
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
            if (restartFilled >= restartCount || result.size() >= targetCount) break;
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
}
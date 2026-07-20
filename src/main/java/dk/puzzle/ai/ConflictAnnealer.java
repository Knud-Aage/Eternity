package dk.puzzle.ai;

import dk.puzzle.util.PieceUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * ConflictAnnealer: simulated-annealing edge-conflict minimiser for COMPLETE
 * 256-piece boards.
 *
 * <p>Why this exists: the ConflictReducer polish passes (rotationPass/swapPass)
 * only ever accept strictly-improving single moves, so they stop at the first
 * local optimum — and mcvRestartFill's greedy per-cell choices can't trade a
 * good early placement for a better global outcome. Getting below their floor
 * requires occasionally accepting a WORSE intermediate board to escape the
 * local optimum, which is exactly what annealing does: uphill moves are
 * accepted with probability exp(-delta/T), where T cools over each cycle from
 * {@link #T_START} to {@link #T_END}, ending in pure descent. Each cycle
 * restarts from the best board found so far (intensification) and, between
 * cycles, a Large-Neighbourhood kick clears a small window around a conflict
 * and re-solves it EXACTLY with the removed pieces — a multi-piece
 * rearrangement no sequence of single swaps could reach through the uphill
 * barrier.</p>
 *
 * <p>Move evaluation is incremental: only the &le;8 board edges incident to the
 * 1-2 touched cells are recomputed per candidate, never the whole board. The
 * two cells of a swap are always drawn from the same geometric class
 * (corner/border/interior cells), so a class-clean input stays class-clean.
 * Border-facing mismatches are weighted {@link #BORDER_WEIGHT}x in the energy
 * so the annealer never profits from turning a grey edge inward; the reported
 * conflict count remains the plain internal-edge count (x/480), directly
 * comparable to the rest of the codebase.</p>
 *
 * <p>Runs several independent walkers in parallel; they share (and restart
 * from) a single global best board. All methods are safe to call from any
 * thread; the input board is never modified.</p>
 */
public final class ConflictAnnealer {

    private static final int W = 16, H = 16, CELLS = 256;

    /**
     * Energy weight of a border-side mismatch (grey edge facing inward, or a
     * non-grey edge facing outward). Heavier than an internal mismatch so the
     * annealer can pass through — but never settle in — states that break the
     * frame.
     */
    private static final int BORDER_WEIGHT = 2;

    // --- SA schedule ---
    private static final double T_START = 1.0;
    private static final double T_END = 0.02;
    private static final int CYCLE_MOVES = 3_000_000;
    /** Rebuild the conflicted-cell sampling list every this many moves. */
    private static final int CONFLICT_LIST_REFRESH = 2048;
    /** Swap partners are drawn near the first cell this often (Chebyshev radius below). */
    private static final int LOCAL_SWAP_PERCENT = 70;
    private static final int LOCAL_SWAP_RADIUS = 4;

    // --- LNS (branch-and-bound window re-solve between cycles) ---
    private static final double LNS_PROBABILITY = 0.6;
    private static final long LNS_NODE_BUDGET = 3_000_000L;

    // --- Ruin-and-recreate: rebuild the whole conflict zone from scratch ---
    // Polished boards are provably near-locally-optimal (B&B windows find
    // nothing), so real gains need a different BASIN: destroy every cell near
    // a conflict and refill greedily with random tie-breaks, then anneal that.
    private static final double RUIN_PROBABILITY = 0.4;
    private static final int RUIN_DILATE_RADIUS = 2;

    /** Result of an optimisation run. {@code board} is a fresh array. */
    public record Result(int[] board, int internalConflicts, int borderConflicts,
                         long moves, long lnsKicks, long lnsWins) {}

    /** Progress callback: called with (boardCopy, internalConflicts) on each new global best. */
    public interface ImprovementListener {
        void onImprovement(int[] board, int internalConflicts, long elapsedMillis);
    }

    private final boolean[] locked = new boolean[CELLS];
    private final int[] cornerCells;
    private final int[] borderCells;
    private final int[] interiorCells;

    private final Object bestLock = new Object();
    private final int[] bestBoard = new int[CELLS];
    private final AtomicInteger bestEnergy = new AtomicInteger(Integer.MAX_VALUE);
    private volatile ImprovementListener listener;
    private long startNanos;

    private final java.util.concurrent.atomic.AtomicLong totalMoves = new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.AtomicLong totalKicks = new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.AtomicLong kickWins = new java.util.concurrent.atomic.AtomicLong();

    private ConflictAnnealer(Set<Integer> lockedCells) {
        if (lockedCells != null) {
            for (int c : lockedCells) locked[c] = true;
        }
        List<Integer> corners = new ArrayList<>(), border = new ArrayList<>(), interior = new ArrayList<>();
        for (int pos = 0; pos < CELLS; pos++) {
            if (locked[pos]) continue;
            switch (cellClass(pos)) {
                case 0 -> corners.add(pos);
                case 1 -> border.add(pos);
                default -> interior.add(pos);
            }
        }
        cornerCells = corners.stream().mapToInt(Integer::intValue).toArray();
        borderCells = border.stream().mapToInt(Integer::intValue).toArray();
        interiorCells = interior.stream().mapToInt(Integer::intValue).toArray();
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /**
     * Minimise edge conflicts on a complete board.
     *
     * @param board       Complete 256-piece board (not modified).
     * @param lockedCells Cells that must not be touched (e.g. center/hints), or null.
     * @param timeMillis  Wall-clock budget.
     * @param threads     Parallel walkers (each is one CPU thread).
     * @param listener    Optional progress callback, or null.
     */
    public static Result optimize(int[] board, Set<Integer> lockedCells, long timeMillis,
                                  int threads, ImprovementListener listener) {
        for (int p : board) {
            if (p == -1 || p == -2) {
                throw new IllegalArgumentException(
                        "ConflictAnnealer needs a complete board — fill holes first (mcvRestartFill/HoleSolver).");
            }
        }
        ConflictAnnealer annealer = new ConflictAnnealer(lockedCells);
        annealer.listener = listener;
        annealer.startNanos = System.nanoTime();

        // Boards from the greedy fill can arrive "class-dirty": border pieces
        // stranded in the interior and vice versa (invisible to the pipeline's
        // internal-only conflict count, but real errors). Repair that first —
        // class-preserving SA moves could never fix it afterwards.
        int[] seedBoard = Arrays.copyOf(board, CELLS);
        annealer.normalizeClasses(seedBoard);
        annealer.orientFrame(seedBoard);

        System.arraycopy(seedBoard, 0, annealer.bestBoard, 0, CELLS);
        annealer.bestEnergy.set(annealer.energy(seedBoard));

        long deadline = annealer.startNanos + timeMillis * 1_000_000L;
        int walkers = Math.max(1, threads);
        CountDownLatch done = new CountDownLatch(walkers);
        for (int i = 0; i < walkers; i++) {
            final long seed = System.nanoTime() * 31 + i * 0x9E3779B97F4A7C15L;
            Thread t = new Thread(() -> {
                try {
                    annealer.runWalker(new Random(seed), deadline);
                } finally {
                    done.countDown();
                }
            }, "conflict-annealer-" + i);
            t.setDaemon(true);
            t.start();
        }
        try {
            done.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        int[] result;
        synchronized (annealer.bestLock) {
            result = Arrays.copyOf(annealer.bestBoard, CELLS);
        }
        return new Result(result, countInternalConflicts(result), countBorderViolations(result),
                annealer.totalMoves.get(), annealer.totalKicks.get(), annealer.kickWins.get());
    }

    /**
     * Standalone class/frame repair: moves stranded border pieces back to the
     * frame (and vice versa) and rotates all frame pieces grey-outward.
     * Modifies {@code board} in place.
     */
    public static void normalize(int[] board, Set<Integer> lockedCells) {
        ConflictAnnealer a = new ConflictAnnealer(lockedCells);
        a.normalizeClasses(board);
        a.orientFrame(board);
    }

    /** Internal edge mismatches (out of 480) — same definition as ConflictReducer.countConflicts. */
    public static int countInternalConflicts(int[] board) {
        int total = 0;
        for (int pos = 0; pos < CELLS; pos++) {
            int p = board[pos];
            if (p == -1 || p == -2) continue;
            if (pos % W < W - 1) {
                int right = board[pos + 1];
                if (right != -1 && right != -2 && PieceUtils.getEast(p) != PieceUtils.getWest(right)) total++;
            }
            if (pos / W < H - 1) {
                int below = board[pos + W];
                if (below != -1 && below != -2 && PieceUtils.getSouth(p) != PieceUtils.getNorth(below)) total++;
            }
        }
        return total;
    }

    /** Outer-frame violations: non-grey facing off-board (out of 64). */
    public static int countBorderViolations(int[] board) {
        int total = 0;
        for (int pos = 0; pos < CELLS; pos++) {
            int p = board[pos];
            if (p == -1 || p == -2) continue;
            int row = pos / W, col = pos % W;
            if (row == 0 && PieceUtils.getNorth(p) != PieceUtils.BORDER_COLOR) total++;
            if (row == H - 1 && PieceUtils.getSouth(p) != PieceUtils.BORDER_COLOR) total++;
            if (col == 0 && PieceUtils.getWest(p) != PieceUtils.BORDER_COLOR) total++;
            if (col == W - 1 && PieceUtils.getEast(p) != PieceUtils.BORDER_COLOR) total++;
        }
        return total;
    }

    // -----------------------------------------------------------------------
    // Walker: SA cycles + LNS kicks
    // -----------------------------------------------------------------------

    private void runWalker(Random rnd, long deadline) {
        int[] cur = new int[CELLS];
        int[] conflictCells = new int[CELLS];
        double coolFactor = Math.exp(Math.log(T_END / T_START) / CYCLE_MOVES);
        boolean haveOwnBoard = false;

        while (System.nanoTime() < deadline && bestEnergy.get() > 0) {

            // Keep an independent trajectory most cycles — restarting every
            // cycle from the shared best degenerates into repeated memoryless
            // shots at the same deep local optimum. Re-sync occasionally.
            if (!haveOwnBoard || rnd.nextInt(100) < 30) {
                synchronized (bestLock) {
                    System.arraycopy(bestBoard, 0, cur, 0, CELLS);
                }
                haveOwnBoard = true;
            }
            if (rnd.nextDouble() < RUIN_PROBABILITY) {
                ruinAndRecreate(cur, rnd);
            }
            int e = energy(cur);
            double t = T_START;
            int numConflictCells = collectConflictCells(cur, conflictCells);
            if (numConflictCells == 0) return; // solved

            for (int m = 0; m < CYCLE_MOVES; m++) {
                if ((m & 4095) == 0 && System.nanoTime() >= deadline) return;
                if ((m % CONFLICT_LIST_REFRESH) == 0) {
                    numConflictCells = collectConflictCells(cur, conflictCells);
                    if (numConflictCells == 0) break;
                }

                int delta;
                int roll = rnd.nextInt(100);
                if (roll < 15) {
                    delta = rotationMove(cur, conflictCells, numConflictCells, t, rnd);
                } else {
                    boolean uniform = roll >= 85; // 15% pure exploration
                    delta = swapMove(cur, conflictCells, numConflictCells, uniform, t, rnd);
                }
                if (delta != Integer.MIN_VALUE) {
                    e += delta;
                    if (e < bestEnergy.get()) {
                        publishBest(cur, e);
                        if (e == 0) return;
                    }
                }
                t *= coolFactor;
            }

            totalMoves.addAndGet(CYCLE_MOVES);

            // Between cycles: exact re-solve of a small window around a conflict.
            if (rnd.nextDouble() < LNS_PROBABILITY) {
                totalKicks.incrementAndGet();
                lnsKick(rnd, deadline);
            }
        }
    }

    /** Try a rotation of a conflicted interior cell. Returns energy delta if applied, MIN_VALUE if not. */
    private int rotationMove(int[] board, int[] conflictCells, int numConflictCells, double t, Random rnd) {
        // Find a conflicted interior cell (a few attempts, then give up this move)
        for (int attempt = 0; attempt < 4; attempt++) {
            int pos = conflictCells[rnd.nextInt(numConflictCells)];
            if (cellClass(pos) != 2 || locked[pos]) continue;

            int piece = board[pos];
            int oldCost = cellCost(board, pos);
            int cand = piece;
            int bestPiece = piece, bestCost = oldCost;
            for (int r = 0; r < 3; r++) {
                cand = PieceUtils.rotate(cand);
                board[pos] = cand;
                int c = cellCost(board, pos);
                if (c < bestCost || (c == bestCost && rnd.nextBoolean())) {
                    bestCost = c;
                    bestPiece = cand;
                }
            }
            board[pos] = piece;

            int delta = bestCost - oldCost;
            if (bestPiece != piece && accept(delta, t, rnd)) {
                board[pos] = bestPiece;
                return delta;
            }
            return Integer.MIN_VALUE;
        }
        return Integer.MIN_VALUE;
    }

    /** Try swapping a (usually conflicted) cell with a random same-class cell. */
    private int swapMove(int[] board, int[] conflictCells, int numConflictCells,
                         boolean uniformFirst, double t, Random rnd) {
        int a;
        if (uniformFirst) {
            a = randomCellOfAnyClass(rnd);
        } else {
            a = conflictCells[rnd.nextInt(numConflictCells)];
            if (locked[a]) return Integer.MIN_VALUE;
        }
        int[] classCells = switch (cellClass(a)) {
            case 0 -> cornerCells;
            case 1 -> borderCells;
            default -> interiorCells;
        };
        if (classCells.length < 2) return Integer.MIN_VALUE;

        // Mostly swap with a nearby same-class cell: productive rearrangements
        // are overwhelmingly local to the conflict zone, and uniform partners
        // spend most moves churning the already-perfect part of the board.
        int b = -1;
        if (cellClass(a) == 2 && rnd.nextInt(100) < LOCAL_SWAP_PERCENT) {
            for (int attempt = 0; attempt < 6; attempt++) {
                int r = a / W + rnd.nextInt(2 * LOCAL_SWAP_RADIUS + 1) - LOCAL_SWAP_RADIUS;
                int c = a % W + rnd.nextInt(2 * LOCAL_SWAP_RADIUS + 1) - LOCAL_SWAP_RADIUS;
                if (r < 1 || r > H - 2 || c < 1 || c > W - 2) continue; // interior only
                int cand = r * W + c;
                if (cand != a && !locked[cand]) {
                    b = cand;
                    break;
                }
            }
        }
        if (b == -1) {
            b = classCells[rnd.nextInt(classCells.length)];
            if (b == a) return Integer.MIN_VALUE;
        }

        int pa = board[a], pb = board[b];
        boolean adjacent = areAdjacent(a, b);
        int oldCost = pairCost(board, a, b, adjacent);

        int bestCost = Integer.MAX_VALUE;
        int bestPa = 0, bestPb = 0;
        int ties = 0;

        if (adjacent) {
            int candA = pb;
            for (int ra = 0; ra < 4; ra++, candA = PieceUtils.rotate(candA)) {
                int candB = pa;
                for (int rb = 0; rb < 4; rb++, candB = PieceUtils.rotate(candB)) {
                    board[a] = candA;
                    board[b] = candB;
                    int c = pairCost(board, a, b, true);
                    if (c < bestCost) {
                        bestCost = c;
                        bestPa = candA;
                        bestPb = candB;
                        ties = 1;
                    } else if (c == bestCost && rnd.nextInt(++ties) == 0) {
                        bestPa = candA;
                        bestPb = candB;
                    }
                }
            }
        } else {
            // Independent: pick the best rotation at each cell separately.
            board[b] = pb; // untouched while evaluating a
            int candA = pb;
            int bestCa = Integer.MAX_VALUE;
            int tiesA = 0;
            for (int r = 0; r < 4; r++, candA = PieceUtils.rotate(candA)) {
                board[a] = candA;
                int c = cellCost(board, a);
                if (c < bestCa) {
                    bestCa = c;
                    bestPa = candA;
                    tiesA = 1;
                } else if (c == bestCa && rnd.nextInt(++tiesA) == 0) {
                    bestPa = candA;
                }
            }
            board[a] = bestPa;
            int candB = pa;
            int bestCb = Integer.MAX_VALUE;
            int tiesB = 0;
            for (int r = 0; r < 4; r++, candB = PieceUtils.rotate(candB)) {
                board[b] = candB;
                int c = cellCost(board, b);
                if (c < bestCb) {
                    bestCb = c;
                    bestPb = candB;
                    tiesB = 1;
                } else if (c == bestCb && rnd.nextInt(++tiesB) == 0) {
                    bestPb = candB;
                }
            }
            bestCost = bestCa + bestCb;
        }

        board[a] = pa;
        board[b] = pb;

        int delta = bestCost - oldCost;
        if (accept(delta, t, rnd)) {
            board[a] = bestPa;
            board[b] = bestPb;
            return delta;
        }
        return Integer.MIN_VALUE;
    }

    private boolean accept(int delta, double t, Random rnd) {
        if (delta <= 0) return true;
        return rnd.nextDouble() < Math.exp(-delta / t);
    }

    private int randomCellOfAnyClass(Random rnd) {
        int pos = rnd.nextInt(CELLS);
        while (locked[pos]) pos = rnd.nextInt(CELLS);
        return pos;
    }

    private void publishBest(int[] board, int e) {
        synchronized (bestLock) {
            if (e >= bestEnergy.get()) return;
            bestEnergy.set(e);
            System.arraycopy(board, 0, bestBoard, 0, CELLS);
            ImprovementListener l = listener;
            if (l != null) {
                l.onImprovement(Arrays.copyOf(board, CELLS), countInternalConflicts(board),
                        (System.nanoTime() - startNanos) / 1_000_000L);
            }
        }
    }

    // -----------------------------------------------------------------------
    // LNS: branch-and-bound re-arrangement of a window around a conflict.
    // Unlike a zero-or-nothing exact refill, this KEEPS the best arrangement
    // found even when zero conflicts is impossible — any strictly better
    // multi-piece rearrangement is an improvement no single swap could reach.
    // -----------------------------------------------------------------------

    private void lnsKick(Random rnd, long deadline) {
        int[] cand;
        synchronized (bestLock) {
            cand = Arrays.copyOf(bestBoard, CELLS);
        }
        int[] conflictCells = new int[CELLS];
        int n = collectConflictCells(cand, conflictCells);
        if (n == 0) return;

        int center = conflictCells[rnd.nextInt(n)];
        int radius = rnd.nextInt(3) == 0 ? 1 : 2; // mostly 5x5, sometimes 3x3

        List<Integer> windowList = new ArrayList<>();
        int cr = center / W, cc = center % W;
        for (int r = Math.max(0, cr - radius); r <= Math.min(H - 1, cr + radius); r++) {
            for (int c = Math.max(0, cc - radius); c <= Math.min(W - 1, cc + radius); c++) {
                int pos = r * W + c;
                if (!locked[pos]) windowList.add(pos);
            }
        }
        if (windowList.size() < 2) return;
        int[] window = windowList.stream().mapToInt(Integer::intValue).toArray();

        if (bnbWindowSearch(cand, window, rnd, deadline)) {
            kickWins.incrementAndGet();
            int e = energy(cand);
            if (e < bestEnergy.get()) {
                publishBest(cand, e);
            }
        }
    }

    /**
     * Branch-and-bound: removes the window's pieces and searches arrangements
     * of exactly those pieces over exactly those cells, minimising the summed
     * cost of every edge incident to the window. The incumbent bound is the
     * ORIGINAL arrangement's cost, so anything found is strictly better.
     * Frame greys are hard constraints (the frame never gets broken again).
     * Returns true (with {@code board} updated) if a better arrangement was found.
     */
    private boolean bnbWindowSearch(int[] board, int[] window, Random rnd, long deadline) {
        int currentCost = windowArrangementCost(board, window);
        if (currentCost == 0) return false;

        int[] pool = new int[window.length];
        for (int i = 0; i < window.length; i++) {
            pool[i] = board[window[i]];
            board[window[i]] = -1;
        }

        BnbState st = new BnbState(board, window, pool, currentCost, deadline, rnd);
        st.dfs(0, window.length);

        if (st.bestFill != null) {
            for (int i = 0; i < window.length; i++) board[window[i]] = st.bestFill[i];
            return true;
        }
        for (int i = 0; i < window.length; i++) board[window[i]] = pool[i]; // restore
        return false;
    }

    /** Summed cost of every edge incident to the window, each counted once. */
    private int windowArrangementCost(int[] board, int[] window) {
        boolean[] inWindow = new boolean[CELLS];
        for (int pos : window) inWindow[pos] = true;
        int cost = 0;
        for (int pos : window) {
            int p = board[pos];
            int row = pos / W, col = pos % W;
            // North/West edges are always counted from this cell (for an
            // in-window neighbour that counts the shared edge exactly once,
            // from its southern/eastern member); South/East edges only when
            // the neighbour is outside the window.
            if (row == 0) {
                if (PieceUtils.getNorth(p) != PieceUtils.BORDER_COLOR) cost += BORDER_WEIGHT;
            } else if (PieceUtils.getNorth(p) != PieceUtils.getSouth(board[pos - W])) {
                cost++;
            }
            if (col == 0) {
                if (PieceUtils.getWest(p) != PieceUtils.BORDER_COLOR) cost += BORDER_WEIGHT;
            } else if (PieceUtils.getWest(p) != PieceUtils.getEast(board[pos - 1])) {
                cost++;
            }
            if (row == H - 1) {
                if (PieceUtils.getSouth(p) != PieceUtils.BORDER_COLOR) cost += BORDER_WEIGHT;
            } else if (!inWindow[pos + W] && PieceUtils.getSouth(p) != PieceUtils.getNorth(board[pos + W])) {
                cost++;
            }
            if (col == W - 1) {
                if (PieceUtils.getEast(p) != PieceUtils.BORDER_COLOR) cost += BORDER_WEIGHT;
            } else if (!inWindow[pos + 1] && PieceUtils.getEast(p) != PieceUtils.getWest(board[pos + 1])) {
                cost++;
            }
        }
        return cost;
    }

    /** DFS state for the window branch-and-bound. */
    private final class BnbState {
        final int[] board;
        final int[] window;
        final int[] pool;
        final boolean[] used;
        final long deadline;
        final Random rnd;
        int bestCost;      // incumbent = original arrangement's cost
        int[] bestFill;    // pieces for window cells in window order, or null
        long nodes = 0;

        BnbState(int[] board, int[] window, int[] pool, int currentCost, long deadline, Random rnd) {
            this.board = board;
            this.window = window;
            this.pool = pool;
            this.used = new boolean[pool.length];
            this.bestCost = currentCost;
            this.deadline = deadline;
            this.rnd = rnd;
        }

        void dfs(int costSoFar, int remaining) {
            if (nodes++ >= LNS_NODE_BUDGET) return;
            if ((nodes & 0xFFFF) == 0 && System.nanoTime() >= deadline) return;
            if (remaining == 0) {
                if (costSoFar < bestCost) {
                    bestCost = costSoFar;
                    bestFill = new int[window.length];
                    for (int i = 0; i < window.length; i++) bestFill[i] = board[window[i]];
                }
                return;
            }

            // Most-constrained cell: the unfilled window cell with the most
            // already-known neighbouring edges.
            int cell = -1, bestKnown = -1;
            for (int pos : window) {
                if (board[pos] != -1) continue;
                int known = 0;
                int row = pos / W, col = pos % W;
                if (row == 0 || board[pos - W] != -1) known++;
                if (row == H - 1 || board[pos + W] != -1) known++;
                if (col == 0 || board[pos - 1] != -1) known++;
                if (col == W - 1 || board[pos + 1] != -1) known++;
                if (known > bestKnown) {
                    bestKnown = known;
                    cell = pos;
                }
            }

            int reqN = requiredColor(board, cell, -W);
            int reqE = requiredColor(board, cell, +1);
            int reqS = requiredColor(board, cell, +W);
            int reqW = requiredColor(board, cell, -1);
            int row = cell / W, col = cell % W;
            int wantedGreys = switch (cellClass(cell)) {
                case 0 -> 2;
                case 1 -> 1;
                default -> 0;
            };

            // Gather candidates with their placement cost; try cheapest first.
            List<int[]> candidates = new ArrayList<>(); // {piece, poolIdx, cost}
            for (int k = 0; k < pool.length; k++) {
                if (used[k]) continue;
                // Exact class match: keeps the frame legal AND stops corner
                // pieces from being consumed by ordinary edge cells.
                if (PieceUtils.getBorderCount(pool[k]) != wantedGreys) continue;
                int piece = pool[k];
                for (int r = 0; r < 4; r++, piece = PieceUtils.rotate(piece)) {
                    // Frame greys are hard: never rebreak the frame.
                    if (row == 0 && PieceUtils.getNorth(piece) != PieceUtils.BORDER_COLOR) continue;
                    if (row == H - 1 && PieceUtils.getSouth(piece) != PieceUtils.BORDER_COLOR) continue;
                    if (col == 0 && PieceUtils.getWest(piece) != PieceUtils.BORDER_COLOR) continue;
                    if (col == W - 1 && PieceUtils.getEast(piece) != PieceUtils.BORDER_COLOR) continue;

                    int c = 0;
                    if (reqN >= 0 && row != 0 && PieceUtils.getNorth(piece) != reqN) c++;
                    if (reqE >= 0 && col != W - 1 && PieceUtils.getEast(piece) != reqE) c++;
                    if (reqS >= 0 && row != H - 1 && PieceUtils.getSouth(piece) != reqS) c++;
                    if (reqW >= 0 && col != 0 && PieceUtils.getWest(piece) != reqW) c++;
                    if (costSoFar + c < bestCost) {
                        candidates.add(new int[]{piece, k, c});
                    }
                }
            }
            if (candidates.isEmpty()) return;
            candidates.sort((x, y) -> x[2] - y[2]);

            for (int[] cnd : candidates) {
                if (costSoFar + cnd[2] >= bestCost) break; // sorted — the rest are no better
                board[cell] = cnd[0];
                used[cnd[1]] = true;
                dfs(costSoFar + cnd[2], remaining - 1);
                board[cell] = -1;
                used[cnd[1]] = false;
                if (nodes >= LNS_NODE_BUDGET) return;
            }
        }
    }

    /**
     * Colour the piece at {@code cell} must show toward {@code cell+offset}:
     * grey (0) when the neighbour is off-board, the neighbour's opposing
     * colour when placed, -1 (wildcard) when the neighbour is an unfilled
     * window cell.
     */
    private int requiredColor(int[] board, int cell, int offset) {
        int row = cell / W, col = cell % W;
        boolean offBoard = switch (offset) {
            case -W -> row == 0;
            case +W -> row == H - 1;
            case -1 -> col == 0;
            case +1 -> col == W - 1;
            default -> throw new IllegalArgumentException();
        };
        if (offBoard) return PieceUtils.BORDER_COLOR;
        int p = board[cell + offset];
        if (p == -1) return -1;
        return switch (offset) {
            case -W -> PieceUtils.getSouth(p);
            case +W -> PieceUtils.getNorth(p);
            case -1 -> PieceUtils.getEast(p);
            default -> PieceUtils.getWest(p);
        };
    }

    // -----------------------------------------------------------------------
    // Ruin-and-recreate
    // -----------------------------------------------------------------------

    /**
     * Destroys every cell within {@link #RUIN_DILATE_RADIUS} of a conflict and
     * refills the zone greedily (most-constrained cell first, cheapest
     * piece/orientation, random tie-breaks). The refill respects the frame as
     * a hard constraint, so the zone may bleed into the perfect region and
     * recruit its pieces — a freedom the pipeline's hole-fill never had.
     */
    private void ruinAndRecreate(int[] board, Random rnd) {
        boolean[] zone = new boolean[CELLS];
        int zoneSize = 0;
        for (int pos = 0; pos < CELLS; pos++) {
            if (cellCost(board, pos) == 0) continue;
            int cr = pos / W, cc = pos % W;
            for (int r = Math.max(0, cr - RUIN_DILATE_RADIUS); r <= Math.min(H - 1, cr + RUIN_DILATE_RADIUS); r++) {
                for (int c = Math.max(0, cc - RUIN_DILATE_RADIUS); c <= Math.min(W - 1, cc + RUIN_DILATE_RADIUS); c++) {
                    int p = r * W + c;
                    if (!zone[p] && !locked[p]) {
                        zone[p] = true;
                        zoneSize++;
                    }
                }
            }
        }
        if (zoneSize < 2) return;

        int[] zoneCells = new int[zoneSize];
        int[] pool = new int[zoneSize];
        int n = 0;
        for (int pos = 0; pos < CELLS; pos++) {
            if (zone[pos]) {
                zoneCells[n] = pos;
                pool[n++] = board[pos];
                board[pos] = -1;
            }
        }

        boolean[] used = new boolean[zoneSize];
        for (int fill = 0; fill < zoneSize; fill++) {
            // Most-constrained unfilled cell (most known neighbouring edges).
            int cell = -1, bestKnown = -1, cellTies = 0;
            for (int pos : zoneCells) {
                if (board[pos] != -1) continue;
                int row = pos / W, col = pos % W;
                int known = 0;
                if (row == 0 || board[pos - W] != -1) known++;
                if (row == H - 1 || board[pos + W] != -1) known++;
                if (col == 0 || board[pos - 1] != -1) known++;
                if (col == W - 1 || board[pos + 1] != -1) known++;
                if (known > bestKnown) {
                    bestKnown = known;
                    cell = pos;
                    cellTies = 1;
                } else if (known == bestKnown && rnd.nextInt(++cellTies) == 0) {
                    cell = pos;
                }
            }

            int row = cell / W, col = cell % W;
            // Exact class match is essential for a greedy fill: a corner piece
            // also satisfies an edge cell's single-grey requirement, and letting
            // one get consumed there strands a corner cell with no legal piece.
            int wantedGreys = switch (cellClass(cell)) {
                case 0 -> 2;
                case 1 -> 1;
                default -> 0;
            };
            int reqN = requiredColor(board, cell, -W);
            int reqE = requiredColor(board, cell, +1);
            int reqS = requiredColor(board, cell, +W);
            int reqW = requiredColor(board, cell, -1);

            int bestPiece = -1, bestIdx = -1, bestCost = Integer.MAX_VALUE, ties = 0;
            for (int k = 0; k < zoneSize; k++) {
                if (used[k]) continue;
                if (PieceUtils.getBorderCount(pool[k]) != wantedGreys) continue;
                int piece = pool[k];
                for (int r = 0; r < 4; r++, piece = PieceUtils.rotate(piece)) {
                    // Frame greys are hard, exactly as in the B&B.
                    if (row == 0 && PieceUtils.getNorth(piece) != PieceUtils.BORDER_COLOR) continue;
                    if (row == H - 1 && PieceUtils.getSouth(piece) != PieceUtils.BORDER_COLOR) continue;
                    if (col == 0 && PieceUtils.getWest(piece) != PieceUtils.BORDER_COLOR) continue;
                    if (col == W - 1 && PieceUtils.getEast(piece) != PieceUtils.BORDER_COLOR) continue;

                    int c = 0;
                    if (reqN >= 0 && row != 0 && PieceUtils.getNorth(piece) != reqN) c++;
                    if (reqE >= 0 && col != W - 1 && PieceUtils.getEast(piece) != reqE) c++;
                    if (reqS >= 0 && row != H - 1 && PieceUtils.getSouth(piece) != reqS) c++;
                    if (reqW >= 0 && col != 0 && PieceUtils.getWest(piece) != reqW) c++;
                    if (c < bestCost) {
                        bestCost = c;
                        bestPiece = piece;
                        bestIdx = k;
                        ties = 1;
                    } else if (c == bestCost && rnd.nextInt(++ties) == 0) {
                        bestPiece = piece;
                        bestIdx = k;
                    }
                }
            }
            if (bestPiece == -1) {
                // Defensive: should be impossible with exact class matching,
                // but never crash the walker — place any leftover piece.
                for (int k = 0; k < zoneSize; k++) {
                    if (!used[k]) {
                        bestPiece = pool[k];
                        bestIdx = k;
                        break;
                    }
                }
            }
            board[cell] = bestPiece;
            used[bestIdx] = true;
        }
    }

    // -----------------------------------------------------------------------
    // Class normalisation (pre-pass)
    // -----------------------------------------------------------------------

    /**
     * Moves every piece to a cell of its own class: corner pieces (2 grey
     * edges) onto corner cells, border pieces (1 grey) onto border cells,
     * interior pieces onto interior cells. Pieces already in place are left
     * alone; the misplaced ones are pairwise swapped (a wrong-class cell
     * always has a matching stranded piece elsewhere, since piece counts per
     * class equal cell counts per class).
     */
    private void normalizeClasses(int[] board) {
        for (int wanted = 0; wanted <= 1; wanted++) { // 0=corner cells, 1=border cells
            int wantedGreys = wanted == 0 ? 2 : 1;
            for (int cell = 0; cell < CELLS; cell++) {
                if (cellClass(cell) != wanted || locked[cell]) continue;
                if (PieceUtils.getBorderCount(board[cell]) == wantedGreys) continue;

                // Find a stranded piece of the class this cell needs.
                for (int other = 0; other < CELLS; other++) {
                    if (other == cell || locked[other] || cellClass(other) == wanted) continue;
                    if (PieceUtils.getBorderCount(board[other]) != wantedGreys) continue;
                    int tmp = board[cell];
                    board[cell] = board[other];
                    board[other] = tmp;
                    break;
                }
            }
        }
    }

    /** Rotates every frame piece so its grey edge(s) face off-board. */
    private void orientFrame(int[] board) {
        for (int cell = 0; cell < CELLS; cell++) {
            if (cellClass(cell) == 2 || locked[cell]) continue;
            int row = cell / W, col = cell % W;
            int piece = board[cell];
            for (int r = 0; r < 4; r++) {
                boolean ok =
                        (row != 0 || PieceUtils.getNorth(piece) == PieceUtils.BORDER_COLOR) &&
                        (row != H - 1 || PieceUtils.getSouth(piece) == PieceUtils.BORDER_COLOR) &&
                        (col != 0 || PieceUtils.getWest(piece) == PieceUtils.BORDER_COLOR) &&
                        (col != W - 1 || PieceUtils.getEast(piece) == PieceUtils.BORDER_COLOR);
                if (ok) break;
                piece = PieceUtils.rotate(piece);
            }
            board[cell] = piece; // best orientation found (exact fit when class-clean)
        }
    }

    // -----------------------------------------------------------------------
    // Energy / cost helpers
    // -----------------------------------------------------------------------

    /** Full-board energy: internal mismatches + BORDER_WEIGHT * frame violations. */
    private int energy(int[] board) {
        return countInternalConflicts(board) + BORDER_WEIGHT * countBorderViolations(board);
    }

    /** Cost of the up-to-4 edges incident to {@code pos} (each internal edge counted once from this cell). */
    private int cellCost(int[] board, int pos) {
        int p = board[pos];
        int row = pos / W, col = pos % W;
        int cost = 0;
        if (row == 0) {
            if (PieceUtils.getNorth(p) != PieceUtils.BORDER_COLOR) cost += BORDER_WEIGHT;
        } else if (PieceUtils.getNorth(p) != PieceUtils.getSouth(board[pos - W])) cost++;
        if (row == H - 1) {
            if (PieceUtils.getSouth(p) != PieceUtils.BORDER_COLOR) cost += BORDER_WEIGHT;
        } else if (PieceUtils.getSouth(p) != PieceUtils.getNorth(board[pos + W])) cost++;
        if (col == 0) {
            if (PieceUtils.getWest(p) != PieceUtils.BORDER_COLOR) cost += BORDER_WEIGHT;
        } else if (PieceUtils.getWest(p) != PieceUtils.getEast(board[pos - 1])) cost++;
        if (col == W - 1) {
            if (PieceUtils.getEast(p) != PieceUtils.BORDER_COLOR) cost += BORDER_WEIGHT;
        } else if (PieceUtils.getEast(p) != PieceUtils.getWest(board[pos + 1])) cost++;
        return cost;
    }

    /** Cost of all edges incident to a or b, counting the shared edge (if adjacent) once. */
    private int pairCost(int[] board, int a, int b, boolean adjacent) {
        int cost = cellCost(board, a) + cellCost(board, b);
        if (adjacent) {
            // The a-b edge was counted from both sides; remove one copy.
            int lo = Math.min(a, b), hi = Math.max(a, b);
            if (hi - lo == 1) {
                if (PieceUtils.getEast(board[lo]) != PieceUtils.getWest(board[hi])) cost--;
            } else {
                if (PieceUtils.getSouth(board[lo]) != PieceUtils.getNorth(board[hi])) cost--;
            }
        }
        return cost;
    }

    private static boolean areAdjacent(int a, int b) {
        int diff = Math.abs(a - b);
        if (diff == W) return true;
        return diff == 1 && a / W == b / W;
    }

    /** 0 = corner cell, 1 = non-corner border cell, 2 = interior cell. */
    private static int cellClass(int pos) {
        int row = pos / W, col = pos % W;
        boolean rEdge = row == 0 || row == H - 1;
        boolean cEdge = col == 0 || col == W - 1;
        if (rEdge && cEdge) return 0;
        if (rEdge || cEdge) return 1;
        return 2;
    }

    /** Fills {@code out} with cells whose incident edges have any cost; returns the count. */
    private int collectConflictCells(int[] board, int[] out) {
        int n = 0;
        for (int pos = 0; pos < CELLS; pos++) {
            if (!locked[pos] && cellCost(board, pos) > 0) out[n++] = pos;
        }
        return n;
    }
}
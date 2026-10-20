package dk.puzzle.tools;

import dk.puzzle.util.PieceUtils;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Randomised large-neighbourhood search with plateau drift for a complete board.
 *
 * <p>Each round picks a handful of cells (4 to 8), lifts their pieces out and puts them back in the best way there is:
 * every assignment of those pieces to those cells with every rotation is searched exactly (depth first, the conflicts
 * around the cells as the bound). A rearrangement with the SAME conflict count as the one it replaces is taken as well,
 * so the board keeps drifting across the flat region of equal-cost boards until some rearrangement lowers the count.
 * This is the recipe a member of the eternity2 group described on Oct 20 2026 (exact re-placement of several tiles plus
 * plateau moves, thousands of rounds per board). {@link PairMovePolish} covers every one- and two-piece move, but its
 * walk over equal-cost boards is only a few steps deep and it never moves three or more pieces at once.</p>
 *
 * <p>Four kinds of region are drawn: a cluster around a conflict, a conflicted cell with scattered partners, two
 * conflict sites at once, and a cluster anywhere (re-arranging a conflict-free neighbourhood moves the board sideways).
 * Measured offline on boards the pipeline had already finished, 20,000 rounds per board (2 to 4 s): 16 of 342 saved
 * 14s became 13s (the pair polish alone: 12), 7 of 427 saved 13s became 12s (the pair polish: 5), and the GPU 12
 * Errors12_Base250_133615_502 of Aug 30 became an 11, from two different random seeds, to the same board. Most of the
 * gains are rearrangements of 2 to 5 pieces between separate conflict spots.</p>
 *
 * <p>Board layout is HoleSolver's: 256 packed pieces, row-major from the top row. The centre cell never moves, and
 * neither does any cell the caller pins (the clue pieces when hints are on). The search is deterministic for a given
 * board (the random seed is derived from it). Only worth running on boards that are already good, which are rare:
 * the default ceiling is 14 conflicts without hints and 19 with. A result is accepted only after the conflict count,
 * the piece set and the locked cells have been re-verified.</p>
 */
public final class PlateauLns {

    private static volatile boolean enabled = !"false".equalsIgnoreCase(System.getenv("ETERNITY_LNS"));
    private static volatile int rounds = 20_000;
    // A board with this many conflicts or fewer gets the larger budget: such boards are rare (about one a day) and the
    // payoff of one fewer conflict is the biggest.
    private static volatile int deepRounds = 100_000;
    private static volatile int deepAtOrBelow = 12;
    private static volatile long maxMillis = 15_000L;
    // Same ceilings as PairMovePolish: the keep thresholds are 13 conflicts without hints and 18 with, and a board one
    // above can still become a keepable one.
    private static volatile int maxConflicts = 14;
    private static volatile int maxConflictsWithHints = 19;

    private static final AtomicLong tried = new AtomicLong(), improvedBoards = new AtomicLong(),
            conflictsRemoved = new AtomicLong(), roundsRun = new AtomicLong(), plateauMoves = new AtomicLong(),
            cappedRounds = new AtomicLong(), timedOut = new AtomicLong(), discarded = new AtomicLong(), nanosSpent = new AtomicLong();

    static {
        try {
            rounds = envInt("ETERNITY_LNS_ROUNDS", rounds);
            deepRounds = envInt("ETERNITY_LNS_DEEP_ROUNDS", deepRounds);
            deepAtOrBelow = envInt("ETERNITY_LNS_DEEP_AT_OR_BELOW", deepAtOrBelow);
            maxMillis = envInt("ETERNITY_LNS_MILLIS", (int) maxMillis);
            maxConflicts = envInt("ETERNITY_LNS_MAX_CONFLICTS", maxConflicts);
            maxConflictsWithHints = envInt("ETERNITY_LNS_MAX_CONFLICTS_HINTS", maxConflictsWithHints);
        } catch (NumberFormatException e) {
            System.err.println("PlateauLns: ignoring malformed ETERNITY_LNS_* setting (" + e.getMessage() + ")");
        }
    }

    private static int envInt(String name, int fallback) {
        String v = System.getenv(name);
        return v == null || v.isBlank() ? fallback : Integer.parseInt(v.trim());
    }

    private PlateauLns() {
    }

    /** Test/runtime hook: replaces the configuration. */
    static void configure(boolean on, int normalRounds, int deepBudgetRounds, int deepAtOrBelowConflicts, long millis,
                          int maxWithoutHints, int maxWithHints) {
        enabled = on;
        rounds = normalRounds;
        deepRounds = deepBudgetRounds;
        deepAtOrBelow = deepAtOrBelowConflicts;
        maxMillis = millis;
        maxConflicts = maxWithoutHints;
        maxConflictsWithHints = maxWithHints;
    }

    public static String stats() {
        return String.format("lns[tried=%d improved=%d conflictsRemoved=%d rounds=%d plateauMoves=%d cappedRounds=%d timedOut=%d discarded=%d seconds=%.1f]",
                tried.get(), improvedBoards.get(), conflictsRemoved.get(), roundsRun.get(), plateauMoves.get(),
                cappedRounds.get(), timedOut.get(), discarded.get(), nanosSpent.get() / 1e9);
    }

    private static final int CENTRE_CELL = 135;
    private static final int MIN_CELLS = 4, MAX_CELLS = 8;
    private static final int NODE_CAP = 20_000;      // per exact re-placement; hit in well under 1% of the rounds
    private static final int SOLUTION_CAP = 64;      // equal-cost arrangements kept per round (reservoir sample)
    private static final int CHECK_TIME_EVERY = 32;  // rounds
    // Region kinds: 0 cluster around a conflict, 1 conflicted cell + scattered partners, 2 two conflict sites, 3 cluster anywhere.
    private static final double[] REGION_WEIGHT = {4, 2, 1, 3};
    private static final int[] NB = new int[256 * 4]; // neighbour per side N, E, S, W; -1 outside the board

    static {
        for (int i = 0; i < 256; i++) {
            int r = i / 16, c = i % 16;
            NB[i * 4] = r > 0 ? i - 16 : -1;
            NB[i * 4 + 1] = c < 15 ? i + 1 : -1;
            NB[i * 4 + 2] = r < 15 ? i + 16 : -1;
            NB[i * 4 + 3] = c > 0 ? i - 1 : -1;
        }
    }

    // ------------------------------------------------------------------ public entry point

    /**
     * @param board        complete 256-cell board (not modified)
     * @param pinned       cells that must not move (clue pieces), or null; the centre cell never moves either way
     * @param hintsEnabled picks the conflict ceiling above which the search is skipped (14 without hints, 19 with)
     * @return a board with strictly fewer conflicts using the same pieces, or null if none was found (or disabled)
     */
    public static int[] improve(int[] board, boolean[] pinned, boolean hintsEnabled) {
        if (!enabled || board == null || board.length != 256) return null;
        int before = TailOptimizer.countConflicts(board); // -1 when a cell is empty
        if (before <= 0 || before > (hintsEnabled ? maxConflictsWithHints : maxConflicts)) return null;
        return improve(board, pinned, budgetFor(before), maxMillis, seedOf(board));
    }

    /** Rounds a board with this many conflicts gets: the larger budget at the deep threshold or below. */
    static int budgetFor(int conflicts) {
        return conflicts <= deepAtOrBelow ? deepRounds : rounds;
    }

    /** As above with an explicit budget and random seed (tests, offline experiments); no ceiling check. */
    static int[] improve(int[] board, boolean[] pinned, int budgetRounds, long millis, long seed) {
        int before = TailOptimizer.countConflicts(board);
        if (before <= 0) return null;
        boolean[] locked = pinned == null ? new boolean[256] : Arrays.copyOf(pinned, 256);
        locked[CENTRE_CELL] = true;
        int free = 0;
        for (boolean l : locked) if (!l) free++;
        if (free < MAX_CELLS) return null;                // a region needs MAX_CELLS cells that may move

        long start = System.nanoTime();
        Search search = new Search(board, locked, seed);
        search.run(budgetRounds, start + millis * 1_000_000L);
        tried.incrementAndGet();
        roundsRun.addAndGet(search.roundsRun);
        plateauMoves.addAndGet(search.plateauMoves);
        cappedRounds.addAndGet(search.cappedRounds);
        if (search.timedOut) timedOut.incrementAndGet();
        nanosSpent.addAndGet(System.nanoTime() - start);
        if (search.bestCost >= before) return null;

        int[] best = search.best;
        if (TailOptimizer.countConflicts(best) != search.bestCost || !samePieces(board, best) || movedLocked(board, best, locked)) {
            System.err.println("PlateauLns: discarded an inconsistent result (expected " + search.bestCost + " conflicts)");
            discarded.incrementAndGet();
            return null;
        }
        improvedBoards.incrementAndGet();
        conflictsRemoved.addAndGet(before - search.bestCost);
        return best;
    }

    /**
     * Test hook: the best way to put the pieces now in {@code cells} back into those cells, searched by the same exact
     * solver the rounds use.
     *
     * @return {conflicts of the whole board after the best rearrangement, number of different arrangements that reach it
     *         (the current one included when it is already the best)}
     */
    static int[] bestRearrangement(int[] board, int[] cells) {
        Solver solver = new Solver();
        solver.prepare(board, cells, cells.length, new Random(1));
        int before = solver.localCost();
        solver.solve(before, Integer.MAX_VALUE);
        int total = TailOptimizer.countConflicts(board) - before + solver.solutionCost[0];
        solver.release();
        return new int[]{total, solver.seen};
    }

    private static long seedOf(int[] board) {
        long h = 0xcbf29ce484222325L;
        for (int v : board) h = (h ^ v) * 0x100000001b3L;
        return h;
    }

    // ------------------------------------------------------------------ the search

    private static final class Search {
        final int[] b;
        final boolean[] locked;
        final Random rnd;
        final Solver solver = new Solver();
        final int[] region = new int[MAX_CELLS];
        final int[] conflicted = new int[256];
        final boolean[] inRegion = new boolean[256];
        final int[] otherArrangements = new int[SOLUTION_CAP];
        int cost;
        int[] best;
        int bestCost;
        int roundsRun, plateauMoves, cappedRounds;
        boolean timedOut;

        Search(int[] start, boolean[] locked, long seed) {
            this.b = start.clone();
            this.locked = locked;
            this.rnd = new Random(seed);
            this.cost = TailOptimizer.countConflicts(b);
            this.best = b.clone();
            this.bestCost = cost;
        }

        void run(int budgetRounds, long deadline) {
            double weightSum = REGION_WEIGHT[0] + REGION_WEIGHT[1] + REGION_WEIGHT[2] + REGION_WEIGHT[3];
            for (int round = 1; round <= budgetRounds && bestCost > 0; round++) {
                if (round % CHECK_TIME_EVERY == 0 && System.nanoTime() > deadline) {
                    timedOut = true;
                    break;
                }
                roundsRun = round;
                int conflictedCount = listConflictedCells();
                double x = rnd.nextDouble() * weightSum;
                int kind = x < REGION_WEIGHT[0] ? 0 : x < REGION_WEIGHT[0] + REGION_WEIGHT[1] ? 1
                        : x < REGION_WEIGHT[0] + REGION_WEIGHT[1] + REGION_WEIGHT[2] ? 2 : 3;
                if (conflictedCount == 0) kind = 3;
                int k = MIN_CELLS + rnd.nextInt(MAX_CELLS - MIN_CELLS + 1);
                int n = pickRegion(kind, k, conflictedCount);

                solver.prepare(b, region, n, rnd);
                int before = solver.localCost();
                solver.solve(before, NODE_CAP);
                if (solver.capped) cappedRounds++;
                int chosen = -1;
                if (solver.minCost < before) {
                    chosen = rnd.nextInt(solver.solutions);                    // strictly better: every kept solution has the minimum
                } else {
                    int others = 0;
                    for (int s = 0; s < solver.solutions; s++) if (!solver.isCurrentArrangement(s)) otherArrangements[others++] = s;
                    if (others > 0) {
                        chosen = otherArrangements[rnd.nextInt(others)];       // equal cost, different arrangement: plateau move
                        plateauMoves++;
                    }
                }
                if (chosen >= 0) {
                    for (int i = 0; i < n; i++) b[solver.cells[i]] = solver.solutionValue[chosen][i];
                    cost += solver.solutionCost[chosen] - before;
                    if (cost < bestCost) {
                        bestCost = cost;
                        best = b.clone();
                    }
                }
                solver.release();
            }
        }

        /** Cells (not locked) with a mismatching or non-grey-border side; returns how many. */
        private int listConflictedCells() {
            int count = 0;
            for (int i = 0; i < 256; i++) {
                if (locked[i]) continue;
                int p = b[i];
                for (int d = 0; d < 4; d++) {
                    int nb = NB[i * 4 + d];
                    if (nb < 0 ? side(p, d) != PieceUtils.BORDER_COLOR : side(p, d) != side(b[nb], (d + 2) & 3)) {
                        conflicted[count++] = i;
                        break;
                    }
                }
            }
            return count;
        }

        /** Fills region[] with k distinct unlocked cells and returns k. */
        private int pickRegion(int kind, int k, int conflictedCount) {
            Arrays.fill(inRegion, false);
            int n = 0;
            int first = kind == 3 || conflictedCount == 0 ? randomFreeCell() : conflicted[rnd.nextInt(conflictedCount)];
            region[n++] = first;
            inRegion[first] = true;
            if (kind == 1) {                                  // one conflicted cell and scattered partners
                while (n < k) {
                    int c = randomFreeCell();
                    if (!inRegion[c]) {
                        inRegion[c] = true;
                        region[n++] = c;
                    }
                }
                return n;
            }
            if (kind == 2 && conflictedCount > 0) {           // a second conflict site
                int second = conflicted[rnd.nextInt(conflictedCount)];
                if (!inRegion[second]) {
                    inRegion[second] = true;
                    region[n++] = second;
                }
            }
            for (int guard = 0; n < k && guard < 200; guard++) { // grow connected blobs around the seeds
                int base = region[rnd.nextInt(n)];
                int nb = NB[base * 4 + rnd.nextInt(4)];
                if (nb < 0 || locked[nb] || inRegion[nb]) continue;
                inRegion[nb] = true;
                region[n++] = nb;
            }
            while (n < k) {
                int c = randomFreeCell();
                if (!inRegion[c]) {
                    inRegion[c] = true;
                    region[n++] = c;
                }
            }
            return n;
        }

        private int randomFreeCell() {
            int c;
            do {
                c = rnd.nextInt(256);
            } while (locked[c]);
            return c;
        }
    }

    // ------------------------------------------------------------------ exact re-placement of k cells

    /**
     * Finds every arrangement of the pieces now in {@code cells} over those cells (any rotation) whose conflicts, counted
     * on all edges touching the cells, are the fewest, bounded by the cost of the current arrangement. Depth first, most
     * constrained cell first; at most {@link #SOLUTION_CAP} of the optimal arrangements are kept (reservoir sample).
     */
    private static final class Solver {
        final int[] cells = new int[MAX_CELLS];
        final int[] posOf = new int[256];
        final int[][] rotations = new int[MAX_CELLS][4];
        final int[] rotationCount = new int[MAX_CELLS];
        final int[] order = new int[MAX_CELLS];
        final int[] placedValue = new int[MAX_CELLS];
        final boolean[] placed = new boolean[MAX_CELLS];
        final int[] current = new int[MAX_CELLS];
        final int[][] solutionValue = new int[SOLUTION_CAP][MAX_CELLS];
        final int[] solutionCost = new int[SOLUTION_CAP];
        int[] board;
        int k, usedMask, bound, nodes, nodeCap, solutions, seen, minCost;
        boolean capped;
        Random rnd;

        Solver() {
            Arrays.fill(posOf, -1);
        }

        void prepare(int[] b, int[] cs, int count, Random random) {
            board = b;
            k = count;
            rnd = random;
            for (int i = 0; i < k; i++) {
                cells[i] = cs[i];
                posOf[cs[i]] = i;
                current[i] = b[cs[i]];
                int p = b[cs[i]], n = 0;
                for (int t = 0; t < 4; t++) {                 // distinct rotations only (a symmetric piece has fewer)
                    boolean seenBefore = false;
                    for (int u = 0; u < n; u++) if (rotations[i][u] == p) seenBefore = true;
                    if (!seenBefore) rotations[i][n++] = p;
                    p = PieceUtils.rotate(p);
                }
                rotationCount[i] = n;
            }
            boolean[] chosen = new boolean[k];                // most constrained cell first, then its neighbours; random tie-break
            for (int depth = 0; depth < k; depth++) {
                int pick = -1, pickScore = -1;
                for (int pos = 0; pos < k; pos++) {
                    if (chosen[pos]) continue;
                    int score = 0;
                    for (int d = 0; d < 4; d++) {
                        int nb = NB[cells[pos] * 4 + d];
                        if (nb < 0) score += 3;
                        else if (posOf[nb] < 0) score += 2;
                        else if (chosen[posOf[nb]]) score += 4;
                    }
                    score = score * 8 + rnd.nextInt(8);
                    if (score > pickScore) {
                        pickScore = score;
                        pick = pos;
                    }
                }
                order[depth] = pick;
                chosen[pick] = true;
            }
        }

        void release() {
            for (int i = 0; i < k; i++) posOf[cells[i]] = -1;
        }

        /** Conflicts on all edges touching the cells as they are now (an edge between two of them counts once). */
        int localCost() {
            int m = 0;
            for (int i = 0; i < k; i++) {
                int c = cells[i], p = board[c];
                for (int d = 0; d < 4; d++) {
                    int nb = NB[c * 4 + d];
                    if (nb < 0) {
                        if (side(p, d) != PieceUtils.BORDER_COLOR) m++;
                    } else {
                        int q = posOf[nb];
                        if (q >= 0 && q < i) continue;
                        if (side(p, d) != side(board[nb], (d + 2) & 3)) m++;
                    }
                }
            }
            return m;
        }

        void solve(int startBound, int cap) {
            bound = startBound;
            nodeCap = cap;
            nodes = 0;
            capped = false;
            usedMask = 0;
            solutions = 0;
            seen = 0;
            minCost = Integer.MAX_VALUE;
            Arrays.fill(placed, false);
            dfs(0, 0);
        }

        private int addedCost(int pos, int v) {
            int c = cells[pos], add = 0;
            for (int d = 0; d < 4; d++) {
                int nb = NB[c * 4 + d];
                int s = side(v, d);
                if (nb < 0) {
                    if (s != PieceUtils.BORDER_COLOR) add++;
                } else {
                    int q = posOf[nb];
                    if (q < 0) {
                        if (s != side(board[nb], (d + 2) & 3)) add++;
                    } else if (placed[q] && s != side(placedValue[q], (d + 2) & 3)) {
                        add++;
                    }
                }
            }
            return add;
        }

        private void dfs(int depth, int cost) {
            if (++nodes > nodeCap) {
                capped = true;
                return;
            }
            if (depth == k) {
                record(cost);
                return;
            }
            int pos = order[depth];
            for (int piece = 0; piece < k; piece++) {
                if ((usedMask >> piece & 1) != 0) continue;
                usedMask |= 1 << piece;
                for (int r = 0; r < rotationCount[piece]; r++) {
                    int v = rotations[piece][r];
                    int add = addedCost(pos, v);
                    if (cost + add > bound) continue;
                    placedValue[pos] = v;
                    placed[pos] = true;
                    dfs(depth + 1, cost + add);
                    placed[pos] = false;
                    if (capped) {
                        usedMask &= ~(1 << piece);
                        return;
                    }
                }
                usedMask &= ~(1 << piece);
            }
        }

        private void record(int cost) {
            if (cost < minCost) minCost = cost;
            if (cost < bound) {                               // a better arrangement: forget the equal-cost ones
                bound = cost;
                solutions = 0;
                seen = 0;
            }
            seen++;
            int slot;
            if (solutions < SOLUTION_CAP) {
                slot = solutions++;
            } else {
                slot = rnd.nextInt(seen);
                if (slot >= SOLUTION_CAP) return;
            }
            System.arraycopy(placedValue, 0, solutionValue[slot], 0, k);
            solutionCost[slot] = cost;
        }

        boolean isCurrentArrangement(int solution) {
            for (int i = 0; i < k; i++) if (solutionValue[solution][i] != current[i]) return false;
            return true;
        }
    }

    // ------------------------------------------------------------------ helpers

    private static int side(int piece, int direction) {       // direction 0 = north, 1 = east, 2 = south, 3 = west
        return (piece >>> (24 - 8 * direction)) & 0xFF;
    }

    /** Every piece must still be there exactly once (compared up to rotation). */
    private static boolean samePieces(int[] a, int[] b) {
        int[] x = new int[256], y = new int[256];
        for (int i = 0; i < 256; i++) {
            x[i] = canonical(a[i]);
            y[i] = canonical(b[i]);
        }
        Arrays.sort(x);
        Arrays.sort(y);
        return Arrays.equals(x, y);
    }

    private static boolean movedLocked(int[] a, int[] b, boolean[] locked) {
        for (int i = 0; i < 256; i++) {
            if (locked[i] && a[i] != b[i]) return true;
        }
        return false;
    }

    private static int canonical(int p) {
        int best = p, q = p;
        for (int i = 0; i < 3; i++) {
            q = PieceUtils.rotate(q);
            if (q < best) best = q;
        }
        return best;
    }
}

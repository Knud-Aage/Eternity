package dk.puzzle.tools;

import dk.puzzle.util.PieceUtils;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Exact re-arrangement of the last cells of the Blackwood fill order.
 *
 * <p>The score gap between our 13s and 17s lives almost entirely in the last ~12 cells placed, and neither the search
 * (which has no way back once it commits) nor {@link HoleSolver}'s region solver (zero-conflict rearrangements around
 * each mismatch only) looks for the arrangement of a whole tail block with the FEWEST mismatches. This does: it frees
 * the last K cells of the fill order and runs a branch and bound over every placement and rotation of exactly those
 * pieces, pruned by a colour-balance lower bound, for an arrangement strictly better than the one already there.
 * Rungs (default K=15 then 20) are tried in turn, each under a node cap, all under a wall-clock cap.</p>
 *
 * <p>Board layout is HoleSolver's: 256 packed pieces, row-major from the top row. The fill-order tail is the top-right
 * block (steps 236..255 are the last four columns of rows 0..4; earlier steps run down the columns to their left).</p>
 */
public final class TailOptimizer {

    private static volatile boolean enabled = !"false".equalsIgnoreCase(System.getenv("ETERNITY_TAIL_OPT"));
    private static volatile int[] rungs = {15, 20};
    private static volatile long[] rungNodeCaps = {2_000_000L, 10_000_000L};
    private static volatile long maxMillis = 3_000L;
    private static volatile int maxConflicts = 20;

    private static final AtomicLong tried = new AtomicLong(), improvedBoards = new AtomicLong(),
            conflictsRemoved = new AtomicLong(), cappedRungs = new AtomicLong(), nanosSpent = new AtomicLong();

    static {
        try {
            int[] r = parseInts(System.getenv("ETERNITY_TAIL_OPT_RUNGS"));
            long[] c = parseLongs(System.getenv("ETERNITY_TAIL_OPT_NODES"));
            if (r != null && r.length > 0) rungs = r;
            if (c != null && c.length > 0) rungNodeCaps = c;
            String ms = System.getenv("ETERNITY_TAIL_OPT_MILLIS");
            if (ms != null && !ms.isBlank()) maxMillis = Long.parseLong(ms.trim());
            String mc = System.getenv("ETERNITY_TAIL_OPT_MAX_CONFLICTS");
            if (mc != null && !mc.isBlank()) maxConflicts = Integer.parseInt(mc.trim());
        } catch (NumberFormatException e) {
            System.err.println("TailOptimizer: ignoring malformed ETERNITY_TAIL_OPT_* setting (" + e.getMessage() + ")");
        }
    }

    private static int[] parseInts(String s) {
        if (s == null || s.isBlank()) return null;
        return Arrays.stream(s.split(",")).map(String::trim).mapToInt(Integer::parseInt).toArray();
    }

    private static long[] parseLongs(String s) {
        if (s == null || s.isBlank()) return null;
        return Arrays.stream(s.split(",")).map(String::trim).mapToLong(Long::parseLong).toArray();
    }

    private TailOptimizer() {
    }

    /** Test/runtime hook: replaces the configuration; caps are matched to rungs by position, the last cap repeats. */
    static void configure(boolean on, int[] newRungs, long[] newCaps, long newMaxMillis, int newMaxConflicts) {
        enabled = on;
        rungs = newRungs.clone();
        rungNodeCaps = newCaps.clone();
        maxMillis = newMaxMillis;
        maxConflicts = newMaxConflicts;
    }

    public static String stats() {
        return String.format("tail[tried=%d improved=%d conflictsRemoved=%d cappedRungs=%d seconds=%.1f]",
                tried.get(), improvedBoards.get(), conflictsRemoved.get(), cappedRungs.get(), nanosSpent.get() / 1e9);
    }

    // ------------------------------------------------------------------ fill order (display orientation, row 0 = top)

    private static final int[] CELL_OF_STEP = new int[256];

    static {
        int[][] tail = {{243, 249, 254, 255}, {242, 248, 252, 253}, {241, 247, 250, 251}, {240, 244, 245, 246}, {236, 237, 238, 239}};
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                int s;
                if (r >= 5) s = 16 * (15 - r) + c;
                else if (c <= 4) s = 176 + 5 * (4 - r) + c;
                else if (c <= 11) s = 201 + 5 * (c - 5) + (4 - r);
                else s = tail[r][c - 12];
                CELL_OF_STEP[s] = r * 16 + c;
            }
        }
    }

    /** Board index of a fill-order step for a search run under {@code searchRotationDegrees} (boards arrive un-rotated). */
    static int cellOfStep(int step, int searchRotationDegrees) {
        int cell = CELL_OF_STEP[step];
        int turns = (((360 - searchRotationDegrees) % 360) / 90 + 4) % 4;
        int r = cell / 16, c = cell % 16;
        for (int t = 0; t < turns; t++) {
            int nr = c, nc = 15 - r;
            r = nr;
            c = nc;
        }
        return r * 16 + c;
    }

    // ------------------------------------------------------------------ conflicts (same metric as HoleSolver.findConflicts)

    /** Internal mismatches plus non-grey border edges; -1 if the board has an empty cell. */
    static int countConflicts(int[] board) {
        int m = 0;
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                int p = board[r * 16 + c];
                if (p == -1) return -1;
                if (r == 0 && PieceUtils.getNorth(p) != PieceUtils.BORDER_COLOR) m++;
                if (r == 15 && PieceUtils.getSouth(p) != PieceUtils.BORDER_COLOR) m++;
                if (c == 0 && PieceUtils.getWest(p) != PieceUtils.BORDER_COLOR) m++;
                if (c == 15 && PieceUtils.getEast(p) != PieceUtils.BORDER_COLOR) m++;
                if (c < 15) {
                    int q = board[r * 16 + c + 1];
                    if (q == -1) return -1;
                    if (PieceUtils.getEast(p) != PieceUtils.getWest(q)) m++;
                }
                if (r < 15) {
                    int q = board[(r + 1) * 16 + c];
                    if (q == -1) return -1;
                    if (PieceUtils.getSouth(p) != PieceUtils.getNorth(q)) m++;
                }
            }
        }
        return m;
    }

    // ------------------------------------------------------------------ public entry point

    /**
     * @param board                complete 256-cell board (not modified)
     * @param pinned               cells that must not move (clue pieces), or null
     * @param searchRotationDegrees BlackwoodSolver.ROTATE_INSTANCE_DEGREES the board was found under (0 in no-hints mode)
     * @return a board with strictly fewer conflicts using the same pieces, or null if none was found (or disabled)
     */
    public static int[] improve(int[] board, boolean[] pinned, int searchRotationDegrees) {
        if (!enabled || board == null || board.length != 256) return null;
        int before = countConflicts(board);
        if (before <= 0 || before > maxConflicts) return null;

        long start = System.nanoTime();
        long deadline = start + maxMillis * 1_000_000L;
        int[] current = board;
        int currentConflicts = before;
        for (int i = 0; i < rungs.length && currentConflicts > 0 && System.nanoTime() < deadline; i++) {
            long cap = rungNodeCaps[Math.min(i, rungNodeCaps.length - 1)];
            Outcome o = runRung(current, pinned, rungs[i], cap, deadline, searchRotationDegrees);
            if (o.capped) cappedRungs.incrementAndGet();
            if (o.board != null) {
                current = o.board;
                currentConflicts = o.conflicts;
            }
        }
        tried.incrementAndGet();
        nanosSpent.addAndGet(System.nanoTime() - start);
        if (currentConflicts < before) {
            improvedBoards.incrementAndGet();
            conflictsRemoved.addAndGet(before - currentConflicts);
            return current;
        }
        return null;
    }

    private record Outcome(int[] board, int conflicts, boolean capped) {
    }

    private static Outcome runRung(int[] board, boolean[] pinned, int k, long nodeCap, long deadline, int rotation) {
        int[] cells = new int[k];
        int n = 0;
        for (int s = 256 - k; s < 256; s++) {
            int cell = cellOfStep(s, rotation);
            if (pinned != null && pinned[cell]) continue;
            if (board[cell] == -1) return new Outcome(null, 0, false);
            cells[n++] = cell;
        }
        cells = Arrays.copyOf(cells, n);
        Solver solver = new Solver(board, cells, nodeCap, deadline);
        int cNow = solver.identityCost();
        int before = countConflicts(board);
        solver.search(cNow);
        if (solver.best >= cNow) return new Outcome(null, before, solver.stop);

        int[] result = board.clone();
        for (int i = 0; i < n; i++) result[cells[i]] = solver.packed[solver.bestTile[i]][solver.bestRot[i]];
        int expected = before - cNow + solver.best;
        if (countConflicts(result) != expected || !samePieces(board, result, cells)) {
            System.err.println("TailOptimizer: discarded an inconsistent result (expected " + expected + " conflicts)");
            return new Outcome(null, before, solver.stop);
        }
        return new Outcome(result, expected, solver.stop);
    }

    /** Every freed piece must still be there exactly once (compared up to rotation). */
    private static boolean samePieces(int[] a, int[] b, int[] cells) {
        int[] x = new int[cells.length], y = new int[cells.length];
        for (int i = 0; i < cells.length; i++) {
            x[i] = canonical(a[cells[i]]);
            y[i] = canonical(b[cells[i]]);
        }
        Arrays.sort(x);
        Arrays.sort(y);
        return Arrays.equals(x, y);
    }

    private static int canonical(int p) {
        int best = p, q = p;
        for (int i = 0; i < 3; i++) {
            q = PieceUtils.rotate(q);
            if (q < best) best = q;
        }
        return best;
    }

    // ------------------------------------------------------------------ the search

    private static final int OFF = -1, FIXED = 0, EARLIER = 1, LATER = 2;

    private static final class Solver {
        final int k;
        final int[] cell;
        final int[][] nbType, nbFreed, fixedToward;   // per freed cell and direction (0=N 1=E 2=S 3=W); colours are compressed
        final int[][][] rot;                          // [tile][rotation][direction] compressed colours
        final int[][] packed;                         // [tile][rotation] packed piece
        final boolean[][] rotValid;
        final boolean[] used;
        final int[][] placed;
        final int[] supply, demand;
        final int colours;
        int border;
        long nodes;
        final long cap, deadline;
        boolean stop;
        int best;
        final int[] bestTile, bestRot, curTile, curRot;
        final int[][] candId, candCost, reqBuf;

        Solver(int[] board, int[] cells, long cap, long deadline) {
            this.k = cells.length;
            this.cell = cells;
            this.cap = cap;
            this.deadline = deadline;
            int[] freedIdx = new int[256];
            Arrays.fill(freedIdx, -1);
            for (int i = 0; i < k; i++) freedIdx[cell[i]] = i;

            int[] colourIdx = new int[256];
            Arrays.fill(colourIdx, -1);
            colourIdx[PieceUtils.BORDER_COLOR] = 0;
            int nc = 1;
            nbType = new int[k][4];
            nbFreed = new int[k][4];
            fixedToward = new int[k][4];
            rot = new int[k][4][4];
            packed = new int[k][4];
            rotValid = new boolean[k][4];
            int[][] base = new int[k][4];
            for (int j = 0; j < k; j++) {
                int p = board[cell[j]];
                base[j][0] = PieceUtils.getNorth(p);
                base[j][1] = PieceUtils.getEast(p);
                base[j][2] = PieceUtils.getSouth(p);
                base[j][3] = PieceUtils.getWest(p);
                for (int d = 0; d < 4; d++) {
                    if (colourIdx[base[j][d]] < 0) colourIdx[base[j][d]] = nc++;
                }
            }
            for (int i = 0; i < k; i++) {
                int idx = cell[i], r = idx / 16, c = idx % 16;
                int[] nb = {r > 0 ? idx - 16 : -1, c < 15 ? idx + 1 : -1, r < 15 ? idx + 16 : -1, c > 0 ? idx - 1 : -1};
                for (int d = 0; d < 4; d++) {
                    if (nb[d] < 0) {
                        nbType[i][d] = OFF;
                    } else if (freedIdx[nb[d]] < 0) {
                        nbType[i][d] = FIXED;
                        int q = board[nb[d]];
                        int col = switch ((d + 2) & 3) {
                            case 0 -> PieceUtils.getNorth(q);
                            case 1 -> PieceUtils.getEast(q);
                            case 2 -> PieceUtils.getSouth(q);
                            default -> PieceUtils.getWest(q);
                        };
                        if (colourIdx[col] < 0) colourIdx[col] = nc++;
                        fixedToward[i][d] = colourIdx[col];
                    } else {
                        nbFreed[i][d] = freedIdx[nb[d]];
                        nbType[i][d] = nbFreed[i][d] < i ? EARLIER : LATER;
                    }
                }
            }
            colours = nc;
            for (int j = 0; j < k; j++) {
                for (int r = 0; r < 4; r++) {
                    int[] e = new int[4];
                    for (int d = 0; d < 4; d++) e[d] = base[j][(d - r + 4) & 3];
                    packed[j][r] = PieceUtils.pack(e[0], e[1], e[2], e[3]);
                    for (int d = 0; d < 4; d++) rot[j][r][d] = colourIdx[e[d]];
                    boolean dup = false;
                    for (int q = 0; q < r; q++) if (Arrays.equals(rot[j][q], rot[j][r])) dup = true;
                    rotValid[j][r] = !dup;
                }
            }
            used = new boolean[k];
            placed = new int[k][4];
            supply = new int[nc];
            demand = new int[nc];
            bestTile = new int[k];
            bestRot = new int[k];
            curTile = new int[k];
            curRot = new int[k];
            candId = new int[k][4 * k];
            candCost = new int[k][4 * k];
            reqBuf = new int[k][4];
        }

        /** Mismatches touching the freed block in its current arrangement. */
        int identityCost() {
            int cost = 0;
            for (int i = 0; i < k; i++) {
                for (int d = 0; d < 4; d++) {
                    int req = requirement(i, d);
                    if (req >= 0 && rot[i][0][d] != req) cost++;
                }
                System.arraycopy(rot[i][0], 0, placed[i], 0, 4);
            }
            return cost;
        }

        int requirement(int i, int d) {
            return switch (nbType[i][d]) {
                case OFF -> 0;
                case FIXED -> fixedToward[i][d];
                case EARLIER -> placed[nbFreed[i][d]][(d + 2) & 3];
                default -> -1;
            };
        }

        void search(int cNow) {
            best = cNow;
            if (cNow == 0) return;
            Arrays.fill(supply, 0);
            Arrays.fill(demand, 0);
            border = 0;
            for (int j = 0; j < k; j++) for (int d = 0; d < 4; d++) supply[rot[j][0][d]]++;
            for (int i = 0; i < k; i++) {
                for (int d = 0; d < 4; d++) {
                    if (nbType[i][d] == OFF) border++;
                    else if (nbType[i][d] == FIXED) demand[fixedToward[i][d]]++;
                }
            }
            if (lowerBound() >= cNow) return;
            for (int i = 0; i < k; i++) Arrays.fill(placed[i], 0);
            dfs(0, 0);
        }

        /** Admissible: every colour must pair up, and each mismatch fixes the parity of at most two colours. */
        int lowerBound() {
            int odd = 0, deficit = 0;
            for (int c = 0; c < colours; c++) {
                int e = supply[c] - demand[c] - (c == 0 ? border : 0);
                if ((e & 1) != 0) odd++;
                if (e < 0) deficit -= e;
            }
            return Math.max(odd / 2, deficit);
        }

        void apply(int i, int j, int r) {
            int[] t = rot[j][r];
            used[j] = true;
            System.arraycopy(t, 0, placed[i], 0, 4);
            for (int d = 0; d < 4; d++) {
                supply[t[d]]--;
                switch (nbType[i][d]) {
                    case OFF -> border--;
                    case FIXED -> demand[fixedToward[i][d]]--;
                    case EARLIER -> demand[placed[nbFreed[i][d]][(d + 2) & 3]]--;
                    default -> demand[t[d]]++;
                }
            }
        }

        void undo(int i, int j, int r) {
            int[] t = rot[j][r];
            used[j] = false;
            for (int d = 0; d < 4; d++) {
                supply[t[d]]++;
                switch (nbType[i][d]) {
                    case OFF -> border++;
                    case FIXED -> demand[fixedToward[i][d]]++;
                    case EARLIER -> demand[placed[nbFreed[i][d]][(d + 2) & 3]]++;
                    default -> demand[t[d]]--;
                }
            }
        }

        void dfs(int i, int acc) {
            int[] ids = candId[i], costs = candCost[i], req = reqBuf[i];
            for (int d = 0; d < 4; d++) req[d] = requirement(i, d);
            int budget = best - 1, n = 0;
            for (int j = 0; j < k; j++) {
                if (used[j]) continue;
                for (int r = 0; r < 4; r++) {
                    if (!rotValid[j][r]) continue;
                    int[] t = rot[j][r];
                    int c = 0;
                    for (int d = 0; d < 4; d++) if (req[d] >= 0 && t[d] != req[d]) c++;
                    if (acc + c > budget) continue;
                    ids[n] = j * 4 + r;
                    costs[n] = c;
                    n++;
                }
            }
            for (int level = 0; level <= 4; level++) {
                for (int q = 0; q < n; q++) {
                    if (costs[q] != level) continue;
                    int newAcc = acc + level;
                    if (newAcc > best - 1) return;
                    int j = ids[q] >> 2, r = ids[q] & 3;
                    apply(i, j, r);
                    nodes++;
                    curTile[i] = j;
                    curRot[i] = r;
                    if (i + 1 == k) {
                        best = newAcc;
                        System.arraycopy(curTile, 0, bestTile, 0, k);
                        System.arraycopy(curRot, 0, bestRot, 0, k);
                    } else if (newAcc + lowerBound() <= best - 1) {
                        dfs(i + 1, newAcc);
                    }
                    undo(i, j, r);
                    if ((nodes & 1023) == 0 && (nodes > cap || System.nanoTime() > deadline)) stop = true;
                    if (stop) return;
                }
            }
        }
    }
}

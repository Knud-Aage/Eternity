package dk.puzzle.tools;

import dk.puzzle.util.PieceUtils;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Exhaustive "change two pieces" polish for a complete board.
 *
 * <p>{@code ConflictReducer}'s polish turns one piece at a time and swaps two pieces AS STORED, keeping a swap only if
 * it lowers the count on the spot. It never turns a piece in its new cell after a swap and it cannot make a neutral
 * move that sets up an improving one. A person fixing a board by hand does both. Run offline over 408 saved
 * 13-conflict boards, the missing moves turned three of them into valid new 12s (Oct 16).</p>
 *
 * <p>The neighbourhood is every single rotation, every swap of two cells with each of the 4 x 4 rotations in the new
 * cells, and every pair of pieces turned in place: 810,390 moves per board, each scored from the edges around the one
 * or two cells it touches. If a move lowers the count the best one is taken and the search restarts from the improved
 * board. If none does, the moves that keep the count are explored breadth-first, up to {@code neutralDepth} of them in a
 * row, looking for an improving move from each board reached. Only moves that touch a mismatching edge are
 * considered: a move between two conflict-free cells cannot lower the count.</p>
 *
 * <p>Board layout is HoleSolver's: 256 packed pieces, row-major from the top row. The centre cell never moves, and
 * neither does any cell the caller pins (the clue pieces when hints are on). Worth running only on boards that are
 * already good (the keep thresholds are 13 conflicts without hints and 18 with): the neighbourhood is small there
 * (a handful of neutral moves), so the whole search takes a few hundredths of a second.</p>
 */
public final class PairMovePolish {

    private static volatile boolean enabled = !"false".equalsIgnoreCase(System.getenv("ETERNITY_PAIR_POLISH"));
    private static volatile int neutralDepth = 3;
    private static volatile int maxNodes = 400;
    private static volatile long maxMillis = 3_000L;
    // Keep thresholds are 13 conflicts without hints and 18 with. Without hints the ceiling sits one above: a 14 can become
    // a keepable 13 (10 of 300 real 14s did), while a 15 hardly ever reaches 13 (0 of 300) and an 11 is out of reach from
    // there. With hints it sits one above as well (18 + 1).
    private static volatile int maxConflicts = 14;
    private static volatile int maxConflictsWithHints = 19;

    private static final AtomicLong tried = new AtomicLong(), improvedBoards = new AtomicLong(),
            conflictsRemoved = new AtomicLong(), scans = new AtomicLong(), cappedSearches = new AtomicLong(),
            nanosSpent = new AtomicLong();

    static {
        try {
            neutralDepth = envInt("ETERNITY_PAIR_POLISH_DEPTH", neutralDepth);
            maxNodes = envInt("ETERNITY_PAIR_POLISH_NODES", maxNodes);
            maxMillis = envInt("ETERNITY_PAIR_POLISH_MILLIS", (int) maxMillis);
            maxConflicts = envInt("ETERNITY_PAIR_POLISH_MAX_CONFLICTS", maxConflicts);
            maxConflictsWithHints = envInt("ETERNITY_PAIR_POLISH_MAX_CONFLICTS_HINTS", maxConflictsWithHints);
        } catch (NumberFormatException e) {
            System.err.println("PairMovePolish: ignoring malformed ETERNITY_PAIR_POLISH_* setting (" + e.getMessage() + ")");
        }
    }

    private static int envInt(String name, int fallback) {
        String v = System.getenv(name);
        return v == null || v.isBlank() ? fallback : Integer.parseInt(v.trim());
    }

    private PairMovePolish() {
    }

    /** Test/runtime hook: replaces the configuration. */
    static void configure(boolean on, int depth, int nodes, long millis, int maxWithoutHints, int maxWithHints) {
        enabled = on;
        neutralDepth = depth;
        maxNodes = nodes;
        maxMillis = millis;
        maxConflicts = maxWithoutHints;
        maxConflictsWithHints = maxWithHints;
    }

    public static String stats() {
        return String.format("pair[tried=%d improved=%d conflictsRemoved=%d scans=%d cappedSearches=%d seconds=%.1f]",
                tried.get(), improvedBoards.get(), conflictsRemoved.get(), scans.get(), cappedSearches.get(),
                nanosSpent.get() / 1e9);
    }

    private static final int CENTRE_CELL = 135;
    private static final int NEUTRAL_CAP_PER_SCAN = 2_000;
    private static final int MAX_QUEUED_BOARDS = 5_000; // about 5 MB
    private static final int ROTATE = 0, SWAP = 1, TURN_BOTH = 2;

    // ------------------------------------------------------------------ public entry point

    /**
     * @param board        complete 256-cell board (not modified)
     * @param pinned       cells that must not move (clue pieces), or null; the centre cell never moves either way
     * @param hintsEnabled picks the conflict ceiling above which the polish is skipped (14 without hints, 19 with)
     * @return a board with strictly fewer conflicts using the same pieces, or null if none was found (or disabled)
     */
    public static int[] improve(int[] board, boolean[] pinned, boolean hintsEnabled) {
        if (!enabled || board == null || board.length != 256) return null;
        int before = TailOptimizer.countConflicts(board); // -1 when a cell is empty
        if (before <= 0 || before > (hintsEnabled ? maxConflictsWithHints : maxConflicts)) return null;

        boolean[] locked = pinned == null ? new boolean[256] : Arrays.copyOf(pinned, 256);
        locked[CENTRE_CELL] = true;

        long start = System.nanoTime();
        Budget budget = new Budget(maxNodes, start + maxMillis * 1_000_000L);
        int[] current = board.clone();
        int cost = before;
        while (cost > 0) {
            Improvement found = findImprovement(current, cost, locked, budget);
            if (found == null) break;
            current = found.board;
            cost = found.cost;
        }
        tried.incrementAndGet();
        scans.addAndGet(budget.nodes);
        if (budget.exhausted) cappedSearches.incrementAndGet();
        nanosSpent.addAndGet(System.nanoTime() - start);
        if (cost >= before) return null;

        if (TailOptimizer.countConflicts(current) != cost || !samePieces(board, current) || movedLocked(board, current, locked)) {
            System.err.println("PairMovePolish: discarded an inconsistent result (expected " + cost + " conflicts)");
            return null;
        }
        improvedBoards.incrementAndGet();
        conflictsRemoved.addAndGet(before - cost);
        return current;
    }

    // ------------------------------------------------------------------ search

    private record Improvement(int[] board, int cost) {
    }

    private record Node(int[] board, int depth) {
    }

    private static final class Budget {
        final int maxNodes;
        final long deadline;
        int nodes;
        boolean exhausted;

        Budget(int maxNodes, long deadline) {
            this.maxNodes = maxNodes;
            this.deadline = deadline;
        }

        boolean take() {
            if (nodes >= maxNodes || System.nanoTime() > deadline) {
                exhausted = true;
                return false;
            }
            nodes++;
            return true;
        }
    }

    /** The first board, reachable from {@code start} through at most {@code neutralDepth} neutral moves and one improving move, that has fewer conflicts. */
    private static Improvement findImprovement(int[] start, int cost, boolean[] locked, Budget budget) {
        ArrayDeque<Node> queue = new ArrayDeque<>();
        Set<Long> seen = new HashSet<>();
        queue.add(new Node(start, 0));
        seen.add(hash(start));
        while (!queue.isEmpty()) {
            if (!budget.take()) return null;
            Node node = queue.poll();
            Scan scan = scan(node.board, locked);
            if (scan.best != null) {
                int[] improved = node.board.clone();
                scan.best.apply(improved);
                return new Improvement(improved, cost + scan.bestDelta);
            }
            if (node.depth >= neutralDepth) continue;
            for (Move m : scan.neutral) {
                if (queue.size() >= MAX_QUEUED_BOARDS) break; // a board this flat is not going to be searched exhaustively anyway
                int[] child = node.board.clone();
                m.apply(child);
                if (seen.add(hash(child))) queue.add(new Node(child, node.depth + 1));
            }
        }
        return null;
    }

    private record Move(int kind, int i, int j, int x, int y) {
        /** ROTATE: turn cell i x quarter-turns. SWAP: cell i gets j's piece turned x, cell j gets i's piece turned y. TURN_BOTH: turn i by x and j by y. */
        void apply(int[] b) {
            switch (kind) {
                case ROTATE -> b[i] = turn(b[i], x);
                case SWAP -> {
                    int pi = b[i], pj = b[j];
                    b[i] = turn(pj, x);
                    b[j] = turn(pi, y);
                }
                default -> {
                    b[i] = turn(b[i], x);
                    b[j] = turn(b[j], y);
                }
            }
        }
    }

    private static final class Scan {
        int bestDelta;   // most negative change in the count seen so far; 0 while nothing improves
        Move best;
        final List<Move> neutral = new ArrayList<>();

        void note(int delta, int kind, int i, int j, int x, int y) {
            if (delta < bestDelta) {
                bestDelta = delta;
                best = new Move(kind, i, j, x, y);
            } else if (delta == 0 && neutral.size() < NEUTRAL_CAP_PER_SCAN) {
                neutral.add(new Move(kind, i, j, x, y));
            }
        }
    }

    /**
     * Scores every move that touches a mismatching edge and keeps the best improving one, plus every neutral one.
     * {@code b} is edited while a move is tried and always restored before this returns.
     */
    private static Scan scan(int[] b, boolean[] locked) {
        Scan scan = new Scan();
        int[] turned = new int[1024]; // [4 * cell + quarter-turns]
        boolean[] hot = new boolean[256];
        for (int c = 0; c < 256; c++) {
            int p = b[c];
            for (int k = 0; k < 4; k++) {
                turned[4 * c + k] = p;
                p = PieceUtils.rotate(p);
            }
            hot[c] = owned(b, c, -1) > 0;
        }
        for (int i = 0; i < 256; i++) {
            if (locked[i]) continue;
            if (hot[i]) {
                int before = owned(b, i, -1);
                for (int k = 1; k < 4; k++) {
                    b[i] = turned[4 * i + k];
                    int delta = owned(b, i, -1) - before;
                    if (delta <= 0) scan.note(delta, ROTATE, i, -1, k, 0);
                }
                b[i] = turned[4 * i];
            }
            for (int j = i + 1; j < 256; j++) {
                if (locked[j] || (!hot[i] && !hot[j])) continue;
                int before = owned(b, i, j) + owned(b, j, i);
                for (int x = 0; x < 4; x++) {
                    for (int y = 0; y < 4; y++) {
                        b[i] = turned[4 * j + x];
                        b[j] = turned[4 * i + y];
                        int delta = owned(b, i, j) + owned(b, j, i) - before;
                        if (delta <= 0) scan.note(delta, SWAP, i, j, x, y);
                    }
                }
                for (int x = 1; x < 4; x++) {
                    for (int y = 1; y < 4; y++) {
                        b[i] = turned[4 * i + x];
                        b[j] = turned[4 * j + y];
                        int delta = owned(b, i, j) + owned(b, j, i) - before;
                        if (delta <= 0) scan.note(delta, TURN_BOTH, i, j, x, y);
                    }
                }
                b[i] = turned[4 * i];
                b[j] = turned[4 * j];
            }
        }
        return scan;
    }

    /**
     * Mismatches on cell {@code x}'s east and south sides (a missing grey counts on the board edge) plus its west and
     * north sides, except the one that faces {@code skip}. For a move touching cells i &lt; j, owned(i, j) + owned(j, i)
     * counts the edge between two adjacent cells once, whichever way round they sit.
     */
    private static int owned(int[] b, int x, int skip) {
        int p = b[x], r = x >> 4, c = x & 15, m = 0;
        if (c < 15) {
            if (PieceUtils.getEast(p) != PieceUtils.getWest(b[x + 1])) m++;
        } else if (PieceUtils.getEast(p) != PieceUtils.BORDER_COLOR) m++;
        if (r < 15) {
            if (PieceUtils.getSouth(p) != PieceUtils.getNorth(b[x + 16])) m++;
        } else if (PieceUtils.getSouth(p) != PieceUtils.BORDER_COLOR) m++;
        if (c > 0) {
            if (x - 1 != skip && PieceUtils.getEast(b[x - 1]) != PieceUtils.getWest(p)) m++;
        } else if (PieceUtils.getWest(p) != PieceUtils.BORDER_COLOR) m++;
        if (r > 0) {
            if (x - 16 != skip && PieceUtils.getSouth(b[x - 16]) != PieceUtils.getNorth(p)) m++;
        } else if (PieceUtils.getNorth(p) != PieceUtils.BORDER_COLOR) m++;
        return m;
    }

    // ------------------------------------------------------------------ helpers

    private static int turn(int piece, int quarterTurns) {
        for (int k = 0; k < quarterTurns; k++) piece = PieceUtils.rotate(piece);
        return piece;
    }

    private static long hash(int[] b) {
        long h = 0xcbf29ce484222325L;
        for (int v : b) h = (h ^ v) * 0x100000001b3L;
        return h;
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

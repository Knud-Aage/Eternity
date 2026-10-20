package dk.puzzle.tools;

import dk.puzzle.util.PieceUtils;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PlateauLnsTest {

    private static final int CENTRE = 135;
    private static final int GREY = PieceUtils.BORDER_COLOR;

    /** name, conflicts before, conflicts the search reached, random seed, rounds, packed board. */
    private record Fixture(String name, int before, int reached, long seed, int rounds, int[] board) {
    }

    /** Saved boards from the live runs (a GPU 12, two 13s, two 14s) that the search improved offline on Oct 20 2026. */
    private static List<Fixture> fixtures() throws IOException {
        List<Fixture> out = new ArrayList<>();
        try (BufferedReader in = new BufferedReader(new InputStreamReader(
                PlateauLnsTest.class.getResourceAsStream("/plateau-lns-boards.tsv"), StandardCharsets.UTF_8))) {
            String line;
            while ((line = in.readLine()) != null) {
                String[] f = line.split("\t");
                out.add(new Fixture(f[0], Integer.parseInt(f[1]), Integer.parseInt(f[2]), Long.parseLong(f[3]), Integer.parseInt(f[4]), decode(f[5])));
            }
        }
        return out;
    }

    /** Bucas edge string (top,right,bottom,left per cell, row-major from the top, 'a' = grey) to HoleSolver's packed board. */
    private static int[] decode(String edges) {
        int[] board = new int[256];
        for (int i = 0; i < 256; i++) {
            int[] c = new int[4];
            for (int d = 0; d < 4; d++) c[d] = edges.charAt(4 * i + d) - 'a';
            board[i] = PieceUtils.pack(c[0], c[1], c[2], c[3]);
        }
        return board;
    }

    private static void defaults() {
        PlateauLns.configure(true, 20_000, 100_000, 12, 15_000L, 14, 19);
    }

    @BeforeEach
    void generousLimits() {
        PlateauLns.configure(true, 20_000, 100_000, 12, 120_000L, 14, 19);
    }

    @AfterEach
    void restoreDefaults() {
        defaults();
    }

    // ------------------------------------------------------------------ small helpers

    /** Random colours 1..22 on every inner edge and a grey rim: a board with no conflicts at all. */
    private static int[] conflictFreeBoard(Random rnd) {
        int[][] below = new int[16][16], right = new int[16][16];
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                below[r][c] = 1 + rnd.nextInt(22);
                right[r][c] = 1 + rnd.nextInt(22);
            }
        }
        int[] b = new int[256];
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                b[r * 16 + c] = PieceUtils.pack(r == 0 ? GREY : below[r - 1][c], c == 15 ? GREY : right[r][c],
                        r == 15 ? GREY : below[r][c], c == 0 ? GREY : right[r][c - 1]);
            }
        }
        return b;
    }

    /** Changes one inner edge colour of {@code count} random cells, which makes {@code count} mismatches. */
    private static int[] damage(int[] board, Random rnd, int count) {
        int[] b = board.clone();
        for (int done = 0; done < count; ) {
            int i = rnd.nextInt(256);
            int d = rnd.nextInt(4);
            int r = i / 16, c = i % 16;
            boolean inner = d == 0 ? r > 0 : d == 1 ? c < 15 : d == 2 ? r < 15 : c > 0;
            if (i == CENTRE || !inner) continue;
            int shift = 24 - 8 * d;
            int old = (b[i] >>> shift) & 0xFF;
            int colour;
            do {
                colour = 1 + rnd.nextInt(22);
            } while (colour == old);
            b[i] = (b[i] & ~(0xFF << shift)) | (colour << shift);
            done++;
        }
        return b;
    }

    private static int[] distinctCells(Random rnd, int k) {
        Set<Integer> chosen = new HashSet<>();
        while (chosen.size() < k) {
            int c = rnd.nextInt(256);
            if (c != CENTRE) chosen.add(c);
        }
        return chosen.stream().mapToInt(Integer::intValue).toArray();
    }

    private static int conflicts(int[] board) {
        return TailOptimizer.countConflicts(board);
    }

    private static int turn(int piece, int quarterTurns) {
        for (int k = 0; k < quarterTurns; k++) piece = PieceUtils.rotate(piece);
        return piece;
    }

    private static int canonical(int p) {
        int best = p, q = p;
        for (int i = 0; i < 3; i++) {
            q = PieceUtils.rotate(q);
            if (q < best) best = q;
        }
        return best;
    }

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

    /** Every assignment of the pieces in {@code cells} to those cells with every rotation: {fewest conflicts, number of arrangements reaching it}. */
    private static int[] bruteForce(int[] board, int[] cells) {
        int k = cells.length;
        int[] pieces = new int[k];
        for (int i = 0; i < k; i++) pieces[i] = board[cells[i]];
        int[] work = board.clone();
        int[] best = {Integer.MAX_VALUE, 0};
        Set<String> arrangements = new HashSet<>();
        permute(0, new boolean[k], pieces, cells, work, best, arrangements);
        return best;
    }

    private static void permute(int pos, boolean[] used, int[] pieces, int[] cells, int[] work, int[] best, Set<String> arrangements) {
        if (pos == cells.length) {
            int cost = conflicts(work);
            String key = Arrays.toString(Arrays.stream(cells).map(c -> work[c]).toArray());
            if (cost < best[0]) {
                best[0] = cost;
                best[1] = 0;
                arrangements.clear();
            }
            if (cost == best[0] && arrangements.add(key)) best[1]++;
            return;
        }
        for (int p = 0; p < pieces.length; p++) {
            if (used[p]) continue;
            used[p] = true;
            int v = pieces[p];
            for (int rot = 0; rot < 4; rot++) {
                work[cells[pos]] = v;
                permute(pos + 1, used, pieces, cells, work, best, arrangements);
                v = PieceUtils.rotate(v);
            }
            used[p] = false;
        }
    }

    // ------------------------------------------------------------------ tests

    @Test
    void theExactReplacementFindsTheFewestConflictsAndEveryTieAtThatCount() {
        Random rnd = new Random(5);
        int regions = 0, withTies = 0;
        for (int round = 0; round < 8; round++) {
            int[] board = damage(conflictFreeBoard(rnd), rnd, 60);
            for (int t = 0; t < 10; t++) {
                int[] cells = distinctCells(rnd, 2 + rnd.nextInt(3)); // 2..4 cells: brute force is k! * 4^k arrangements
                int[] fast = PlateauLns.bestRearrangement(board, cells);
                int[] slow = bruteForce(board, cells);
                assertEquals(slow[0], fast[0], "fewest conflicts for " + Arrays.toString(cells));
                assertEquals(slow[1], fast[1], "arrangements at that count for " + Arrays.toString(cells));
                regions++;
                if (slow[1] > 1) withTies++;
            }
        }
        assertEquals(80, regions);
        assertTrue(withTies > 0, "the damaged boards should contain regions with equal-cost alternatives");

        for (int t = 0; t < 12; t++) { // pieces shuffled round their cells: the best arrangement is a strict improvement, conflict-free
            int[] perfect = conflictFreeBoard(rnd);
            int[] cells = distinctCells(rnd, 3 + rnd.nextInt(2));
            int[] board = perfect.clone();
            for (int i = 0; i < cells.length; i++) board[cells[i]] = turn(perfect[cells[(i + 1) % cells.length]], rnd.nextInt(4));
            int[] fast = PlateauLns.bestRearrangement(board, cells);
            int[] slow = bruteForce(board, cells);
            assertEquals(0, slow[0]);
            assertEquals(slow[0], fast[0], "fewest conflicts for " + Arrays.toString(cells));
            assertEquals(slow[1], fast[1], "arrangements at that count for " + Arrays.toString(cells));
        }
    }

    @Test
    void restoresAScrambledClusterOfAConflictFreeBoard() {
        Random rnd = new Random(11);
        for (int trial = 0; trial < 6; trial++) {
            int[] perfect = conflictFreeBoard(rnd);
            int r0 = 2 + rnd.nextInt(10), c0 = 3 + rnd.nextInt(8);
            int[] cells = {r0 * 16 + c0, r0 * 16 + c0 + 1, (r0 + 1) * 16 + c0, (r0 + 1) * 16 + c0 + 1, (r0 + 2) * 16 + c0 + 1};
            int[] scrambled = perfect.clone();
            for (int i = cells.length - 1; i > 0; i--) { // shuffle the five pieces and turn some of them
                int j = rnd.nextInt(i + 1);
                int tmp = scrambled[cells[i]];
                scrambled[cells[i]] = PieceUtils.rotate(scrambled[cells[j]]);
                scrambled[cells[j]] = tmp;
            }
            if (scrambled[CENTRE] != perfect[CENTRE] || conflicts(scrambled) == 0) continue;
            boolean restored = false;
            for (long seed = 1; seed <= 6 && !restored; seed++) {
                int[] out = PlateauLns.improve(scrambled, null, 5_000, 60_000L, seed);
                restored = out != null && conflicts(out) == 0 && samePieces(perfect, out);
            }
            assertTrue(restored, "trial " + trial + " was not restored to a conflict-free board");
        }
    }

    @Test
    void neverMovesPinnedCellsOrTheCentreAndKeepsTheSetOfPieces() {
        Random rnd = new Random(23);
        boolean[] pinned = new boolean[256];
        for (int n = 0; n < 8; n++) pinned[rnd.nextInt(256)] = true;
        int improvedBoards = 0;
        for (int t = 0; t < 10; t++) {
            int[] board = conflictFreeBoard(rnd);
            int a, b;
            do { // two unpinned pieces a few cells apart in a row trade places, one of them turned
                a = rnd.nextInt(16) * 16 + rnd.nextInt(13);
                b = a + 1 + rnd.nextInt(3);
            } while (pinned[a] || pinned[b] || a == CENTRE || b == CENTRE);
            int tmp = board[a];
            board[a] = PieceUtils.rotate(board[b]);
            board[b] = tmp;
            if (conflicts(board) == 0) continue;
            for (long seed = 1; seed <= 2; seed++) {
                int[] out = PlateauLns.improve(board, pinned, 3_000, 60_000L, seed);
                if (out == null) continue;
                improvedBoards++;
                assertTrue(conflicts(out) < conflicts(board));
                assertTrue(samePieces(board, out));
                assertEquals(board[CENTRE], out[CENTRE]);
                for (int i = 0; i < 256; i++) if (pinned[i]) assertEquals(board[i], out[i], "pinned cell " + i);
            }
        }
        assertTrue(improvedBoards > 0, "some of the scrambled boards should be repaired");

        for (int t = 0; t < 6; t++) { // the cell holding the wrong piece is itself pinned: the board may not be repaired by moving it
            int[] board = conflictFreeBoard(rnd);
            int a = rnd.nextInt(16) * 16 + rnd.nextInt(13), b = a + 1 + rnd.nextInt(3);
            if (a == CENTRE || b == CENTRE) continue;
            int tmp = board[a];
            board[a] = PieceUtils.rotate(board[b]);
            board[b] = tmp;
            boolean[] almostAllPinned = new boolean[256];  // everything but twelve other cells is pinned, the wrong piece too
            Arrays.fill(almostAllPinned, true);
            for (int freed = 0; freed < 12; ) {
                int c = rnd.nextInt(256);
                if (c != a && c != CENTRE && almostAllPinned[c]) {
                    almostAllPinned[c] = false;
                    freed++;
                }
            }
            for (long seed = 1; seed <= 2; seed++) {
                int[] out = PlateauLns.improve(board, almostAllPinned, 3_000, 60_000L, seed);
                if (out == null) continue;
                for (int i = 0; i < 256; i++) if (almostAllPinned[i]) assertEquals(board[i], out[i], "pinned cell " + i);
            }
        }

        boolean[] tooFewFree = new boolean[256];           // fewer movable cells than a region needs: nothing to do, and no endless loop
        Arrays.fill(tooFewFree, true);
        for (int i = 0; i < 7; i++) tooFewFree[10 + i] = false;
        int[] board = conflictFreeBoard(rnd);
        int[] broken = board.clone();
        broken[10] = PieceUtils.rotate(board[11]);
        broken[11] = board[10];
        assertTimeoutPreemptively(Duration.ofSeconds(10), () -> assertNull(PlateauLns.improve(broken, tooFewFree, 3_000, 60_000L, 1)));

        assertTrue(PlateauLns.stats().contains("discarded=0 "), "the final check must never have had to throw a result away: " + PlateauLns.stats());
    }

    @Test
    void skipsPerfectIncompleteAndTooBadBoardsAndHonoursTheSwitch() {
        Random rnd = new Random(31);
        int[] perfect = conflictFreeBoard(rnd);
        assertNull(PlateauLns.improve(perfect, null, false), "nothing to improve");

        int[] incomplete = damage(perfect, rnd, 3);
        incomplete[17] = -1;
        assertNull(PlateauLns.improve(incomplete, null, false), "a board with an empty cell is skipped");

        int[] bad = damage(perfect, rnd, 25);
        assertNull(PlateauLns.improve(bad, null, false), "25 conflicts is above both ceilings");
        assertNull(PlateauLns.improve(bad, null, true));

        PlateauLns.configure(true, 5_000, 5_000, 12, 60_000L, 2, 10);
        int[] swapped = perfect.clone();
        int tmp = swapped[40];
        swapped[40] = swapped[41];
        swapped[41] = tmp;
        int cost = conflicts(swapped);
        assertTrue(cost > 2 && cost <= 10, "test board should sit between the two ceilings, was " + cost);
        assertNull(PlateauLns.improve(swapped, null, false), "above the no-hints ceiling");
        assertNotNull(PlateauLns.improve(swapped, null, true), "within the ceiling with hints");

        PlateauLns.configure(false, 5_000, 5_000, 12, 60_000L, 14, 19);
        assertNull(PlateauLns.improve(swapped, null, true), "switched off");
    }

    @Test
    void theBudgetIsLargerForBoardsAtTheDeepThreshold() {
        PlateauLns.configure(true, 100, 5_000, 12, 60_000L, 14, 19);
        assertEquals(5_000, PlateauLns.budgetFor(11));
        assertEquals(5_000, PlateauLns.budgetFor(12));
        assertEquals(100, PlateauLns.budgetFor(13));
        assertEquals(100, PlateauLns.budgetFor(14));
    }

    @Test
    void givesTheSameAnswerTwiceForTheSameBoard() throws IOException {
        PlateauLns.configure(true, 3_000, 3_000, 12, 60_000L, 14, 19);
        Fixture f = fixtures().get(3);                       // a 14 that different seeds improve in nine different ways
        int[] first = PlateauLns.improve(f.board, null, false);
        assertNotNull(first, "the 14 should improve within 3,000 rounds");
        assertArrayEquals(first, PlateauLns.improve(f.board, null, false));
        assertArrayEquals(first, PlateauLns.improve(f.board, null, false));
    }

    @Test
    void countsConflictsTheWayHoleSolverDoes() {
        Random rnd = new Random(53);
        for (int t = 0; t < 20; t++) {
            int[] board = damage(conflictFreeBoard(rnd), rnd, rnd.nextInt(30));
            int top = rnd.nextInt(16);
            board[top] = PieceUtils.pack(7, PieceUtils.getEast(board[top]), PieceUtils.getSouth(board[top]), PieceUtils.getWest(board[top])); // a coloured edge pointing out of the rim
            assertEquals(HoleSolver.findConflicts(board).size(), conflicts(board));
        }
    }

    @Test
    void statsReportWhatWasDone() throws IOException {
        Fixture f = fixtures().get(1);                       // a 13 that improves within a few hundred rounds
        assertNotNull(PlateauLns.improve(f.board, null, f.rounds, 120_000L, f.seed));
        String stats = PlateauLns.stats();
        assertTrue(stats.startsWith("lns[tried="), stats);
        assertTrue(stats.contains("plateauMoves=") && stats.contains("conflictsRemoved="), stats);
        assertTrue(!stats.contains("improved=0 "), stats);
    }

    @Test
    void improvesRealSavedBoardsThatTheOfflineSearchImproved() throws IOException {
        List<Fixture> fixtures = fixtures();
        assertEquals(5, fixtures.size());
        for (Fixture f : fixtures) {
            assertEquals(f.before, conflicts(f.board), f.name);
            int[] improved = PlateauLns.improve(f.board, null, f.rounds, 120_000L, f.seed);
            assertNotNull(improved, f.name + " should improve with seed " + f.seed + " and " + f.rounds + " rounds");
            int after = conflicts(improved);
            assertTrue(after < f.before, f.name);
            assertTrue(after <= f.reached, f.name + ": reached " + after + ", offline " + f.reached);
            assertTrue(samePieces(f.board, improved), f.name);
            assertEquals(f.board[CENTRE], improved[CENTRE], f.name);
        }
    }
}

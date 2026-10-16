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
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PairMovePolishTest {

    private static final int CENTRE = 135;

    /** name, conflicts before, conflicts the offline search reached, packed board. */
    private record Fixture(String name, int before, int reached, int[] board) {
    }

    /** Three saved 13-conflict boards (two GPU, one C#) that the offline pair-move search turned into valid 12s. */
    private static List<Fixture> fixtures() throws IOException {
        List<Fixture> out = new ArrayList<>();
        try (BufferedReader in = new BufferedReader(new InputStreamReader(
                PairMovePolishTest.class.getResourceAsStream("/pair-polish-boards.tsv"), StandardCharsets.UTF_8))) {
            String line;
            while ((line = in.readLine()) != null) {
                String[] f = line.split("\t");
                out.add(new Fixture(f[0], Integer.parseInt(f[1]), Integer.parseInt(f[2]), decode(f[3])));
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
        PairMovePolish.configure(true, 3, 400, 3_000L, 15, 20);
    }

    @BeforeEach
    void generousLimits() {
        PairMovePolish.configure(true, 3, 400, 120_000L, 15, 20);
    }

    @AfterEach
    void restoreDefaults() {
        defaults();
    }

    private static int turn(int p, int quarterTurns) {
        for (int k = 0; k < quarterTurns; k++) p = PieceUtils.rotate(p);
        return p;
    }

    private static int canonical(int p) {
        int best = p, q = p;
        for (int i = 0; i < 3; i++) {
            q = PieceUtils.rotate(q);
            best = Math.min(best, q);
        }
        return best;
    }

    private static int[] canonicalSorted(int[] board) {
        int[] c = new int[board.length];
        for (int i = 0; i < c.length; i++) c[i] = canonical(board[i]);
        Arrays.sort(c);
        return c;
    }

    // ------------------------------------------------------------------ real boards

    @Test
    void fixturesAreThirteenConflictBoardsUnderBothMetrics() throws IOException {
        for (Fixture f : fixtures()) {
            assertEquals(13, f.before, f.name);
            assertEquals(f.before, TailOptimizer.countConflicts(f.board), f.name);
            assertEquals(f.before, HoleSolver.findConflicts(f.board).size(), f.name);
        }
    }

    @Test
    void turnsTheSavedThirteensIntoTwelves() throws IOException {
        for (Fixture f : fixtures()) {
            int[] original = f.board.clone();
            int[] improved = PairMovePolish.improve(f.board, null, false);
            assertNotNull(improved, f.name + " should improve");
            assertArrayEquals(original, f.board, f.name + ": the input board must not be modified");
            assertEquals(f.reached, HoleSolver.findConflicts(improved).size(), f.name);
            assertEquals(f.reached, TailOptimizer.countConflicts(improved), f.name);
            assertArrayEquals(canonicalSorted(f.board), canonicalSorted(improved), f.name + ": every piece must still be there once");
            assertEquals(f.board[CENTRE], improved[CENTRE], f.name + ": the centre cell moved");
        }
    }

    @Test
    void neutralStepsAreWhatReachTwoOfThem() throws IOException {
        List<Fixture> fixtures = fixtures();
        // depth 0 = a single improving move only: just the first board has one
        PairMovePolish.configure(true, 0, 400, 120_000L, 13, 18);
        boolean[] improved = new boolean[fixtures.size()];
        for (int i = 0; i < fixtures.size(); i++) {
            improved[i] = PairMovePolish.improve(fixtures.get(i).board, null, false) != null;
        }
        assertArrayEquals(new boolean[]{true, false, false}, improved);
    }

    @Test
    void aNodeCapOfOneKeepsOnlyTheSingleMoveRepair() throws IOException {
        List<Fixture> fixtures = fixtures();
        PairMovePolish.configure(true, 3, 1, 120_000L, 13, 18);
        assertNotNull(PairMovePolish.improve(fixtures.get(0).board, null, false));
        assertNull(PairMovePolish.improve(fixtures.get(1).board, null, false));
        assertNull(PairMovePolish.improve(fixtures.get(2).board, null, false));
    }

    @Test
    void doesNothingWhenDisabledOrAboveTheCeiling() throws IOException {
        int[] board = fixtures().get(0).board;
        PairMovePolish.configure(false, 3, 400, 120_000L, 13, 18);
        assertNull(PairMovePolish.improve(board, null, false));
        PairMovePolish.configure(true, 3, 400, 120_000L, 12, 18);
        assertNull(PairMovePolish.improve(board, null, false), "13 conflicts is above a no-hints ceiling of 12");
        assertNotNull(PairMovePolish.improve(board, null, true), "...but within the hints ceiling");
        PairMovePolish.configure(true, 3, 400, 120_000L, 13, 12);
        assertNull(PairMovePolish.improve(board, null, true), "13 conflicts is above a hints ceiling of 12");
        assertNotNull(PairMovePolish.improve(board, null, false));
    }

    @Test
    void ignoresIncompleteAndMalformedBoards() throws IOException {
        int[] holed = fixtures().get(0).board.clone();
        holed[17] = -1;
        assertNull(PairMovePolish.improve(holed, null, false));
        assertNull(PairMovePolish.improve(new int[10], null, false));
        assertNull(PairMovePolish.improve(null, null, false));
    }

    @Test
    void neverMovesPinnedCells() throws IOException {
        for (Fixture f : fixtures()) {
            int[] free = PairMovePolish.improve(f.board, null, false);
            assertNotNull(free, f.name);
            boolean[] pinned = new boolean[256];
            int pins = 0;
            for (int i = 0; i < 256; i++) {
                if (free[i] != f.board[i]) {
                    pinned[i] = true;
                    pins++;
                }
            }
            assertTrue(pins > 0, f.name);
            int[] locked = PairMovePolish.improve(f.board, pinned, false);
            if (locked != null) {
                for (int i = 0; i < 256; i++) {
                    if (pinned[i]) assertEquals(f.board[i], locked[i], f.name + ": pinned cell " + i + " moved");
                }
                assertTrue(TailOptimizer.countConflicts(locked) < f.before, f.name);
            }
        }
    }

    // ------------------------------------------------------------------ synthetic boards

    /** A 16x16 board of random pieces that fit together perfectly (zero conflicts), grey on the frame only. */
    private static int[] randomPerfectBoard(long seed, int colours) {
        Random rnd = new Random(seed);
        int[][] across = new int[16][17]; // colour on the seam left of column c in row r
        int[][] down = new int[17][16];   // colour on the seam above row r in column c
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c <= 16; c++) across[r][c] = (c == 0 || c == 16) ? 0 : 1 + rnd.nextInt(colours);
        }
        for (int r = 0; r <= 16; r++) {
            for (int c = 0; c < 16; c++) down[r][c] = (r == 0 || r == 16) ? 0 : 1 + rnd.nextInt(colours);
        }
        int[] board = new int[256];
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                board[r * 16 + c] = PieceUtils.pack(down[r][c], across[r][c + 1], down[r + 1][c], across[r][c]);
            }
        }
        assertEquals(0, TailOptimizer.countConflicts(board));
        return board;
    }

    private static int[] swapped(int[] b, int i, int j, int x, int y) {
        int[] c = b.clone();
        c[i] = turn(b[j], x);
        c[j] = turn(b[i], y);
        return c;
    }

    private static int[] turnedBoth(int[] b, int i, int j, int x, int y) {
        int[] c = b.clone();
        c[i] = turn(b[i], x);
        c[j] = turn(b[j], y);
        return c;
    }

    /** Cell pairs covering far apart, side by side, one above the other, frame and corner cells. */
    private static final int[][] PAIRS = {
            {3, 200}, {17, 18}, {40, 56}, {0, 15}, {5, 250}, {100, 101}, {150, 166}, {16, 31}, {239, 255}, {64, 80}
    };

    @Test
    void undoesAnyTwoPieceSwapOnAPerfectBoard() {
        PairMovePolish.configure(true, 3, 400, 120_000L, 99, 99);
        int[] perfect = randomPerfectBoard(11, 12);
        int checked = 0;
        for (int[] pair : PAIRS) {
            for (int x = 0; x < 4; x++) {
                for (int y = 0; y < 4; y++) {
                    int[] broken = swapped(perfect, pair[0], pair[1], x, y);
                    if (TailOptimizer.countConflicts(broken) == 0) continue; // the pieces happened to fit anyway
                    int[] fixed = PairMovePolish.improve(broken, null, false);
                    String what = "swap " + Arrays.toString(pair) + " turns " + x + "," + y;
                    assertNotNull(fixed, what);
                    assertEquals(0, TailOptimizer.countConflicts(fixed), what);
                    assertArrayEquals(canonicalSorted(perfect), canonicalSorted(fixed), what);
                    checked++;
                }
            }
        }
        assertTrue(checked > 100, "only " + checked + " broken boards were exercised");
    }

    @Test
    void undoesAnyTwoPieceTurnOnAPerfectBoard() {
        PairMovePolish.configure(true, 3, 400, 120_000L, 99, 99);
        int[] perfect = randomPerfectBoard(23, 12);
        int checked = 0;
        for (int[] pair : PAIRS) {
            for (int x = 1; x < 4; x++) {
                for (int y = 1; y < 4; y++) {
                    int[] broken = turnedBoth(perfect, pair[0], pair[1], x, y);
                    if (TailOptimizer.countConflicts(broken) == 0) continue;
                    int[] fixed = PairMovePolish.improve(broken, null, false);
                    String what = "turn " + Arrays.toString(pair) + " by " + x + "," + y;
                    assertNotNull(fixed, what);
                    assertEquals(0, TailOptimizer.countConflicts(fixed), what);
                    checked++;
                }
            }
        }
        assertTrue(checked > 50, "only " + checked + " broken boards were exercised");
    }

    @Test
    void undoesASinglePieceTurn() {
        PairMovePolish.configure(true, 3, 400, 120_000L, 99, 99);
        int[] perfect = randomPerfectBoard(5, 12);
        for (int cell : new int[]{0, 7, 15, 16, 31, 100, 240, 255}) {
            for (int k = 1; k < 4; k++) {
                int[] broken = perfect.clone();
                broken[cell] = turn(perfect[cell], k);
                if (TailOptimizer.countConflicts(broken) == 0) continue;
                int[] fixed = PairMovePolish.improve(broken, null, false);
                assertNotNull(fixed, "cell " + cell + " turned " + k);
                assertEquals(0, TailOptimizer.countConflicts(fixed), "cell " + cell + " turned " + k);
            }
        }
    }

    @Test
    void neverMovesTheCentreCellEvenWhenItIsTheOnePieceOutOfPlace() {
        PairMovePolish.configure(true, 3, 400, 120_000L, 99, 99);
        int[] perfect = randomPerfectBoard(31, 12);
        int[] broken = swapped(perfect, CENTRE, 40, 1, 2);
        assertTrue(TailOptimizer.countConflicts(broken) > 0);
        int[] fixed = PairMovePolish.improve(broken, null, false);
        if (fixed != null) assertEquals(broken[CENTRE], fixed[CENTRE]);
        // with the centre held back the best that the other cells can do is still not worse than before
        int after = fixed == null ? TailOptimizer.countConflicts(broken) : TailOptimizer.countConflicts(fixed);
        assertTrue(after <= TailOptimizer.countConflicts(broken));
    }

    @Test
    void randomDamageNeverProducesAnInconsistentBoard() {
        PairMovePolish.configure(true, 2, 60, 120_000L, 99, 99);
        Random rnd = new Random(99);
        int[] perfect = randomPerfectBoard(77, 9);
        for (int round = 0; round < 40; round++) {
            int[] b = perfect.clone();
            for (int m = 0; m < 1 + rnd.nextInt(4); m++) {
                int i = rnd.nextInt(256), j = rnd.nextInt(256);
                if (i == j || i == CENTRE || j == CENTRE) continue;
                b = rnd.nextBoolean() ? swapped(b, i, j, rnd.nextInt(4), rnd.nextInt(4)) : turnedBoth(b, i, j, rnd.nextInt(4), rnd.nextInt(4));
            }
            int before = TailOptimizer.countConflicts(b);
            int[] out = PairMovePolish.improve(b, null, false);
            if (before == 0) {
                assertNull(out);
                continue;
            }
            if (out != null) {
                assertTrue(TailOptimizer.countConflicts(out) < before, "round " + round);
                assertArrayEquals(canonicalSorted(b), canonicalSorted(out), "round " + round);
                assertEquals(b[CENTRE], out[CENTRE], "round " + round);
            }
        }
    }

    /** Every interior seam has one colour and the frame is grey, except one piece whose east edge shows a colour nothing else has. */
    private static int[] flatBoardWithOneUnfixableEdge() {
        int[] board = new int[256];
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                board[r * 16 + c] = PieceUtils.pack(r == 0 ? 0 : 1, c == 15 ? 0 : 1, r == 15 ? 0 : 1, c == 0 ? 0 : 1);
            }
        }
        board[40] = PieceUtils.pack(1, 2, 1, 1);
        return board;
    }

    @Test
    void staysWithinItsBudgetOnABoardWithHundredsOfNeutralMoves() {
        // the odd piece can sit in any interior cell at the same cost, so the neutral search has nothing to stop it but the caps
        PairMovePolish.configure(true, 3, 400, 3_000L, 15, 20);
        int[] board = flatBoardWithOneUnfixableEdge();
        assertEquals(1, TailOptimizer.countConflicts(board));
        assertTimeoutPreemptively(Duration.ofSeconds(60), () -> assertNull(PairMovePolish.improve(board, null, false)));
        assertTrue(PairMovePolish.stats().contains("cappedSearches="), PairMovePolish.stats());
    }

    @Test
    void statsLineReportsTheCounters() throws IOException {
        PairMovePolish.improve(fixtures().get(0).board, null, false);
        assertTrue(PairMovePolish.stats().startsWith("pair[tried="), PairMovePolish.stats());
    }
}

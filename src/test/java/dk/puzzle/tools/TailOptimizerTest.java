package dk.puzzle.tools;

import dk.puzzle.util.PieceUtils;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TailOptimizerTest {

    /** name, conflicts before, best conflicts the offline experiment reached, packed board. */
    private record Fixture(String name, int before, int reached, int[] board) {
    }

    private static List<Fixture> fixtures() throws IOException {
        List<Fixture> out = new ArrayList<>();
        try (BufferedReader in = new BufferedReader(new InputStreamReader(
                TailOptimizerTest.class.getResourceAsStream("/tail-optimizer-boards.tsv"), StandardCharsets.UTF_8))) {
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

    @BeforeEach
    void generousLimits() {
        TailOptimizer.configure(true, new int[]{15, 20}, new long[]{100_000_000L}, 120_000L, 20);
    }

    @AfterEach
    void restoreDefaults() {
        TailOptimizer.configure(true, new int[]{15, 20}, new long[]{2_000_000L, 10_000_000L}, 3_000L, 20);
    }

    @Test
    void conflictCountMatchesHoleSolverOnRealBoards() throws IOException {
        for (Fixture f : fixtures()) {
            assertEquals(f.before, TailOptimizer.countConflicts(f.board), f.name);
            assertEquals(f.before, HoleSolver.findConflicts(f.board).size(), f.name);
        }
    }

    @Test
    void improvesRealBoardsAndKeepsEverythingElseIntact() throws IOException {
        for (Fixture f : fixtures()) {
            int[] improved = TailOptimizer.improve(f.board, null, 0);
            assertNotNull(improved, f.name + " should improve");
            int after = HoleSolver.findConflicts(improved).size();
            assertTrue(after <= f.reached, f.name + ": " + f.before + " -> " + after + ", the offline experiment reached " + f.reached);
            assertTrue(after < f.before, f.name);
            assertEquals(after, TailOptimizer.countConflicts(improved), f.name);
            // only the last 20 cells of the fill order may differ, and they must hold the same pieces
            boolean[] tail = new boolean[256];
            for (int s = 236; s < 256; s++) tail[TailOptimizer.cellOfStep(s, 0)] = true;
            int[] x = new int[20], y = new int[20];
            int n = 0;
            for (int i = 0; i < 256; i++) {
                if (tail[i]) {
                    x[n] = canonical(f.board[i]);
                    y[n] = canonical(improved[i]);
                    n++;
                } else {
                    assertEquals(f.board[i], improved[i], f.name + ": cell " + i + " outside the tail moved");
                }
            }
            Arrays.sort(x);
            Arrays.sort(y);
            assertArrayEquals(x, y, f.name + ": the tail must hold the same pieces");
        }
    }

    private static int canonical(int p) {
        int best = p, q = p;
        for (int i = 0; i < 3; i++) {
            q = PieceUtils.rotate(q);
            best = Math.min(best, q);
        }
        return best;
    }

    @Test
    void neverMovesPinnedCells() throws IOException {
        boolean[] pinned = new boolean[256];
        int[] pinnedCells = {TailOptimizer.cellOfStep(255, 0), TailOptimizer.cellOfStep(249, 0), TailOptimizer.cellOfStep(240, 0)};
        for (int c : pinnedCells) pinned[c] = true;
        for (Fixture f : fixtures()) {
            int[] improved = TailOptimizer.improve(f.board, pinned, 0);
            if (improved == null) continue;
            for (int c : pinnedCells) assertEquals(f.board[c], improved[c], f.name + ": pinned cell " + c + " moved");
            assertTrue(TailOptimizer.countConflicts(improved) < f.before, f.name);
        }
    }

    @Test
    void matchesBruteForceOnAScrambledFiveCellTail() throws IOException {
        Fixture f = fixtures().get(0);
        int[] cells = new int[5];
        for (int i = 0; i < 5; i++) cells[i] = TailOptimizer.cellOfStep(251 + i, 0);
        int[] pieces = new int[5];
        for (int i = 0; i < 5; i++) pieces[i] = f.board[cells[i]];

        Random rnd = new Random(7);
        int[] scrambled = f.board.clone();
        int[] order = {3, 0, 4, 2, 1};
        for (int i = 0; i < 5; i++) {
            int p = pieces[order[i]];
            for (int r = rnd.nextInt(4); r > 0; r--) p = PieceUtils.rotate(p);
            scrambled[cells[i]] = p;
        }

        int[] best = {TailOptimizer.countConflicts(scrambled)};
        int scrambledConflicts = best[0];
        permute(new int[]{0, 1, 2, 3, 4}, 0, pieces, cells, scrambled.clone(), best);

        TailOptimizer.configure(true, new int[]{5}, new long[]{Long.MAX_VALUE}, 600_000L, 99);
        int[] improved = TailOptimizer.improve(scrambled, null, 0);
        assertNotNull(improved);
        assertTrue(best[0] < scrambledConflicts);
        assertEquals(best[0], TailOptimizer.countConflicts(improved), "must reach the brute-force optimum");
    }

    private static void permute(int[] perm, int k, int[] pieces, int[] cells, int[] work, int[] best) {
        if (k == perm.length) {
            rotations(perm, 0, pieces, cells, work, best);
            return;
        }
        for (int i = k; i < perm.length; i++) {
            int t = perm[k]; perm[k] = perm[i]; perm[i] = t;
            permute(perm, k + 1, pieces, cells, work, best);
            t = perm[k]; perm[k] = perm[i]; perm[i] = t;
        }
    }

    private static void rotations(int[] perm, int i, int[] pieces, int[] cells, int[] work, int[] best) {
        if (i == perm.length) {
            best[0] = Math.min(best[0], TailOptimizer.countConflicts(work));
            return;
        }
        int p = pieces[perm[i]];
        for (int r = 0; r < 4; r++) {
            work[cells[i]] = p;
            rotations(perm, i + 1, pieces, cells, work, best);
            p = PieceUtils.rotate(p);
        }
    }

    @Test
    void skipsIncompleteBoardsAndBoardsAboveTheConflictLimit() throws IOException {
        Fixture f = fixtures().get(0);
        int[] holed = f.board.clone();
        holed[100] = -1;
        assertNull(TailOptimizer.improve(holed, null, 0));

        TailOptimizer.configure(true, new int[]{15, 20}, new long[]{100_000_000L}, 120_000L, f.before - 1);
        assertNull(TailOptimizer.improve(f.board, null, 0));
    }

    @Test
    void doesNothingWhenDisabled() throws IOException {
        TailOptimizer.configure(false, new int[]{15, 20}, new long[]{100_000_000L}, 120_000L, 20);
        assertNull(TailOptimizer.improve(fixtures().get(0).board, null, 0));
    }

}

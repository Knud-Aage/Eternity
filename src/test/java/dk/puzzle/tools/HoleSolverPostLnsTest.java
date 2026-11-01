package dk.puzzle.tools;

import dk.puzzle.util.PieceUtils;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

/**
 * Once the plateau LNS has lowered the count, HoleSolver gives the pair polish another pass and then the tail optimizer. The stages here are
 * stubs that hand back prepared boards, so the test pins the order, what each stage receives, and that a stage which finds nothing, returns a
 * worse board or throws costs nothing.
 */
class HoleSolverPostLnsTest {

    /** A flat 16x16 board: border edges 0, inner edges 1, so no conflicts at all. */
    private static int[] flat() {
        int[] board = new int[256];
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                board[r * 16 + c] = PieceUtils.pack(r == 0 ? 0 : 1, c == 15 ? 0 : 1, r == 15 ? 0 : 1, c == 0 ? 0 : 1);
            }
        }
        return board;
    }

    /** The flat board with n interior pieces whose east edge does not fit their right neighbour: exactly n conflicts, none touching another. */
    private static int[] withConflicts(int n) {
        int[] board = flat();
        for (int i = 0; i < n; i++) {
            board[(3 + i) * 16 + 5] = PieceUtils.pack(1, 2, 1, 1);
        }
        return board;
    }

    @Test
    void thePairPolishRunsFirstAndTheTailOptimizerGetsItsResult() {
        int[] b3 = withConflicts(3), b2 = withConflicts(2), b1 = withConflicts(1);
        List<String> calls = new ArrayList<>();
        UnaryOperator<int[]> pair = b -> { calls.add("pair"); assertSame(b3, b); return b2; };
        UnaryOperator<int[]> tail = b -> { calls.add("tail"); assertSame(b2, b); return b1; };

        int[] result = HoleSolver.polishAfterPlateauGain(b3, false, pair, tail);

        assertEquals(List.of("pair", "tail"), calls);
        assertSame(b1, result);
    }

    @Test
    void theTailOptimizerStillRunsOnTheLnsBoardWhenThePairPolishFindsNothing() {
        int[] b3 = withConflicts(3), b2 = withConflicts(2);
        List<String> calls = new ArrayList<>();
        UnaryOperator<int[]> pair = b -> { calls.add("pair"); return null; };
        UnaryOperator<int[]> tail = b -> { calls.add("tail"); assertSame(b3, b); return b2; };

        int[] result = HoleSolver.polishAfterPlateauGain(b3, false, pair, tail);

        assertEquals(List.of("pair", "tail"), calls);
        assertSame(b2, result);
    }

    @Test
    void aPairPolishThatReturnsAWorseOrEqualBoardIsIgnored() {
        int[] b3 = withConflicts(3);
        UnaryOperator<int[]> worse = b -> withConflicts(5);
        UnaryOperator<int[]> equal = b -> withConflicts(3);
        UnaryOperator<int[]> tailSeesTheOriginal = b -> { assertSame(b3, b); return null; };

        assertSame(b3, HoleSolver.polishAfterPlateauGain(b3, false, worse, tailSeesTheOriginal));
        assertSame(b3, HoleSolver.polishAfterPlateauGain(b3, false, equal, tailSeesTheOriginal));
    }

    @Test
    void aFailingPairPolishCostsNothingAndTheTailOptimizerStillRuns() {
        int[] b3 = withConflicts(3), b2 = withConflicts(2);
        UnaryOperator<int[]> pair = b -> { throw new IllegalStateException("boom"); };
        UnaryOperator<int[]> tail = b -> { assertSame(b3, b); return b2; };

        assertSame(b2, HoleSolver.polishAfterPlateauGain(b3, false, pair, tail));
    }

    @Test
    void whenNeitherStageHelpsTheSameBoardComesBack() {
        int[] b3 = withConflicts(3);
        UnaryOperator<int[]> nothing = b -> null;

        assertSame(b3, HoleSolver.polishAfterPlateauGain(b3, false, nothing, nothing));
    }

    @Test
    void theSyntheticBoardsHaveTheConflictCountsTheTestsRelyOn() {
        // guard for the helper above: flat() is conflict free and withConflicts(n) has exactly n
        for (int n = 0; n <= 5; n++) {
            int[] b = withConflicts(n);
            int[] improved = withConflicts(n - 1 < 0 ? 0 : n - 1);
            int[] kept = HoleSolver.polishAfterPlateauGain(b, false, x -> improved, x -> null);
            if (n == 0) assertSame(b, kept);          // nothing can be better than zero
            else assertSame(improved, kept);          // one conflict fewer is accepted
        }
    }
}

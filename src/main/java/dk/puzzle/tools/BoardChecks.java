package dk.puzzle.tools;

import dk.puzzle.util.PieceUtils;

import java.util.Arrays;

/** Sanity checks the polish stages run on their own result before handing a board back. */
final class BoardChecks {

    private BoardChecks() {
    }

    /** Every piece must still be there exactly once (compared up to rotation). */
    static boolean samePieces(int[] a, int[] b) {
        int[] x = new int[256], y = new int[256];
        for (int i = 0; i < 256; i++) {
            x[i] = canonical(a[i]);
            y[i] = canonical(b[i]);
        }
        Arrays.sort(x);
        Arrays.sort(y);
        return Arrays.equals(x, y);
    }

    /** True if any locked cell differs between the two boards. */
    static boolean movedLocked(int[] a, int[] b, boolean[] locked) {
        for (int i = 0; i < 256; i++) {
            if (locked[i] && a[i] != b[i]) return true;
        }
        return false;
    }

    /** The smallest packed value among the piece's four rotations. */
    static int canonical(int p) {
        int best = p, q = p;
        for (int i = 0; i < 3; i++) {
            q = PieceUtils.rotate(q);
            if (q < best) best = q;
        }
        return best;
    }
}

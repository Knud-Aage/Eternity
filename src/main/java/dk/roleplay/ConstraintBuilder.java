package dk.roleplay;

import java.util.Arrays;

public class ConstraintBuilder {
    public static int[] build(int macroIdx, int[][] mainBoard) {
        int[] constraints = new int[16];
        Arrays.fill(constraints, PieceUtils.pack(PieceUtils.WILDCARD, PieceUtils.WILDCARD, PieceUtils.WILDCARD,
                PieceUtils.WILDCARD));

        int mRow = macroIdx / 4;
        int mCol = macroIdx % 4;

        for (
                int i = 0;
                i < 16;
                i++
        ) {
            int r = i / 4, c = i % 4;
            int n = PieceUtils.WILDCARD, e = PieceUtils.WILDCARD, s = PieceUtils.WILDCARD, w = PieceUtils.WILDCARD;

            // Borders
            if (mRow == 0 && r == 0) {
                n = 0;
            }
            if (mRow == 3 && r == 3) {
                s = 0;
            }
            if (mCol == 0 && c == 0) {
                w = 0;
            }
            if (mCol == 3 && c == 3) {
                e = 0;
            }

            // Neighbor match (North Macro)
            if (r == 0 && mRow > 0 && mainBoard[macroIdx - 4] != null) {
                n = PieceUtils.getSouth(mainBoard[macroIdx - 4][12 + c]);
            }
            // Neighbor match (West Macro)
            if (c == 0 && mCol > 0 && mainBoard[macroIdx - 1] != null) {
                w = PieceUtils.getEast(mainBoard[macroIdx - 1][r * 4 + 3]);
            }

            constraints[i] = PieceUtils.pack(n, e, s, w);
        }
        return constraints;
    }
}

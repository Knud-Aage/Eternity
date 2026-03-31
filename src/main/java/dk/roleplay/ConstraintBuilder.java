package dk.roleplay;

import java.util.Arrays;

public class ConstraintBuilder {

    public static int[] build(int macroIdx, int[][] mainBoard) {
        int[] constraints = new int[16];

        // CRITICAL FIX: Initialize all edges to WILDCARD (255) instead of Java's default 0 (Border)
        int wildcardPacked = PieceUtils.pack(PieceUtils.WILDCARD, PieceUtils.WILDCARD, PieceUtils.WILDCARD, PieceUtils.WILDCARD);
        Arrays.fill(constraints, wildcardPacked);

        int mRow = macroIdx / 4;
        int mCol = macroIdx % 4;

        // 1. Absolute Board Borders
        for (int i = 0; i < 16; i++) {
            int r = i / 4;
            int c = i % 4;
            int globalRow = mRow * 4 + r;
            int globalCol = mCol * 4 + c;

            // Apply absolute border colors to the outer edges of the 16x16 board
            if (globalRow == 0)  applyNorth(constraints, i, PieceUtils.BORDER_COLOR);
            if (globalRow == 15) applySouth(constraints, i, PieceUtils.BORDER_COLOR);
            if (globalCol == 0)  applyWest(constraints, i, PieceUtils.BORDER_COLOR);
            if (globalCol == 15) applyEast(constraints, i, PieceUtils.BORDER_COLOR);
        }

        // 2. Adjacent Macro-Tile Borders (Dynamic Constraints)
        // If there's a macro-tile already placed ABOVE us, we must match its bottom edge
        if (mRow > 0 && mainBoard[macroIdx - 4] != null) {
            for (int c = 0; c < 4; c++) {
                int pieceAbove = mainBoard[macroIdx - 4][12 + c]; // The 4 bottom-row pieces of the tile above
                applyNorth(constraints, c, PieceUtils.getSouth(pieceAbove));
            }
        }

        // If there's a macro-tile already placed to the LEFT of us, we must match its right edge
        if (mCol > 0 && mainBoard[macroIdx - 1] != null) {
            for (int r = 0; r < 4; r++) {
                int pieceLeft = mainBoard[macroIdx - 1][r * 4 + 3]; // The 4 rightmost-column pieces of the tile left
                applyWest(constraints, r * 4, PieceUtils.getEast(pieceLeft));
            }
        }

/*      // (Optional) If you ever decide to build out-of-order, check South and East macros
        if (mRow < 3 && mainBoard[macroIdx + 4] != null) {
            for (int c = 0; c < 4; c++) {
                int pieceBelow = mainBoard[macroIdx + 4][c];
                applySouth(constraints, 12 + c, PieceUtils.getNorth(pieceBelow));
            }
        }
        if (mCol < 3 && mainBoard[macroIdx + 1] != null) {
            for (int r = 0; r < 4; r++) {
                int pieceRight = mainBoard[macroIdx + 1][r * 4];
                applyEast(constraints, r * 4 + 3, PieceUtils.getWest(pieceRight));
            }
        }
*/
        return constraints;
    }

    // Safely injects the 8-bit color ID into the exact edge slot, preserving the other 24 bits
    private static void applyNorth(int[] arr, int idx, int color) {
        arr[idx] = (arr[idx] & 0x00FFFFFF) | ((color & 0xFF) << 24);
    }

    private static void applyEast(int[] arr, int idx, int color) {
        arr[idx] = (arr[idx] & 0xFF00FFFF) | ((color & 0xFF) << 16);
    }

    private static void applySouth(int[] arr, int idx, int color) {
        arr[idx] = (arr[idx] & 0xFFFF00FF) | ((color & 0xFF) << 8);
    }

    private static void applyWest(int[] arr, int idx, int color) {
        arr[idx] = (arr[idx] & 0xFFFFFF00) | (color & 0xFF);
    }
}


/*
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
*/
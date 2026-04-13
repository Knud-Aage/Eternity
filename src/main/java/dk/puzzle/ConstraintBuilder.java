package dk.puzzle;

import java.util.Arrays;

/**
 * Constructs constraint arrays for 4x4 macro-tiles.
 * Aggregates information from board boundaries, adjacent placed tiles,
 * and fixed hint pieces to build a requirement mask for candidate generation.
 */
public class ConstraintBuilder {

    /**
     * Builds a 16-piece constraint array for a specific macro-tile position.
     *
     * @param macroIdx  The index of the macro-tile being solved (0-15)
     * @param mainBoard The current state of the 16x16 puzzle board
     * @return An array of 16 packed integers representing the edge requirements for each slot
     */
    public static int[] build(int macroIdx, int[][] mainBoard) {
        int[] constraints = new int[16];

        int wildcardPacked = PieceUtils.pack(PieceUtils.WILDCARD, PieceUtils.WILDCARD, PieceUtils.WILDCARD,
                PieceUtils.WILDCARD);
        Arrays.fill(constraints, wildcardPacked);

        int mRow = macroIdx / 4;
        int mCol = macroIdx % 4;

        for (int i = 0; i < 16; i++) {
            int r = i / 4;
            int c = i % 4;
            int globalRow = mRow * 4 + r;
            int globalCol = mCol * 4 + c;

            if (globalRow == 0) {
                applyNorth(constraints, i, PieceUtils.BORDER_COLOR);
            }
            if (globalRow == 15) {
                applySouth(constraints, i, PieceUtils.BORDER_COLOR);
            }
            if (globalCol == 0) {
                applyWest(constraints, i, PieceUtils.BORDER_COLOR);
            }
            if (globalCol == 15) {
                applyEast(constraints, i, PieceUtils.BORDER_COLOR);
            }
        }

        // Match ABOVE neighbor
        if (mRow > 0 && mainBoard[macroIdx - 4] != null) {
            for (int c = 0; c < 4; c++) {
                int pieceAbove = mainBoard[macroIdx - 4][12 + c]; // bottom row of above macro
                if (pieceAbove != PieceUtils.WILDCARD) {
                    applyNorth(constraints, c, PieceUtils.getSouth(pieceAbove));
                }
            }
        }

        // Match BELOW neighbor (NEW for Spiral)
        if (mRow < 3 && mainBoard[macroIdx + 4] != null) {
            for (int c = 0; c < 4; c++) {
                int pieceBelow = mainBoard[macroIdx + 4][c]; // top row of below macro
                if (pieceBelow != PieceUtils.WILDCARD) {
                    applySouth(constraints, 12 + c, PieceUtils.getNorth(pieceBelow));
                }
            }
        }

        // Match LEFT neighbor
        if (mCol > 0 && mainBoard[macroIdx - 1] != null) {
            for (int r = 0; r < 4; r++) {
                int pieceLeft = mainBoard[macroIdx - 1][r * 4 + 3]; // right col of left macro
                if (pieceLeft != PieceUtils.WILDCARD) {
                    applyWest(constraints, r * 4, PieceUtils.getEast(pieceLeft));
                }
            }
        }

        // Match RIGHT neighbor (NEW for Spiral)
        if (mCol < 3 && mainBoard[macroIdx + 1] != null) {
            for (int r = 0; r < 4; r++) {
                int pieceRight = mainBoard[macroIdx + 1][r * 4]; // left col of right macro
                if (pieceRight != PieceUtils.WILDCARD) {
                    applyEast(constraints, r * 4 + 3, PieceUtils.getWest(pieceRight));
                }
            }
        }

        if (macroIdx == 5) {
            // Force Position 15 to perfectly match the centerpiece in memory
            applyNorth(constraints, 15, 18);
            applyEast(constraints, 15, 12);
            applySouth(constraints, 15, 18);
            applyWest(constraints, 15, 3);

            applyEast(constraints, 14, 3);
            // The piece ABOVE (11) must connect to the centerpiece's North (18)
            applySouth(constraints, 11, 18);
        }

        return constraints;
    }

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

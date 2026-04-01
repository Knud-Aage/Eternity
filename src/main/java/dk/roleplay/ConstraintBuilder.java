package dk.roleplay;

import java.util.Arrays;

public class ConstraintBuilder {

    public static int[] build(int macroIdx, int[][] mainBoard) {
        int[] constraints = new int[16];

        // Initialize all edges to WILDCARD (255) instead of Java's default 0 (Border)
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
        // Match ABOVE neighbor
        if (mRow > 0 && mainBoard[macroIdx - 4] != null) {
            for (int c = 0; c < 4; c++) {
                int pieceAbove = mainBoard[macroIdx - 4][12 + c];
                if (pieceAbove != PieceUtils.WILDCARD) {
                    applyNorth(constraints, c, PieceUtils.getSouth(pieceAbove));
                }
            }
        }

        // Match LEFT neighbor
        if (mCol > 0 && mainBoard[macroIdx - 1] != null) {
            for (int r = 0; r < 4; r++) {
                int pieceLeft = mainBoard[macroIdx - 1][r * 4 + 3];
                if (pieceLeft != PieceUtils.WILDCARD) {
                    applyWest(constraints, r * 4, PieceUtils.getEast(pieceLeft));
                }
            }
        }

        // 3. Official Fixed Piece Match (#139: N=18, E=18, S=3, W=12 at Global 7,7)
        // Global (7,7) is Macro-Tile 5, internal position 15
        
        if (macroIdx == 5) {
            // Force Position 15 to perfectly match the centerpiece.
            // The PermutationGenerator will be forced to use Piece #139 here.
            applyNorth(constraints, 15, 18);
            applyEast(constraints, 15, 18);
            applySouth(constraints, 15, 3);
            applyWest(constraints, 15, 12);
        }
        
        // Match West of fixed piece (Macro 5, Pos 14 matches Fixed Piece Pos 15 West)
        if (macroIdx == 5) applyEast(constraints, 14, 12);

        // Match East of fixed piece (Macro 6, Pos 12 matches Fixed Piece Pos 15 East)
        if (macroIdx == 6) applyWest(constraints, 12, 18);
        
        // Match South of fixed piece (Macro 9, Pos 3 matches Fixed Piece Pos 15 South)
        if (macroIdx == 9) applyNorth(constraints, 3, 3);

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

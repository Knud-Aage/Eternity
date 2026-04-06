package dk.roleplay;

import java.util.*;

public class PieceInventory {
    public final int[] allOrientations = new int[1024];
    public final int[] physicalMapping = new int[1024];

    public final List<Integer> corners = new ArrayList<>();
    public final List<Integer> edges = new ArrayList<>();
    public final List<Integer> interior = new ArrayList<>();

    public List<Integer>[][] compatibility;
    public int[] colorFrequency;

    @SuppressWarnings("unchecked")
    public PieceInventory(int[] basePieces) {
        int maxColor = 0;
        for (int p : basePieces) {
            maxColor = Math.max(maxColor, Math.max(PieceUtils.getNorth(p), Math.max(PieceUtils.getEast(p),
                    Math.max(PieceUtils.getSouth(p), PieceUtils.getWest(p)))));
        }

        int colorCount = maxColor + 1;
        compatibility = new ArrayList[4][colorCount];
        colorFrequency = new int[colorCount];
        for (int i = 0; i < 4; i++) {
            for (int c = 0; c < colorCount; c++) compatibility[i][c] = new ArrayList<>();
        }

        for (int i = 0; i < 256; i++) {
            int oriented = basePieces[i];
            for (int r = 0; r < 4; r++) {
                int orientationIdx = i * 4 + r;
                allOrientations[orientationIdx] = oriented;
                physicalMapping[orientationIdx] = i;

                compatibility[0][PieceUtils.getNorth(oriented)].add(orientationIdx);
                compatibility[1][PieceUtils.getEast(oriented)].add(orientationIdx);
                compatibility[2][PieceUtils.getSouth(oriented)].add(orientationIdx);
                compatibility[3][PieceUtils.getWest(oriented)].add(orientationIdx);

                colorFrequency[PieceUtils.getNorth(oriented)]++;
                colorFrequency[PieceUtils.getEast(oriented)]++;
                colorFrequency[PieceUtils.getSouth(oriented)]++;
                colorFrequency[PieceUtils.getWest(oriented)]++;

                // Categorize based on the border count of THIS orientation
                int borders = PieceUtils.getBorderCount(oriented);
                if (borders == 2) corners.add(orientationIdx);
                else if (borders == 1) edges.add(orientationIdx);
                else interior.add(orientationIdx);

                oriented = PieceUtils.rotate(oriented);
            }
        }
        System.out.println("Inventory Initialized:");
        System.out.println(" - " + corners.size() + " oriented corners");
        System.out.println(" - " + edges.size() + " oriented edges");
        System.out.println(" - " + interior.size() + " oriented interior pieces");
    }

    // --- UPDATED MATCHING RULE ---
    private boolean matches(int p, int n_req, int e_req, int s_req, int w_req) {
        if (!colorMatches(n_req, PieceUtils.getNorth(p))) return false;
        if (!colorMatches(e_req, PieceUtils.getEast(p))) return false;
        if (!colorMatches(s_req, PieceUtils.getSouth(p))) return false;
        if (!colorMatches(w_req, PieceUtils.getWest(p))) return false;
        return true;
    }

    private boolean colorMatches(int req, int pieceColor) {
        if (req == PieceUtils.WILDCARD) return true; // We don't care about this edge yet

        // Borders must strictly match borders (0 == 0)
        if (req == PieceUtils.BORDER_COLOR || pieceColor == PieceUtils.BORDER_COLOR) {
            return req == pieceColor;
        }

        // Internal tabs and blanks MUST sum to 23
        return (req + pieceColor == 23);
    }
}

package dk.puzzle;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Manages the collection of Eternity II puzzle pieces.
 * Pre-calculates rotations, categorizes pieces by type (Corner, Edge, Interior),
 * and maintains compatibility tables for fast lookup.
 */
public class PieceInventory {
    public final int[] allOrientations = new int[1024];
    public final int[] physicalMapping = new int[1024];

    public final List<Integer> corners = new ArrayList<>();
    public final List<Integer> edges = new ArrayList<>();
    public final List<Integer> interior = new ArrayList<>();

    // New: Precomputed maps for fast lookup of orientations by edge pattern
    public final Map<Integer, List<Integer>> northEdgeToOrientations = new HashMap<>();
    public final Map<Integer, List<Integer>> eastEdgeToOrientations = new HashMap<>();
    public final Map<Integer, List<Integer>> southEdgeToOrientations = new HashMap<>();
    public final Map<Integer, List<Integer>> westEdgeToOrientations = new HashMap<>();


    public List<Integer>[][] compatibility; // This seems unused in MasterSolverPBP, but keeping it for now
    public int[] colorFrequency; // This seems unused in MasterSolverPBP, but keeping it for now

    /**
     * Initializes the inventory from base physical pieces.
     *
     * @param basePieces Array of 256 physical pieces (North, East, South, West).
     */
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

                // Populate compatibility (if still used elsewhere)
                compatibility[0][PieceUtils.getNorth(oriented)].add(orientationIdx);
                compatibility[1][PieceUtils.getEast(oriented)].add(orientationIdx);
                compatibility[2][PieceUtils.getSouth(oriented)].add(orientationIdx);
                compatibility[3][PieceUtils.getWest(oriented)].add(orientationIdx);

                // Populate new edge-to-orientations maps
                northEdgeToOrientations.computeIfAbsent(PieceUtils.getNorth(oriented), k -> new ArrayList<>()).add(orientationIdx);
                eastEdgeToOrientations.computeIfAbsent(PieceUtils.getEast(oriented), k -> new ArrayList<>()).add(orientationIdx);
                southEdgeToOrientations.computeIfAbsent(PieceUtils.getSouth(oriented), k -> new ArrayList<>()).add(orientationIdx);
                westEdgeToOrientations.computeIfAbsent(PieceUtils.getWest(oriented), k -> new ArrayList<>()).add(orientationIdx);


                colorFrequency[PieceUtils.getNorth(oriented)]++;
                colorFrequency[PieceUtils.getEast(oriented)]++;
                colorFrequency[PieceUtils.getSouth(oriented)]++;
                colorFrequency[PieceUtils.getWest(oriented)]++;

                int borders = PieceUtils.getBorderCount(oriented);
                if (borders == 2) {
                    corners.add(orientationIdx);
                } else if (borders == 1) {
                    edges.add(orientationIdx);
                } else {
                    interior.add(orientationIdx);
                }

                oriented = PieceUtils.rotate(oriented);
            }
        }
    }

    /**
     * Determines the appropriate piece pool (Corner, Edge, or Interior) for a given board position.
     *
     * @param mIdx Macro-tile index (0-15).
     * @param pIdx Piece index within the macro-tile (0-15).
     * @return A List of orientation indices corresponding to the correct piece type for that position.
     */
    public List<Integer> getPoolFor(int mIdx, int pIdx) {
        int mRow = mIdx / 4, mCol = mIdx % 4;
        int r = pIdx / 4, c = pIdx % 4;
        int globalRow = mRow * 4 + r;
        int globalCol = mCol * 4 + c;

        boolean n = (globalRow == 0), s = (globalRow == 15), w = (globalCol == 0), e = (globalCol == 15);
        int b = 0;
        if (n) {
            b++;
        }
        if (s) {
            b++;
        }
        if (w) {
            b++;
        }
        if (e) {
            b++;
        }

        if (b == 2) {
            return corners;
        }
        if (b == 1) {
            return edges;
        }
        return interior;
    }

    /**
     * Implements the color matching logic, including the Sum-to-23 rule.
     *
     * @param req        The required color from a neighboring edge.
     * @param pieceColor The color of the edge on the current piece.
     * @return true if they match legally.
     */
    public boolean colorMatches(int req, int pieceColor) {
        if (req == PieceUtils.WILDCARD) {
            return true;
        }
        if (req == PieceUtils.BORDER_COLOR || pieceColor == PieceUtils.BORDER_COLOR) {
            return req == pieceColor;
        }
        return (req + pieceColor == 23);
    }
}

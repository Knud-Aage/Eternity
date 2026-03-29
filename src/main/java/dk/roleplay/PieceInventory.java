package dk.roleplay;

import java.util.*;

public class PieceInventory {
    public final int[] allOrientations = new int[1024];
    public final int[] physicalMapping = new int[1024];
    
    public final List<Integer> corners = new ArrayList<>();
    public final List<Integer> edges = new ArrayList<>();
    public final List<Integer> interior = new ArrayList<>();
    
    public final List<Integer>[][] compatibility = new ArrayList[4][23];
    public final int[] colorFrequency = new int[23];

    public PieceInventory(int[] basePieces) {
        for (int i = 0; i < 4; i++) {
            for (int c = 0; c < 23; c++) compatibility[i][c] = new ArrayList<>();
        }

        for (int i = 0; i < 256; i++) {
            int p = basePieces[i];
            int borders = PieceUtils.getBorderCount(p);
            
            int oriented = p;
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

                if (borders == 2) corners.add(orientationIdx);
                else if (borders == 1) edges.add(orientationIdx);
                else interior.add(orientationIdx);
                
                oriented = PieceUtils.rotate(oriented);
            }
        }
    }

    public List<Integer> getPoolFor(int mIdx, int pIdx) {
        int mRow = mIdx / 4, mCol = mIdx % 4;
        int r = pIdx / 4, c = pIdx % 4;
        int globalRow = mRow * 4 + r;
        int globalCol = mCol * 4 + c;

        boolean n = (globalRow == 0), s = (globalRow == 15), w = (globalCol == 0), e = (globalCol == 15);
        int b = 0;
        if (n) b++; if (s) b++; if (w) b++; if (e) b++;

        if (b == 2) return corners;
        if (b == 1) return edges;
        return interior;
    }
}

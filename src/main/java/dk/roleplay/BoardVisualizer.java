package dk.roleplay;

import javax.swing.*;
import java.awt.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class BoardVisualizer extends JPanel {
    private static final int PIECE_SIZE = 50;
    private static final Map<Integer, Color> COLORS = new HashMap<>();

    static {
        COLORS.put(0, Color.DARK_GRAY); // Border color - dark gray, not black
        COLORS.put(PieceUtils.WILDCARD, Color.MAGENTA);
        Random r = new Random(123);
        for (int i = 1; i <= 255; i++) {
            COLORS.put(i, new Color(r.nextInt(200) + 55, r.nextInt(200) + 55, r.nextInt(200) + 55));
        }
    }

    private final int[][] mainBoardRef;

    public BoardVisualizer(int[][] mainBoardRef) {
        this.mainBoardRef = mainBoardRef;
        setPreferredSize(new Dimension(16 * PIECE_SIZE, 16 * PIECE_SIZE));
        setBackground(new Color(40, 40, 40));
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        for (int m = 0; m < 16; m++) {
            int[] tile = mainBoardRef[m];
            // BUG FIX: mCol must use % 4, not / 4
            int xOffset = (m % 4) * 4 * PIECE_SIZE;
            int yOffset = (m / 4) * 4 * PIECE_SIZE;

            if (tile == null) {
                g2d.setColor(new Color(60, 60, 60));
                g2d.fillRect(xOffset, yOffset, PIECE_SIZE * 4, PIECE_SIZE * 4);
                g2d.setColor(new Color(80, 80, 80));
                g2d.drawRect(xOffset, yOffset, PIECE_SIZE * 4, PIECE_SIZE * 4);
                continue;
            }

            for (int p = 0; p < 16; p++) {
                int pRow = p / 4;
                int pCol = p % 4;
                int x = xOffset + (pCol * PIECE_SIZE);
                int y = yOffset + (pRow * PIECE_SIZE);
                drawPiece(g2d, x, y, tile[p]);
            }
        }
    }

    private void drawPiece(Graphics2D g2d, int x, int y, int piece) {
        int cx = x + (PIECE_SIZE / 2), cy = y + (PIECE_SIZE / 2);
        int n = PieceUtils.getNorth(piece), e = PieceUtils.getEast(piece);
        int s = PieceUtils.getSouth(piece), w = PieceUtils.getWest(piece);

        // North triangle
        g2d.setColor(COLORS.getOrDefault(n, Color.GRAY));
        g2d.fillPolygon(new int[]{x, x + PIECE_SIZE, cx}, new int[]{y, y, cy}, 3);
        // East triangle
        g2d.setColor(COLORS.getOrDefault(e, Color.GRAY));
        g2d.fillPolygon(new int[]{x + PIECE_SIZE, x + PIECE_SIZE, cx}, new int[]{y, y + PIECE_SIZE, cy}, 3);
        // South triangle
        g2d.setColor(COLORS.getOrDefault(s, Color.GRAY));
        g2d.fillPolygon(new int[]{x, x + PIECE_SIZE, cx}, new int[]{y + PIECE_SIZE, y + PIECE_SIZE, cy}, 3);
        // West triangle
        g2d.setColor(COLORS.getOrDefault(w, Color.GRAY));
        g2d.fillPolygon(new int[]{x, x, cx}, new int[]{y, y + PIECE_SIZE, cy}, 3);

        // Grid lines
        g2d.setColor(Color.BLACK);
        g2d.drawRect(x, y, PIECE_SIZE, PIECE_SIZE);
    }
}

/*
package dk.roleplay;

import javax.swing.*;
import java.awt.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class BoardVisualizer extends JPanel {
    private static final int PIECE_SIZE = 50;
    private static final Map<Integer, Color> COLORS = new HashMap<>();

    static {
        COLORS.put(0, Color.BLACK); // Border
        COLORS.put(PieceUtils.WILDCARD, Color.MAGENTA);
        Random r = new Random(123);
        for (int i = 1; i <= 255; i++) {
            COLORS.put(i, new Color(r.nextInt(200) + 55, r.nextInt(200) + 55, r.nextInt(200) + 55));
        }
    }

    private final int[][] mainBoardRef;

    public BoardVisualizer(int[][] mainBoardRef) {
        this.mainBoardRef = mainBoardRef;
        setPreferredSize(new Dimension(16 * PIECE_SIZE, 16 * PIECE_SIZE));
        setBackground(new Color(40, 40, 40));
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;

        for (int m = 0; m < 16; m++) {
            int[] tile = mainBoardRef[m];
            int mRow = m / 4;
            int mCol = m / 4; // FIXED: Should be % 4
            int xOffset = (m % 4) * 4 * PIECE_SIZE;
            int yOffset = (m / 4) * 4 * PIECE_SIZE;

            if (tile == null) {
                g2d.setColor(new Color(60, 60, 60));
                g2d.drawRect(xOffset, yOffset, PIECE_SIZE * 4, PIECE_SIZE * 4);
                continue;
            }

            for (int p = 0; p < 16; p++) {
                int pRow = p / 4;
                int pCol = p % 4;
                int x = xOffset + (pCol * PIECE_SIZE);
                int y = yOffset + (pRow * PIECE_SIZE);
                drawPiece(g2d, x, y, tile[p]);
            }
        }
    }

    private void drawPiece(Graphics2D g2d, int x, int y, int piece) {
        int cx = x + (PIECE_SIZE / 2), cy = y + (PIECE_SIZE / 2);
        int n = PieceUtils.getNorth(piece), e = PieceUtils.getEast(piece);
        int s = PieceUtils.getSouth(piece), w = PieceUtils.getWest(piece);

        g2d.setColor(COLORS.getOrDefault(n, Color.GRAY));
        g2d.fillPolygon(new int[]{x, x + PIECE_SIZE, cx}, new int[]{y, y, cy}, 3);
        g2d.setColor(COLORS.getOrDefault(e, Color.GRAY));
        g2d.fillPolygon(new int[]{x + PIECE_SIZE, x + PIECE_SIZE, cx}, new int[]{y, y + PIECE_SIZE, cy}, 3);
        g2d.setColor(COLORS.getOrDefault(s, Color.GRAY));
        g2d.fillPolygon(new int[]{x, x + PIECE_SIZE, cx}, new int[]{y + PIECE_SIZE, y + PIECE_SIZE, cy}, 3);
        g2d.setColor(COLORS.getOrDefault(w, Color.GRAY));
        g2d.fillPolygon(new int[]{x, x, cx}, new int[]{y, y + PIECE_SIZE, cy}, 3);

        g2d.setColor(Color.BLACK);
        g2d.drawRect(x, y, PIECE_SIZE, PIECE_SIZE);
    }
}
*/
package dk.roleplay;

import javax.swing.*;
import java.awt.*;

public class BoardVisualizer extends JPanel {
    private final int[][] board;
    private final int TILE_SIZE = 45;

    public BoardVisualizer(int[][] board) {
        this.board = board;
        setPreferredSize(new Dimension(16 * TILE_SIZE, 16 * TILE_SIZE));
        setBackground(new Color(40, 40, 40));
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        for (int r = 0; r < 16; r++) {
            if (board[r] == null) continue;
            for (int c = 0; c < 16; c++) {
                int p = board[r][c];
                if (p != -1 && p != PieceUtils.pack(PieceUtils.WILDCARD, PieceUtils.WILDCARD, PieceUtils.WILDCARD, PieceUtils.WILDCARD)) {
                    drawPiece(g2, c * TILE_SIZE, r * TILE_SIZE, p);
                }
            }
        }
    }

    private void drawPiece(Graphics2D g2, int x, int y, int piece) {
        int n = PieceUtils.getNorth(piece);
        int e = PieceUtils.getEast(piece);
        int s = PieceUtils.getSouth(piece);
        int w = PieceUtils.getWest(piece);

        int cx = x + TILE_SIZE / 2;
        int cy = y + TILE_SIZE / 2;

        // Draw solid, clean triangles using the official color palette
        g2.setColor(getBackgroundColor(n));
        g2.fillPolygon(new int[]{x, x + TILE_SIZE, cx}, new int[]{y, y, cy}, 3);

        g2.setColor(getBackgroundColor(e));
        g2.fillPolygon(new int[]{x + TILE_SIZE, x + TILE_SIZE, cx}, new int[]{y, y + TILE_SIZE, cy}, 3);

        g2.setColor(getBackgroundColor(s));
        g2.fillPolygon(new int[]{x, x + TILE_SIZE, cx}, new int[]{y + TILE_SIZE, y + TILE_SIZE, cy}, 3);

        g2.setColor(getBackgroundColor(w));
        g2.fillPolygon(new int[]{x, x, cx}, new int[]{y, y + TILE_SIZE, cy}, 3);

        // Draw the subtle black grid lines
        g2.setColor(new Color(30, 30, 30));
        g2.drawRect(x, y, TILE_SIZE, TILE_SIZE);
        g2.drawLine(x, y, x + TILE_SIZE, y + TILE_SIZE);
        g2.drawLine(x, y + TILE_SIZE, x + TILE_SIZE, y);
    }

    private Color getBackgroundColor(int val) {
        switch (val) {
            case 0: return new Color(100, 105, 110); // Border Grey
            case 1: case 17: return new Color(100, 200, 230); // Light Blue
            case 2: case 6: case 13: return new Color(230, 130, 180); // Light Pink
            case 3: case 9: case 15: return new Color(80, 180, 100); // Green
            case 4: case 12: case 20: return new Color(40, 60, 120); // Dark Blue
            case 5: case 19: return new Color(240, 150, 50); // Orange
            case 7: case 8: return new Color(130, 80, 160); // Purple
            case 10: return new Color(80, 100, 200); // Royal Blue
            case 11: case 14: case 18: return new Color(240, 220, 80); // Yellow
            case 16: case 22: return new Color(160, 100, 80); // Brown
            case 21: return new Color(180, 60, 100); // Dark Pink
            default: return Color.BLACK;
        }
    }
}
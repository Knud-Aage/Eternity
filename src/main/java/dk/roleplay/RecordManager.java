package dk.roleplay;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class RecordManager {
    private static final int PIECE_SIZE = 50;
    private static final Map<Integer, Color> COLORS = new HashMap<>();

    // Keep the exact same color palette as the BoardVisualizer
    static {
        COLORS.put(0, Color.DARK_GRAY);
        COLORS.put(PieceUtils.WILDCARD, Color.MAGENTA);
        Random r = new Random(123);
        for (int i = 1; i <= 255; i++) {
            COLORS.put(i, new Color(r.nextInt(200) + 55, r.nextInt(200) + 55, r.nextInt(200) + 55));
        }
        new File("records").mkdirs(); // Ensure the save folder exists
    }

    public static synchronized void saveRecord(int[][] mainBoard, int macrosPlaced) {
        int piecesCount = macrosPlaced * 16;
        String baseName = "records/Record_" + piecesCount + "Pieces";

        saveImage(mainBoard, baseName + ".png");
        saveText(mainBoard, baseName + ".csv");
        System.out.println(">>> NEW RECORD! Saved Image and Data for " + piecesCount + " pieces.");
    }

    private static void saveImage(int[][] mainBoard, String filename) {
        BufferedImage img = new BufferedImage(16 * PIECE_SIZE, 16 * PIECE_SIZE, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = img.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2d.setColor(new Color(40, 40, 40));
        g2d.fillRect(0, 0, img.getWidth(), img.getHeight());

        for (int m = 0; m < 16; m++) {
            if (mainBoard[m] == null) continue;
            int xOffset = (m % 4) * 4 * PIECE_SIZE;
            int yOffset = (m / 4) * 4 * PIECE_SIZE;

            for (int p = 0; p < 16; p++) {
                int pRow = p / 4;
                int pCol = p % 4;
                drawPiece(g2d, xOffset + (pCol * PIECE_SIZE), yOffset + (pRow * PIECE_SIZE), mainBoard[m][p]);
            }
        }
        g2d.dispose();
        try { ImageIO.write(img, "PNG", new File(filename)); } catch (Exception e) { e.printStackTrace(); }
    }

    private static void drawPiece(Graphics2D g2d, int x, int y, int piece) {
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

    private static void saveText(int[][] mainBoard, String filename) {
        try (PrintWriter out = new PrintWriter(filename)) {
            out.println("MacroIndex,InternalPos,North,East,South,West");
            for (int m = 0; m < 16; m++) {
                if (mainBoard[m] == null) continue;
                for (int p = 0; p < 16; p++) {
                    int piece = mainBoard[m][p];
                    out.printf("%d,%d,%d,%d,%d,%d\n", m, p,
                            PieceUtils.getNorth(piece), PieceUtils.getEast(piece),
                            PieceUtils.getSouth(piece), PieceUtils.getWest(piece));
                }
            }
        } catch (Exception e) { e.printStackTrace(); }
    }
}
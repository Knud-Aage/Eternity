package dk.roleplay;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class RecordManager {
    private static final int PIECE_SIZE = 50;

    // Pre-rotated images: [patternId 1-22][rotation 0=N, 1=E, 2=S, 3=W]
    private static final BufferedImage[][] rotatedImages = new BufferedImage[23][4];

    static {
        new File("records").mkdirs(); // Ensure the save folder exists
        loadImages();
    }

    private static void loadImages() {
        for (int i = 1; i <= 22; i++) {
            try {
                File imgFile = new File("Assets" + File.separator + "pattern" + i + ".png");
                if (imgFile.exists()) {
                    BufferedImage base = ImageIO.read(imgFile);
                    if (base != null) {
                        rotatedImages[i][0] = base;
                        rotatedImages[i][1] = rotateImage(base, 90);
                        rotatedImages[i][2] = rotateImage(base, 180);
                        rotatedImages[i][3] = rotateImage(base, 270);
                    }
                }
            } catch (Exception e) {
                // Silently fallback if an image is missing
            }
        }
    }

    private static BufferedImage rotateImage(BufferedImage img, double angle) {
        int w = img.getWidth();
        int h = img.getHeight();
        BufferedImage rotated = new BufferedImage(w, h, img.getType());
        Graphics2D g2 = rotated.createGraphics();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);

        AffineTransform at = new AffineTransform();
        at.translate(w / 2.0, h / 2.0);
        at.rotate(Math.toRadians(angle));
        at.translate(-w / 2.0, -h / 2.0);

        g2.drawImage(img, at, null);
        g2.dispose();
        return rotated;
    }

    public static synchronized void saveRecord(int[][] mainBoard, int piecesPlaced) {
        String baseName = "records/Record_" + piecesPlaced + "Pieces";

        saveImage(mainBoard, baseName + ".png");
        saveText(mainBoard, baseName + ".csv");
        String now = LocalDateTime.now()
                .format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        System.out.println(now + " >>> NEW RECORD! Saved Image and Data for " + piecesPlaced + " pieces.");
    }

    private static void saveImage(int[][] mainBoard, String filename) {
        BufferedImage img = new BufferedImage(16 * PIECE_SIZE, 16 * PIECE_SIZE, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = img.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);

        g2d.setColor(new Color(30, 30, 30));
        g2d.fillRect(0, 0, img.getWidth(), img.getHeight());

        for (int r = 0; r < 16; r++) {
            if (mainBoard[r] == null) {
                continue;
            }
            for (int c = 0; c < 16; c++) {
                int piece = mainBoard[r][c];
                if (piece != -1 && piece != PieceUtils.pack(PieceUtils.WILDCARD, PieceUtils.WILDCARD,
                        PieceUtils.WILDCARD, PieceUtils.WILDCARD)) {
                    int x = c * PIECE_SIZE;
                    int y = r * PIECE_SIZE;
                    drawPiece(g2d, x, y, piece);
                }
            }
        }
        g2d.dispose();
        try {
            ImageIO.write(img, "PNG", new File(filename));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void drawPiece(Graphics2D g2d, int tileX, int tileY, int piece) {
        int n = PieceUtils.getNorth(piece);
        int e = PieceUtils.getEast(piece);
        int s = PieceUtils.getSouth(piece);
        int w = PieceUtils.getWest(piece);

        int cx = tileX + PIECE_SIZE / 2;
        int cy = tileY + PIECE_SIZE / 2;

        Polygon north = new Polygon(new int[]{tileX, tileX + PIECE_SIZE, cx}, new int[]{tileY, tileY, cy}, 3);
        Polygon east = new Polygon(new int[]{tileX + PIECE_SIZE, tileX + PIECE_SIZE, cx}, new int[]{tileY,
                tileY + PIECE_SIZE, cy}, 3);
        Polygon south = new Polygon(new int[]{tileX, tileX + PIECE_SIZE, cx}, new int[]{tileY + PIECE_SIZE,
                tileY + PIECE_SIZE, cy}, 3);
        Polygon west = new Polygon(new int[]{tileX, tileX, cx}, new int[]{tileY, tileY + PIECE_SIZE, cy}, 3);

        drawPatternTriangle(g2d, n, north, tileX, tileY, PIECE_SIZE, 0); // 0 = North
        drawPatternTriangle(g2d, e, east, tileX, tileY, PIECE_SIZE, 1); // 1 = East
        drawPatternTriangle(g2d, s, south, tileX, tileY, PIECE_SIZE, 2); // 2 = South
        drawPatternTriangle(g2d, w, west, tileX, tileY, PIECE_SIZE, 3); // 3 = West

        // Draw a subtle border around the piece
        g2d.setColor(new Color(30, 30, 30, 100));
        g2d.setStroke(new BasicStroke(1));
        g2d.drawRect(tileX, tileY, PIECE_SIZE, PIECE_SIZE);
        g2d.drawLine(tileX, tileY, tileX + PIECE_SIZE, tileY + PIECE_SIZE);
        g2d.drawLine(tileX, tileY + PIECE_SIZE, tileX + PIECE_SIZE, tileY);
    }

    private static void drawPatternTriangle(Graphics2D g2d, int patternId, Polygon clip, int tileX, int tileY,
                                            int size, int rotIdx) {
        Shape oldClip = g2d.getClip();
        g2d.setClip(clip);

        // Fill with the solid background color first
        g2d.setColor(getFallbackColor(patternId));
        g2d.fill(clip);

        // Draw the image anchored to the EDGE
        if (patternId > 0 && patternId <= 22 && rotatedImages[patternId][rotIdx] != null) {
            int edgeCenterX = 0;
            int edgeCenterY = 0;

            if (rotIdx == 0) {      // North
                edgeCenterX = tileX + size / 2;
                edgeCenterY = tileY;
            } else if (rotIdx == 1) { // East
                edgeCenterX = tileX + size;
                edgeCenterY = tileY + size / 2;
            } else if (rotIdx == 2) { // South
                edgeCenterX = tileX + size / 2;
                edgeCenterY = tileY + size;
            } else if (rotIdx == 3) { // West
                edgeCenterX = tileX;
                edgeCenterY = tileY + size / 2;
            }

            double scaleFactor = 0.6;
            int imgSize = (int) (size * scaleFactor);

            int drawX = edgeCenterX - (imgSize / 2);
            int drawY = edgeCenterY - (imgSize / 2);

            g2d.drawImage(rotatedImages[patternId][rotIdx], drawX, drawY, imgSize, imgSize, null);
        }

        g2d.setClip(oldClip);
    }

    private static void saveText(int[][] mainBoard, String filename) {
        try (PrintWriter out = new PrintWriter(filename)) {
            // Updated to simple Row, Column coordinates
            out.println("Row,Col,North,East,South,West");
            for (int r = 0; r < 16; r++) {
                if (mainBoard[r] == null) {
                    continue;
                }
                for (int c = 0; c < 16; c++) {
                    int piece = mainBoard[r][c];
                    if (piece != -1) {
                        out.printf("%d,%d,%d,%d,%d,%d\n", r, c,
                                PieceUtils.getNorth(piece), PieceUtils.getEast(piece),
                                PieceUtils.getSouth(piece), PieceUtils.getWest(piece));
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static Color getFallbackColor(int val) {
        switch (val) {
            case 0:
                return new Color(100, 105, 110);
            case 1:
            case 17:
                return new Color(100, 200, 230);
            case 2:
            case 6:
            case 13:
                return new Color(230, 130, 180);
            case 3:
            case 9:
            case 15:
                return new Color(80, 180, 100);
            case 4:
            case 12:
            case 20:
                return new Color(40, 60, 120);
            case 5:
            case 19:
                return new Color(240, 150, 50);
            case 7:
            case 8:
                return new Color(130, 80, 160);
            case 10:
                return new Color(80, 100, 200);
            case 11:
            case 14:
            case 18:
                return new Color(240, 220, 80);
            case 16:
            case 22:
                return new Color(160, 100, 80);
            case 21:
                return new Color(180, 60, 100);
            default:
                return new Color(40, 40, 40);
        }
    }
}
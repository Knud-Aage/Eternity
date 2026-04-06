package dk.roleplay;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.File;

public class BoardVisualizer extends JPanel {
    private final int[][] board;
    // Pre-rotated images: [patternId 1-22][rotation 0=N, 1=E, 2=S, 3=W]
    private final BufferedImage[][] rotatedImages = new BufferedImage[23][4];
    private boolean imagesLoaded = false;

    public BoardVisualizer(int[][] board) {
        this.board = board;
        setBackground(new Color(30, 30, 30));
        loadImages();
    }

    private void loadImages() {
        File assetsDir = new File("Assets");
        System.out.println("Looking for assets in: " + assetsDir.getAbsolutePath());
        System.out.println("Assets folder exists: " + assetsDir.exists());

        int loadedCount = 0;
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
                        loadedCount++;
                    }
                } else {
                    System.out.println("  MISSING: " + imgFile.getAbsolutePath());
                }
            } catch (Exception e) {
                System.out.println("  ERROR loading pattern" + i + ".png: " + e.getMessage());
            }
        }
        imagesLoaded = loadedCount > 0;
        System.out.println("Loaded " + loadedCount + "/22 pattern images. Mode: "
                + (imagesLoaded ? "IMAGE" : "FALLBACK COLORS"));
    }

    private BufferedImage rotateImage(BufferedImage src, int degrees) {
        int w = src.getWidth();
        int h = src.getHeight();
        BufferedImage dst = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = dst.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.drawImage(src, AffineTransform.getRotateInstance(
                Math.toRadians(degrees), w / 2.0, h / 2.0), null);
        g.dispose();
        return dst;
    }

    @Override
    public Dimension getPreferredSize() {
        return new Dimension(16 * 50, 16 * 50);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);

        int tileSize = Math.max(4, Math.min((getWidth() - 4) / 16, (getHeight() - 4) / 16));
        int offsetX = (getWidth() - 16 * tileSize) / 2;
        int offsetY = (getHeight() - 16 * tileSize) / 2;

        for (int r = 0; r < 16; r++) {
            if (board[r] == null) {
                continue;
            }
            for (int c = 0; c < 16; c++) {
                int p = board[r][c];
                if (p == -1) {
                    continue;
                }
                drawPiece(g2, offsetX + c * tileSize, offsetY + r * tileSize, tileSize, p);
            }
        }
    }

    private void drawPiece(Graphics2D g2, int x, int y, int size, int piece) {
        int n = PieceUtils.getNorth(piece);
        int e = PieceUtils.getEast(piece);
        int s = PieceUtils.getSouth(piece);
        int w = PieceUtils.getWest(piece);

        int cx = x + size / 2;
        int cy = y + size / 2;

        Polygon nPoly = new Polygon(new int[]{x, x + size, cx}, new int[]{y, y, cy}, 3);
        Polygon ePoly = new Polygon(new int[]{x + size, x + size, cx}, new int[]{y, y + size, cy}, 3);
        Polygon sPoly = new Polygon(new int[]{x, x + size, cx}, new int[]{y + size, y + size, cy}, 3);
        Polygon wPoly = new Polygon(new int[]{x, x, cx}, new int[]{y, y + size, cy}, 3);

        // KEY FIX: draw the full tile-sized pre-rotated image, clipped to the triangle.
        // No runtime rotation or translate needed — pre-rotation handles orientation,
        // and the polygon clip restricts visibility to just this triangle's area.
        drawTriangle(g2, n, nPoly, x, y, size, 0); // North: image as-is      (0°)
        drawTriangle(g2, e, ePoly, x, y, size, 1); // East:  image rotated 90°
        drawTriangle(g2, s, sPoly, x, y, size, 2); // South: image rotated 180°
        drawTriangle(g2, w, wPoly, x, y, size, 3); // West:  image rotated 270°

        // Grid lines
        g2.setColor(new Color(0, 0, 0, 120));
        g2.setStroke(new BasicStroke(1f));
        g2.drawRect(x, y, size, size);
        g2.setColor(new Color(0, 0, 0, 50));
        g2.drawLine(x, y, x + size, y + size);
        g2.drawLine(x, y + size, x + size, y);
    }

    private void drawTriangle(Graphics2D g2, int patternId, Polygon clip,
                              int tileX, int tileY, int size, int rotIdx) {
        Shape oldClip = g2.getClip();
        g2.setClip(clip);

        if (imagesLoaded && patternId >= 1 && patternId <= 22
                && rotatedImages[patternId][rotIdx] != null) {
            // Draw the full pre-rotated image over the whole tile square.
            // The clip polygon makes only this triangle's portion visible.
            g2.drawImage(rotatedImages[patternId][rotIdx], tileX, tileY, size, size, null);
        } else {
            g2.setColor(getFallbackColor(patternId));
            g2.fill(clip);
        }

        g2.setClip(oldClip);
    }

    private Color getFallbackColor(int val) {
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
                return Color.DARK_GRAY;
        }
    }
}
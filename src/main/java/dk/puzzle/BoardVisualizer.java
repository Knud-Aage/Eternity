package dk.puzzle;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.File;

/**
 * A graphical component responsible for rendering the Eternity II puzzle board.
 * This panel manages a 16x16 grid of puzzle pieces, handling the rendering of
 * piece patterns through loaded image assets or fallback geometric shapes.
 * It supports dynamic updates to the board state and provides a visual
 * representation of the solver's progress.
 */
public class BoardVisualizer extends JPanel {
    private final int[][] board;
    // Pre-rotated images: [patternId 1-22][rotation 0=N, 1=E, 2=S, 3=W]
    private final BufferedImage[][] rotatedImages = new BufferedImage[23][4];

    /**
     * Constructs a new BoardVisualizer with a reference to the puzzle board.
     * This constructor sets up the initial UI environment, including the background
     * color, and triggers the loading and rotation of pattern images from disk.
     *
     * @param board A 16x16 integer array representing the shared state of the puzzle board.
     *              Each integer represents a bit-packed puzzle piece.
     */
    public BoardVisualizer(int[][] board) {
        this.board = board;
        setBackground(new Color(30, 30, 30));
        loadImages();
    }

    private void loadImages() {
        File assetsDir = new File("Assets");

        for (int i = 1; i <= 22; i++) {
            try {
                // Ensure your files are named exactly "pattern1.png", "pattern2.png", etc. inside the "Assets" folder
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
                System.out.println("Could not load image: pattern" + i + ".png");
            }
        }
    }

    private BufferedImage rotateImage(BufferedImage img, double angle) {
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

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);

        int maxWidth = getWidth() - 40;
        int maxHeight = getHeight() - 40;
        int tileSize = Math.min(maxWidth / 16, maxHeight / 16);

        int offsetX = (getWidth() - (16 * tileSize)) / 2;
        int offsetY = (getHeight() - (16 * tileSize)) / 2;

        for (int r = 0; r < 16; r++) {
            if (board[r] == null) {
                continue;
            }
            for (int c = 0; c < 16; c++) {
                int p = board[r][c];
                if (p != -1 && p != PieceUtils.pack(PieceUtils.WILDCARD, PieceUtils.WILDCARD, PieceUtils.WILDCARD,
                        PieceUtils.WILDCARD)) {
                    int x = offsetX + (c * tileSize);
                    int y = offsetY + (r * tileSize);
                    drawPiece(g2, x, y, tileSize, p);
                }
            }
        }
    }

    private void drawPiece(Graphics2D g2, int tileX, int tileY, int size, int piece) {
        int n = PieceUtils.getNorth(piece);
        int e = PieceUtils.getEast(piece);
        int s = PieceUtils.getSouth(piece);
        int w = PieceUtils.getWest(piece);

        int cx = tileX + size / 2;
        int cy = tileY + size / 2;

        Polygon north = new Polygon(new int[]{tileX, tileX + size, cx}, new int[]{tileY, tileY, cy}, 3);
        Polygon east = new Polygon(new int[]{tileX + size, tileX + size, cx}, new int[]{tileY, tileY + size, cy}, 3);
        Polygon south = new Polygon(new int[]{tileX, tileX + size, cx}, new int[]{tileY + size, tileY + size, cy}, 3);
        Polygon west = new Polygon(new int[]{tileX, tileX, cx}, new int[]{tileY, tileY + size, cy}, 3);

        drawPatternTriangle(g2, n, north, tileX, tileY, size, 0); // 0 = North
        drawPatternTriangle(g2, e, east, tileX, tileY, size, 1); // 1 = East
        drawPatternTriangle(g2, s, south, tileX, tileY, size, 2); // 2 = South
        drawPatternTriangle(g2, w, west, tileX, tileY, size, 3); // 3 = West

        // Draw a subtle border around the piece
        g2.setColor(new Color(30, 30, 30, 100));
        g2.setStroke(new BasicStroke(1));
        g2.drawRect(tileX, tileY, size, size);
        g2.drawLine(tileX, tileY, tileX + size, tileY + size);
        g2.drawLine(tileX, tileY + size, tileX + size, tileY);
    }

    private void drawPatternTriangle(Graphics2D g2, int patternId, Polygon clip, int tileX, int tileY, int size,
                                     int rotIdx) {
        Shape oldClip = g2.getClip();
        g2.setClip(clip);

        // 1. ALWAYS fill the triangle with the solid background color first!
        g2.setColor(getFallbackColor(patternId));
        g2.fill(clip);

        // 2. Draw the image anchored to the EDGE of the piece (not the center!)
        if (patternId > 0 && patternId <= 22 && rotatedImages[patternId][rotIdx] != null) {

            // Calculate the exact midpoint of the outer edge for this specific direction
            int edgeCenterX = 0;
            int edgeCenterY = 0;

            if (rotIdx == 0) {      // North Edge
                edgeCenterX = tileX + size / 2;
                edgeCenterY = tileY;
            } else if (rotIdx == 1) { // East Edge
                edgeCenterX = tileX + size;
                edgeCenterY = tileY + size / 2;
            } else if (rotIdx == 2) { // South Edge
                edgeCenterX = tileX + size / 2;
                edgeCenterY = tileY + size;
            } else if (rotIdx == 3) { // West Edge
                edgeCenterX = tileX;
                edgeCenterY = tileY + size / 2;
            }

            // Shrink the pattern to 60% of the piece size
            double scaleFactor = 0.6;
            int imgSize = (int) (size * scaleFactor);

            // Calculate the top-left coordinate so the image is perfectly centered on the edge
            int drawX = edgeCenterX - (imgSize / 2);
            int drawY = edgeCenterY - (imgSize / 2);

            g2.drawImage(rotatedImages[patternId][rotIdx], drawX, drawY, imgSize, imgSize, null);
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
                return new Color(40, 40, 40);
        }
    }
}
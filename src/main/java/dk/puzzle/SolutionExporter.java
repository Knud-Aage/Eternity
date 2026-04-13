package dk.puzzle;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * Provides functionality to export the puzzle board state to external files.
 * This class generates a formatted text map showing the bit-packed representation
 * of pieces and a PNG image that visualizes the board using the {@link BoardVisualizer}.
 */
public class SolutionExporter {
    /**
     * Saves the current board state to both a text file (.txt) and an image file (.png).
     *
     * @param board    The 16x16 board state represented as a 2D array of macro-tiles.
     * @param baseName The base filename (without extension) to use for the exported files.
     */
    public static void save(int[][] board, String baseName) {
        saveTextMap(board, baseName + ".txt");
        saveImage(board, baseName + ".png");
    }

    private static void saveTextMap(int[][] board, String filename) {
        try (PrintWriter pw = new PrintWriter(new FileWriter(filename))) {
            pw.println("Eternity II Solution Map");
            for (
                    int mRow = 0;
                    mRow < 4;
                    mRow++
            ) {
                for (
                        int pRow = 0;
                        pRow < 4;
                        pRow++
                ) {
                    for (
                            int mCol = 0;
                            mCol < 4;
                            mCol++
                    ) {
                        int[] tile = board[mRow * 4 + mCol];
                        for (
                                int pCol = 0;
                                pCol < 4;
                                pCol++
                        ) {
                            int piece = (tile != null) ? tile[pRow * 4 + pCol] : -1;
                            pw.printf("[%08X] ", piece);
                        }
                        pw.print(" | ");
                    }
                    pw.println();
                }
                pw.println("------------------------------------------------------------------");
            }
            System.out.println("Text map saved to " + filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void saveImage(int[][] board, String filename) {
        int pieceSize = 100;
        int boardSize = 16 * pieceSize;
        BufferedImage img = new BufferedImage(boardSize, boardSize, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = img.createGraphics();

        BoardVisualizer viz = new BoardVisualizer(board);
        viz.setSize(boardSize, boardSize);
        viz.paint(g2d);

        g2d.dispose();
        try {
            ImageIO.write(img, "png", new File(filename));
            System.out.println("Solution image saved to " + filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

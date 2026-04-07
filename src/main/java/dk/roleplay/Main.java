package dk.roleplay;

import javax.swing.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Main entry point for the Eternity II Solver application.
 * Orchestrates piece loading, hardware initialization, and solver execution.
 */
public class Main {
    private static final int CENTER_PIECE_INDEX = 138;
    private static final int CENTER_PIECE_ROTATION = 2;

    private static final AtomicInteger currentScore = new AtomicInteger(0);
    private static final int[][] currentDisplayBoard = new int[16][];

    /**
     * Application main method.
     * 
     * @param args Command line arguments
     */
    public static void main(String[] args) {
        StartupDialog dialog = new StartupDialog(null);
        dialog.setVisible(true);

        if (!dialog.isStartClicked()) {
            System.out.println("Setup cancelled. Exiting...");
            System.exit(0);
        }

        System.out.println("Loading Eternity II Engine...");

        int[] basePieces = loadPieces();
        PieceInventory inventory = new PieceInventory(basePieces);

        Runnable solverTask;

        System.out.println("Strategy: Piece-by-Piece (Linear).");

        int targetPiece = basePieces[CENTER_PIECE_INDEX];
        for (int r = 0; r < CENTER_PIECE_ROTATION; r++) {
            int n = PieceUtils.getNorth(targetPiece);
            int e = PieceUtils.getEast(targetPiece);
            int s = PieceUtils.getSouth(targetPiece);
            int w = PieceUtils.getWest(targetPiece);
            targetPiece = PieceUtils.pack(w, n, e, s);
        }

        solverTask = new MasterSolverPBP(inventory, targetPiece, dialog.isUseGpu());

        if (dialog.isUseGpu()) {
            System.out.println("Hardware: GPU CUDA Handoff Enabled!");
        } else {
            System.out.println("Hardware: Running purely natively on CPU.");
        }

        Thread solverThread = new Thread(solverTask, "SolverThread");
        solverThread.start();

        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Eternity II Solver");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setExtendedState(JFrame.MAXIMIZED_BOTH);

            BoardVisualizer viz = new BoardVisualizer(currentDisplayBoard);
            frame.add(viz);
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);

            new javax.swing.Timer(100, e -> {
                viz.repaint();
                int score = currentScore.get();
                String status = score == 0 ? " [SEARCHING...]" : (score == 256 ? " [SOLVED!]" : " [SOLVING...]");
                frame.setTitle("Eternity II - " + score + "/256 Pieces" + status);
            }).start();
        });
    }

    /**
     * Updates the current display board and score.
     * 
     * @param score The new best score (number of pieces placed)
     * @param board The current state of the board
     */
    public static synchronized void updateDisplay(int score, int[][] board) {
        currentScore.set(score);
        for (int i = 0; i < 16; i++) {
            if (board[i] != null) {
                currentDisplayBoard[i] = board[i].clone();
            } else {
                currentDisplayBoard[i] = null;
            }
        }
    }

    private static int[] loadPieces() {
        try (BufferedReader br = new BufferedReader(new FileReader("pieces.csv"))) {
            int[] pieces = new int[256];
            int i = 0;
            String line;
            while ((line = br.readLine()) != null && i < 256) {
                String[] pts = line.split(",");
                if (pts.length < 4) {
                    continue;
                }

                int e = Integer.parseInt(pts[0].trim());
                int s = Integer.parseInt(pts[1].trim());
                int w = Integer.parseInt(pts[2].trim());
                int n = Integer.parseInt(pts[3].trim());

                pieces[i++] = PieceUtils.pack(n, e, s, w);
            }
            if (i == 256) {
                return pieces;
            }
        } catch (Exception e) {
            System.out.println("pieces.csv not found or unreadable. Using mock data.");
        }
        return generateMock();
    }

    private static int[] generateMock() {
        int DIM = 16;
        int[][] hEdges = new int[DIM][DIM + 1];
        int[][] vEdges = new int[DIM + 1][DIM];
        Random rnd = new Random(99);
        int INTERIOR_COLORS = 4;
        for (int r = 0; r < DIM; r++) for (int c = 1; c < DIM; c++) hEdges[r][c] = rnd.nextInt(INTERIOR_COLORS) + 1;
        for (int r = 1; r < DIM; r++) for (int c = 0; c < DIM; c++) vEdges[r][c] = rnd.nextInt(INTERIOR_COLORS) + 1;
        int[] pieces = new int[256];
        int idx = 0;
        for (int r = 0; r < DIM; r++) {
            for (int c = 0; c < DIM; c++) {
                pieces[idx++] = PieceUtils.pack(vEdges[r][c], hEdges[r][c + 1], vEdges[r + 1][c], hEdges[r][c]);
            }
        }
        return pieces;
    }
}

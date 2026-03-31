package dk.roleplay;

import javax.swing.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

public class Main {
    private static final AtomicInteger currentScore = new AtomicInteger(0);
    private static final int[][] currentDisplayBoard = new int[16][];

    public static void main(String[] args) {
        int[] basePieces = loadPieces();
        PieceInventory inventory = new PieceInventory(basePieces);
        CandidateValidator validator;

        try {
            validator = new GpuValidator();
            System.out.println("GPU Validator Initialized.");
        } catch (Throwable e) {
            System.out.println("GPU unavailable (" + e.getMessage() + "). Using CPU Validator.");
            validator = new CpuValidator();
        }

        MasterSolver solver = new MasterSolver(inventory, validator, currentScore, new Random().nextLong());
        Thread solverThread = new Thread(solver, "SolverThread");
        solverThread.setDaemon(true);
        solverThread.start();

        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Eternity II Solver");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
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
                if (pts.length < 4) continue;

                // Inside Main.java -> loadPieces()
                int n = Integer.parseInt(pts[0].trim());
                int e = Integer.parseInt(pts[1].trim());
                int s = Integer.parseInt(pts[2].trim());
                int w = Integer.parseInt(pts[3].trim());

                pieces[i++] = PieceUtils.pack(n, e, s, w);

                // Pack them in the correct Clockwise (N, E, S, W) order for the bitwise engine
                pieces[i++] = PieceUtils.pack(n, e, s, w);
            }
            if (i == 256) {
                System.out.println("Loaded 256 pieces from pieces.csv");
                return pieces;
            }
            System.out.println("pieces.csv found but only had " + i + " valid lines. Falling back to mock data.");
        } catch (Exception e) {
            System.out.println("pieces.csv not found or unreadable. Using mock data.");
        }
        return generateMock();
    }

    private static int[] generateMock() {
        System.out.println("Generating solvable mock puzzle data...");
        int DIM = 16;
        int[][] hEdges = new int[DIM][DIM + 1];
        int[][] vEdges = new int[DIM + 1][DIM];
        Random rnd = new Random(99);
        int INTERIOR_COLORS = 4;

        for (int r = 0; r < DIM; r++) {
            for (int c = 1; c < DIM; c++) hEdges[r][c] = rnd.nextInt(INTERIOR_COLORS) + 1;
        }
        for (int r = 1; r < DIM; r++) {
            for (int c = 0; c < DIM; c++) vEdges[r][c] = rnd.nextInt(INTERIOR_COLORS) + 1;
        }

        int[] pieces = new int[256];
        int idx = 0;
        for (int r = 0; r < DIM; r++) {
            for (int c = 0; c < DIM; c++) {
                int north = vEdges[r][c];
                int south = vEdges[r + 1][c];
                int west  = hEdges[r][c];
                int east  = hEdges[r][c + 1];
                pieces[idx++] = PieceUtils.pack(north, east, south, west);
            }
        }

        for (int i = 255; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int tmp = pieces[i];
            pieces[i] = pieces[j];
            pieces[j] = tmp;
        }
        return pieces;
    }
}

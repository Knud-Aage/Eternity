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
        // --- 1. SHOW THE STARTUP GUI ---
        // (If you already have a JFrame for your visualizer, you can pass it here instead of null)
        StartupDialog dialog = new StartupDialog(null);
        dialog.setVisible(true); // This pauses the program until the user clicks Start

        // If the user closed the window without clicking start, exit safely
        if (!dialog.isStartClicked()) {
            System.out.println("Setup cancelled. Exiting...");
            System.exit(0);
        }

        System.out.println("Loading Eternity II Engine...");


        // --- 2. INITIALIZE INVENTORY ---
        int[] basePieces = loadPieces();
        PieceInventory inventory = new PieceInventory(basePieces);
        // (Make sure you load your pieces.csv here just like you always do)

        // --- 3. HARDWARE SELECTION ---
        CandidateValidator validator;
        if (dialog.isUseGpu()) {
            System.out.println("Hardware: GPU Validator selected.");
            validator = new GpuValidator(); // Your CUDA inspector
        } else {
            System.out.println("Hardware: CPU Validator selected.");
            validator = new CpuValidator(); // Your instant Pass-Through CPU
        }

        // --- 4. STRATEGY SELECTION & LAUNCH ---
        Runnable solverTask;
        if (dialog.isUsePbp()) {
            System.out.println("Strategy: Piece-by-Piece (Linear).");
            solverTask = new MasterSolverPBP(inventory);

            if (dialog.isUseGpu()) {
                System.out.println("Hardware: GPU CUDA Handoff Enabled!");
            } else {
                System.out.println("Hardware: Running purely natively on CPU.");
            }
        } else {
            System.out.println("Strategy: Divide & Conquer (Outward Spiral).");
            AtomicInteger scoreRef = new AtomicInteger(0);
            solverTask = new MasterSolver(inventory, validator, scoreRef, System.currentTimeMillis());
        }

        // --- 5. START THE THREAD ---
        Thread solverThread = new Thread(solverTask, "SolverThread");
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
                if (pts.length < 4) {
                    continue;
                }

                // CRITICAL FIX: The CSV is Top, Bottom, Left, Right
                int n = Integer.parseInt(pts[0].trim()); // Top (North)
                int s = Integer.parseInt(pts[1].trim()); // Bottom (South)
                int w = Integer.parseInt(pts[2].trim()); // Left (West)
                int e = Integer.parseInt(pts[3].trim()); // Right (East)

                // Pack them into our 32-bit integer in the correct Clockwise order
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
                int west = hEdges[r][c];
                int east = hEdges[r][c + 1];
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

package dk.roleplay;

import javax.swing.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

public class Main {
    // =====================================================================
    // --- ETERNITY II CONFIGURATION ZONE ---
    // The exact array index of your center piece (139th piece = Index 138)
    private static final int CENTER_PIECE_INDEX = 138;

    // The official center piece must be oriented correctly.
    // Change this to spin the piece: 0 = As is, 1 = 90° Clockwise, 2 = 180°, 3 = 270°
    private static final int CENTER_PIECE_ROTATION = 1;
    // =====================================================================

    private static final AtomicInteger currentScore = new AtomicInteger(0);
    private static final int[][] currentDisplayBoard = new int[16][];

    public static void main(String[] args) {
        // --- 1. SHOW THE STARTUP GUI ---
        StartupDialog dialog = new StartupDialog(null);
        dialog.setVisible(true);

        if (!dialog.isStartClicked()) {
            System.out.println("Setup cancelled. Exiting...");
            System.exit(0);
        }

        System.out.println("Loading Eternity II Engine...");

        // --- 2. INITIALIZE INVENTORY ---
        int[] basePieces = loadPieces();
        PieceInventory inventory = new PieceInventory(basePieces);

        // --- 3. HARDWARE SELECTION ---
        CandidateValidator validator;
        if (dialog.isUseGpu()) {
            System.out.println("Hardware: GPU Validator selected.");
            validator = new GpuValidator();
        } else {
            System.out.println("Hardware: CPU Validator selected.");
            validator = new CpuValidator();
        }

        // --- 4. STRATEGY SELECTION & LAUNCH ---
        Runnable solverTask;
        if (dialog.isUsePbp()) {
            System.out.println("Strategy: Piece-by-Piece (Linear).");

            // Extract the true center piece and apply any necessary rotations!
            int targetPiece = basePieces[CENTER_PIECE_INDEX];
            for (int r = 0; r < CENTER_PIECE_ROTATION; r++) {
                int n = PieceUtils.getNorth(targetPiece);
                int e = PieceUtils.getEast(targetPiece);
                int s = PieceUtils.getSouth(targetPiece);
                int w = PieceUtils.getWest(targetPiece);
                targetPiece = PieceUtils.pack(w, n, e, s); // Rotate Clockwise
            }

            solverTask = new MasterSolverPBP(inventory, targetPiece);

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

                // --- CRITICAL FIX: NEW CSV FORMAT (East, South, West, North) ---
                int e = Integer.parseInt(pts[0].trim()); // Column 1: East
                int s = Integer.parseInt(pts[1].trim()); // Column 2: South
                int w = Integer.parseInt(pts[2].trim()); // Column 3: West
                int n = Integer.parseInt(pts[3].trim()); // Column 4: North

                // Pack them into our 32-bit integer in the engine's standard (N, E, S, W) order
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
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
        // --- 1. SHOW THE STARTUP GUI ---
        StartupDialog dialog = new StartupDialog(null);
        dialog.setVisible(true);

        // If the user closed the window without clicking start, exit gracefully.
        if (!dialog.isStartClicked()) {
            System.out.println("Startup cancelled. Exiting...");
            System.exit(0);
        }

        System.out.println("Loading Eternity II Engine...");

        // --- 2. INITIALIZE INVENTORY ---
        // We MUST load the pieces before we can start the solver!
        int[] basePieces = loadPieces();
        PieceInventory inventory = new PieceInventory(basePieces);

        // --- 3. PREPARE CENTER PIECE (for PBP) ---
        int targetPiece = basePieces[CENTER_PIECE_INDEX];
        for (int r = 0; r < CENTER_PIECE_ROTATION; r++) {
            int n = PieceUtils.getNorth(targetPiece);
            int e = PieceUtils.getEast(targetPiece);
            int s = PieceUtils.getSouth(targetPiece);
            int w = PieceUtils.getWest(targetPiece);
            targetPiece = PieceUtils.pack(w, n, e, s);
        }

        // --- 4. HARDWARE AND STRATEGY SELECTION ---
        boolean useGpu = dialog.isUseGpu();
        boolean usePbp = dialog.isUsePbp();
        boolean useSpiral = dialog.isUseSpiral();

        Runnable solverTask = null;

        if (usePbp) {
            // Map the boolean from the dialog to the Enum in MasterSolverPBP
            MasterSolverPBP.BuildStrategy strategy = useSpiral ?
                    MasterSolverPBP.BuildStrategy.SPIRAL :
                    MasterSolverPBP.BuildStrategy.TYPEWRITER;

            System.out.println("Hardware: " + (useGpu ? "GPU CUDA Handoff Enabled!" : "Running natively on CPU."));

            // Initialize the PBP solver with the correct strategy
            solverTask = new MasterSolverPBP(inventory, targetPiece, useGpu, strategy);

        } else {
            // --- MACRO SOLVER PATH ---
            System.out.println("Strategy: Macro Solver (Divide & Conquer) Selected.");
            CandidateValidator validator;
            if (useGpu) {
                System.out.println("Hardware: GPU Validator selected.");
                try {
                    validator = new GpuValidator();
                } catch (Throwable e) {
                    System.out.println("GPU unavailable. Falling back to CPU. Error: " + e.getMessage());
                    validator = new CpuValidator();
                }
            } else {
                System.out.println("Hardware: CPU Validator selected.");
                validator = new CpuValidator();
            }

            // NOTE: Add your MasterSolver initialization for the Macro method here if you still use it!
            // solverTask = new MasterSolver(inventory, validator, targetPiece);
            System.out.println("Please ensure Macro solver is linked if you intend to use it.");
        }

        // --- 5. START THE SOLVER THREAD ---
        if (solverTask != null) {
            Thread solverThread = new Thread(solverTask, "SolverThread");
            solverThread.start();
        }

        // --- 6. START THE VISUALIZER GUI ---
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
        int dim = 16;
        int[][] hEdges = new int[dim][dim + 1];
        int[][] vEdges = new int[dim + 1][dim];
        Random rnd = new Random(99);
        int interiorColors = 4;
        for (int r = 0; r < dim; r++) for (int c = 1; c < dim; c++) hEdges[r][c] = rnd.nextInt(interiorColors) + 1;
        for (int r = 1; r < dim; r++) for (int c = 0; c < dim; c++) vEdges[r][c] = rnd.nextInt(interiorColors) + 1;
        int[] pieces = new int[256];
        int idx = 0;
        for (int r = 0; r < dim; r++) {
            for (int c = 0; c < dim; c++) {
                pieces[idx++] = PieceUtils.pack(vEdges[r][c], hEdges[r][c + 1], vEdges[r + 1][c], hEdges[r][c]);
            }
        }
        return pieces;
    }
}
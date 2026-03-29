package dk.roleplay;

import javax.swing.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

public class Main {
    private static final AtomicInteger currentScore = new AtomicInteger(0);
    private static final List<MasterSolver> workers = new ArrayList<>();
    private static final int[][] currentDisplayBoard = new int[16][16];

    public static void main(String[] args) {
        int[] basePieces = loadPieces("pieces.csv");
        if (basePieces == null) {
            return;
        }

        PieceInventory inventory = new PieceInventory(basePieces);
        CandidateValidator validator;

        try {
            validator = new GpuValidator();
            System.out.println("GPU Validator Initialized.");
        } catch (Throwable e) {
            validator = new CpuValidator();
            System.out.println("GPU fail. Using CPU Validator.");
        }

        // Single-thread for visualization test
        int cores = 1;
        for (
                int i = 0;
                i < cores;
                i++
        ) {
            MasterSolver solver = new MasterSolver(inventory, validator, currentScore, 42);
            workers.add(solver);
            new Thread(solver).start();
        }

        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Eternity II Continuous Solving Viewer");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            BoardVisualizer viz = new BoardVisualizer(currentDisplayBoard);
            frame.add(viz);
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);

            new javax.swing.Timer(50, e -> {
                viz.repaint();
                frame.setTitle("Eternity II Solving - Current Piece Count: " + currentScore.get());
            }).start();
        });
    }

    public static synchronized void updateDisplay(int score, int[][] board) {
        currentScore.set(score);
        for (
                int i = 0;
                i < 16;
                i++
        ) {
            if (board[i] != null) {
                currentDisplayBoard[i] = board[i].clone();
            } else {
                currentDisplayBoard[i] = null;
            }
        }
    }

    private static int[] loadPieces(String filename) {
        int[] pieces = new int[256];
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            int i = 0;
            while ((line = br.readLine()) != null && i < 256) {
                String[] pts = line.split(",");
                if (pts.length < 4) {
                    continue;
                }
                pieces[i++] = PieceUtils.pack(Integer.parseInt(pts[0].trim()), Integer.parseInt(pts[1].trim()),
                        Integer.parseInt(pts[2].trim()), Integer.parseInt(pts[3].trim()));
            }
        } catch (Exception e) {
            return generateMock();
        }
        return pieces;
    }

    private static int[] generateMock() {
        int[] pieces = new int[256];
        Random rnd = new Random(42);
        for (
                int i = 0;
                i < 256;
                i++
        ) {
            pieces[i] = PieceUtils.pack(rnd.nextInt(23), rnd.nextInt(23), rnd.nextInt(23), rnd.nextInt(23));
        }
        return pieces;
    }
}

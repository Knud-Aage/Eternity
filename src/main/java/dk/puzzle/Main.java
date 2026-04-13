package dk.puzzle;

import javax.swing.*;
import java.awt.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

public class Main {
    private static final int CENTER_PIECE_INDEX = 138;
    private static final int CENTER_PIECE_ROTATION = 2;

    private static final AtomicInteger currentScore = new AtomicInteger(0);
    private static final int[][] currentDisplayBoard = new int[16][];

    public static void main(String[] args) {
        StartupDialog dialog = new StartupDialog(null);
        dialog.setVisible(true);

        if (!dialog.isStartClicked()) {
            System.exit(0);
        }

        System.out.println("Loading Eternity II Engine...");

        int[] basePieces = loadPieces();
        PieceInventory inventory = new PieceInventory(basePieces);

        int targetPiece = basePieces[CENTER_PIECE_INDEX];
        for (int r = 0; r < CENTER_PIECE_ROTATION; r++) {
            int n = PieceUtils.getNorth(targetPiece);
            int e = PieceUtils.getEast(targetPiece);
            int s = PieceUtils.getSouth(targetPiece);
            int w = PieceUtils.getWest(targetPiece);
            targetPiece = PieceUtils.pack(w, n, e, s);
        }

        boolean useGpu = dialog.isUseGpu();
        boolean usePbp = dialog.isUsePbp();
        boolean useSpiral = dialog.isUseSpiral();
        boolean lockCenter = dialog.isLockCenter();

        Runnable solverTask = null;

        if (usePbp) {
            MasterSolverPBP.BuildStrategy strategy = useSpiral ?
                    MasterSolverPBP.BuildStrategy.SPIRAL :
                    MasterSolverPBP.BuildStrategy.TYPEWRITER;

            solverTask = new MasterSolverPBP(inventory, targetPiece, useGpu, strategy, lockCenter);
        } else {
            System.out.println("Macro solver ikke implementeret i denne main.");
        }

        if (solverTask != null) {
            Thread solverThread = new Thread(solverTask, "SolverThread");
            solverThread.start();
        }

        final Runnable finalSolverTask = solverTask;

        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Eternity II Solver");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setExtendedState(JFrame.MAXIMIZED_BOTH);
            frame.setLayout(new BorderLayout());

            BoardVisualizer viz = new BoardVisualizer(currentDisplayBoard);
            frame.add(viz, BorderLayout.CENTER);

            if (finalSolverTask instanceof MasterSolverPBP pbpSolver) {

                JPanel controlPanel = new JPanel();
                controlPanel.setLayout(new BoxLayout(controlPanel, BoxLayout.Y_AXIS));
                controlPanel.setBorder(BorderFactory.createTitledBorder(
                        BorderFactory.createLineBorder(Color.GRAY),
                        "Director Controls",
                        0, 0, new Font("Arial", Font.BOLD, 14), Color.LIGHT_GRAY));
                controlPanel.setPreferredSize(new Dimension(300, 0));
                controlPanel.setBackground(new Color(40, 42, 45));

                Font labelFont = new Font("Arial", Font.BOLD, 13);
                Color textColor = Color.WHITE;

                // 1. Extinction Threshold Slider
                JLabel extLabel = new JLabel("Extinction Trigger: 98%");
                extLabel.setFont(labelFont);
                extLabel.setForeground(textColor);
                extLabel.setAlignmentX(Component.CENTER_ALIGNMENT);

                JSlider extSlider = new JSlider(50, 100, 98);
                extSlider.setBackground(new Color(40, 42, 45));
                extSlider.addChangeListener(e -> {
                    extLabel.setText("Extinction Trigger: " + extSlider.getValue() + "%");
                    pbpSolver.setExtinctionThreshold(extSlider.getValue() / 100.0);
                });

                // 2. Batch Size Dropdown (NYT)
                JLabel batchLabel = new JLabel("Target Seeds: AUTO (Dynamic)");
                batchLabel.setFont(labelFont);
                batchLabel.setForeground(textColor);
                batchLabel.setAlignmentX(Component.CENTER_ALIGNMENT);

                String[] batchOptions = {"AUTO (Dynamic)", "50", "100", "250", "500", "1000", "2500", "5000", "10000"};
                JComboBox<String> batchBox = new JComboBox<>(batchOptions);
                batchBox.setMaximumSize(new Dimension(150, 30));
                batchBox.setAlignmentX(Component.CENTER_ALIGNMENT);
                batchBox.addActionListener(e -> {
                    String selected = (String) batchBox.getSelectedItem();
                    if (selected.startsWith("AUTO")) {
                        batchLabel.setText("Target Seeds: AUTO (Dynamic)");
                        pbpSolver.setBatchSizeOverride(-1); // -1 = Slå auto til igen
                    } else {
                        batchLabel.setText("Target Seeds: " + selected + " (LOCKED)");
                        pbpSolver.setBatchSizeOverride(Integer.parseInt(selected));
                    }
                });

                // 3. Base Camp Override Slider
                JLabel campLabel = new JLabel("Force Next Base Camp: 70");
                campLabel.setFont(labelFont);
                campLabel.setForeground(textColor);
                campLabel.setAlignmentX(Component.CENTER_ALIGNMENT);

                JSlider campSlider = new JSlider(0, 200, 70);
                campSlider.setBackground(new Color(40, 42, 45));
                campSlider.setMajorTickSpacing(50);
                campSlider.setMinorTickSpacing(10);
                campSlider.setPaintTicks(true);
                campSlider.addChangeListener(e -> campLabel.setText("Force Next Base Camp: " + campSlider.getValue()));

                // 4. Override Button
                JButton forceBtn = new JButton("FORCE JUMP!");
                forceBtn.setFont(new Font("Arial", Font.BOLD, 16));
                forceBtn.setBackground(new Color(200, 50, 50));
                forceBtn.setForeground(Color.WHITE);
                forceBtn.setAlignmentX(Component.CENTER_ALIGNMENT);
                forceBtn.addActionListener(e -> pbpSolver.triggerManualOverride(campSlider.getValue()));

                controlPanel.add(Box.createVerticalStrut(30));
                controlPanel.add(extLabel);
                controlPanel.add(Box.createVerticalStrut(10));
                controlPanel.add(extSlider);

                controlPanel.add(Box.createVerticalStrut(30));

                controlPanel.add(batchLabel);
                controlPanel.add(Box.createVerticalStrut(10));
                controlPanel.add(batchBox);

                controlPanel.add(Box.createVerticalStrut(50));

                controlPanel.add(campLabel);
                controlPanel.add(Box.createVerticalStrut(10));
                controlPanel.add(campSlider);
                controlPanel.add(Box.createVerticalStrut(20));
                controlPanel.add(forceBtn);

                frame.add(controlPanel, BorderLayout.EAST);
            }

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
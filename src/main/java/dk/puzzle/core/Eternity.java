package dk.puzzle.core;

import dk.puzzle.ui.BoardVisualizer;
import dk.puzzle.model.PieceInventory;
import dk.puzzle.util.PieceUtils;
import dk.puzzle.ui.StartupDialog;

import javax.swing.*;
import java.awt.*;
import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.time.LocalDateTime;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * The primary entry point and coordinator for the Eternity II solver application.
 * 
 * <p>This class is responsible for initializing the application environment,
 * loading puzzle piece data, launching the graphical user interface, and
 * managing the lifecycle of the solver threads. It also provides the control
 * interface (Swing) for real-time adjustment of solver parameters.</p>
 */
public class Eternity {
    private static final int CENTER_PIECE_INDEX = 138;
    private static final int CENTER_PIECE_ROTATION = 2;

    public static final AtomicInteger currentScore = new AtomicInteger(0);
    private static final int[][] currentDisplayBoard = new int[16][];
    private static final AtomicInteger highScore = new AtomicInteger(0);
    private static final Logger logger = LogManager.getLogger(Eternity.class);


    /**
     * Orchestrates the startup sequence of the application.
     * 
     * <p>This method initializes logging, displays the {@link StartupDialog}, 
     * loads the piece inventory, configures the center piece, starts the 
     * requested {@link EternitySolver} strategy, and constructs the main GUI frame.</p>
     * 
     * @param args Command-line arguments (currently unused).
     */
    public static void main(String[] args) {
//        initLogging();
        
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

        // Read the checkbox from your startup dialog GUI
//        boolean useOfficialConstraints = myCenterPieceCheckbox.isSelected();

        if (usePbp) {
            EternitySolver.BuildStrategy strategy = useSpiral ?
                    EternitySolver.BuildStrategy.SPIRAL :
                    EternitySolver.BuildStrategy.TYPEWRITER;

            // Pass the checkbox variable in as the very last parameter!
            solverTask = new EternitySolver(inventory, targetPiece, useGpu, strategy, lockCenter);
        }

//        if (usePbp) {
//            EternitySolver.BuildStrategy strategy = useSpiral ?
//                    EternitySolver.BuildStrategy.SPIRAL :
//                    EternitySolver.BuildStrategy.TYPEWRITER;
//
//            solverTask = new EternitySolver(inventory, targetPiece, useGpu, strategy, lockCenter);
//        } else {
//            logger.info("Macro solver not implemented in this main.");
//        }

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

            if (finalSolverTask instanceof EternitySolver pbpSolver) {

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

                JLabel jlabelCpuDepth = new JLabel("CPU Handoff Depth: " + EternitySolver.userCpuHandoffDepth);
                JLabel jlabelSurgeonHoles = new JLabel("Surgeon Holes (LNS): " + EternitySolver.userSurgeonHoles);

                JSlider cpuDepthSlider = new JSlider(JSlider.HORIZONTAL, 2, 20, EternitySolver.userCpuHandoffDepth);
                cpuDepthSlider.setMajorTickSpacing(2);
                cpuDepthSlider.setPaintTicks(true);
                cpuDepthSlider.setPaintLabels(true);

                cpuDepthSlider.addChangeListener(e -> {
                    int value = cpuDepthSlider.getValue();
                    EternitySolver.userCpuHandoffDepth = value;
                    jlabelCpuDepth.setText("CPU Handoff Depth: " + value);
                });

                JSlider surgeonHolesSlider = new JSlider(JSlider.HORIZONTAL, 5, 100, EternitySolver.userSurgeonHoles);
                surgeonHolesSlider.setMajorTickSpacing(20);
                surgeonHolesSlider.setMinorTickSpacing(5);
                surgeonHolesSlider.setPaintTicks(true);
                surgeonHolesSlider.setPaintLabels(true);

                surgeonHolesSlider.addChangeListener(e -> {
                    int value = surgeonHolesSlider.getValue();
                    EternitySolver.userSurgeonHoles = value;
                    jlabelSurgeonHoles.setText("Surgeon Holes (LNS): " + value);
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

                controlPanel.add(Box.createVerticalStrut(50));

                controlPanel.add(campLabel);
                controlPanel.add(Box.createVerticalStrut(10));
                controlPanel.add(campSlider);
                controlPanel.add(Box.createVerticalStrut(20));
                controlPanel.add(forceBtn);

                controlPanel.add(jlabelCpuDepth);
                controlPanel.add(cpuDepthSlider);
                controlPanel.add(jlabelSurgeonHoles);
                controlPanel.add(surgeonHolesSlider);

                JLabel stagLabel = new JLabel("Auto-Reset Timeout: 20 min");
                stagLabel.setFont(labelFont);
                stagLabel.setForeground(textColor);
                stagLabel.setAlignmentX(Component.CENTER_ALIGNMENT);

                JSlider stagSlider = new JSlider(5, 120, 20);
                stagSlider.setBackground(new Color(40, 42, 45));
                stagSlider.setMajorTickSpacing(30);
                stagSlider.setMinorTickSpacing(5);
                stagSlider.setPaintTicks(true);
                stagSlider.addChangeListener(e -> {
                    stagLabel.setText("Auto-Reset Timeout: " + stagSlider.getValue() + " min");
                    pbpSolver.setStagnationLimit(stagSlider.getValue());
                });

                controlPanel.add(Box.createVerticalStrut(30));
                controlPanel.add(stagLabel);
                controlPanel.add(Box.createVerticalStrut(10));
                controlPanel.add(stagSlider);

                // --- NEW: DEEP VARIANT SAVE THRESHOLD SLIDER ---
                JLabel thresholdLabel = new JLabel("Save Variants At Depth: 198");
                thresholdLabel.setFont(labelFont);
                thresholdLabel.setForeground(textColor);
                thresholdLabel.setAlignmentX(Component.CENTER_ALIGNMENT);

                // Range from 190 to 250, default starting at 198
                JSlider thresholdSlider = new JSlider(JSlider.HORIZONTAL, 190, 250, 198);
                thresholdSlider.setBackground(new Color(40, 42, 45));
                thresholdSlider.setMajorTickSpacing(10);
                thresholdSlider.setMinorTickSpacing(2);
                thresholdSlider.setPaintTicks(true);
                thresholdSlider.setPaintLabels(true);

                thresholdSlider.addChangeListener(e -> {
                    int val = thresholdSlider.getValue();
                    thresholdLabel.setText("Save Variants At Depth: " + val);

                    // Only send the command to the solver when the user stops dragging
                    if (!thresholdSlider.getValueIsAdjusting()) {
                        pbpSolver.setVariantSaveThreshold(val);
                    }
                });

                controlPanel.add(Box.createVerticalStrut(30));
                controlPanel.add(thresholdLabel);
                controlPanel.add(Box.createVerticalStrut(10));
                controlPanel.add(thresholdSlider);
                // ------------------------------------------------

                frame.add(controlPanel, BorderLayout.EAST);
                }

            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);

            new javax.swing.Timer(100, e -> {
                viz.repaint();
                int current = currentScore.get();
                int record = highScore.get();

                String status = current == 0 ? " [SEARCHING...]" : (current == 256 ? " [SOLVED!]" : " [WORKING...]");
                frame.setTitle(String.format("Eternity II - Current: %d | Record: %d | Total: 256 pieces %s", current, record, status));
            }).start();
        });
    }

    /**
     * Updates the shared state used for real-time visualization of the puzzle board.
     * 
     * <p>This method is thread-safe, allowing solver threads to push updates to the 
     * UI without causing race conditions on the display board data.</p>
     * 
     * @param current The number of pieces placed in the current iteration.
     * @param record The highest number of pieces successfully placed during this session.
     * @param board A 2D array representing the 16x16 board state to be rendered.
     */
    public static synchronized void updateDisplay(int current, int record, int[][] board) {
        currentScore.set(current);
        highScore.set(record);
        for (int i = 0; i < 16; i++) {
            if (board[i] != null) {
                currentDisplayBoard[i] = board[i].clone();
            } else {
                currentDisplayBoard[i] = null;
            }
        }
    }

    public static int[] loadPieces() {
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
            logger.info("pieces.csv not found or unreadable. Using mock data.");
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

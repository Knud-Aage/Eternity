package dk.puzzle.ui;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;

/**
 * A modal configuration dialog presented at application startup.
 * 
 * <p>This dialog allows the user to configure the solving strategy, the specific 
 * grid build order (e.g., Spiral vs. Typewriter), and select the hardware 
 * validator (e.g., GPU-accelerated CUDA or CPU-only).</p>
 */
public class StartupDialog extends JDialog {
    private final JComboBox<String> strategyBox;
    private final JComboBox<String> buildOrderBox; // <--- NEW: For Layered vs Spiral
    private final JComboBox<String> hardwareBox;
    private final JCheckBox lockCenterBox;

    private boolean lockCenter = true;
    private boolean startClicked = false;
    private boolean usePbp = false;
    private boolean useGpu = false;
    private boolean useSpiral = false; // <--- NEW: Stores the build order choice

    /**
     * Constructs a new {@code StartupDialog} instance.
     *
     * @param parent The parent {@link JFrame} to which this modal dialog is attached,
     *               providing a reference for centering and modality.
     */
    public StartupDialog(JFrame parent) {
        super(parent, "Eternity II Engine Setup", true);
        setLayout(new BorderLayout(10, 10));
        setSize(400, 230); // Made slightly taller to fit the 3rd row
        setLocationRelativeTo(parent);

        // --- Create UI Elements ---
        // Changed to 3 rows to accommodate the new dropdown
        JPanel panel = new JPanel(new GridLayout(4, 2, 10, 10));
        panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // 1. Solving Strategy
        panel.add(new JLabel("Solving Strategy:"));
        strategyBox = new JComboBox<>(new String[]{
                "Piece-by-Piece"
        });
        panel.add(strategyBox);

        // 2. Build Order (NEW)
        panel.add(new JLabel("PBP Build Order:"));
        buildOrderBox = new JComboBox<>(new String[]{
                "Layered (Typewriter)",
                "Spiral (Advanced)"
        });
        panel.add(buildOrderBox);

        // 3. Hardware Validator
        panel.add(new JLabel("Hardware Validator:"));
        hardwareBox = new JComboBox<>(new String[]{
                "GPU Accelerated (CUDA)",
                "CPU Only"
        });
        panel.add(hardwareBox);
        panel.add(new JLabel("Game Rules:"));
        lockCenterBox = new JCheckBox("Lock Official Center Piece (Hard)", true);
        panel.add(lockCenterBox);

        add(panel, BorderLayout.CENTER);

        // --- Start Button ---
        JButton startBtn = new JButton("Start Engine");
        startBtn.setFont(new Font("Arial", Font.BOLD, 14));
        startBtn.addActionListener((ActionEvent e) -> {
            usePbp = strategyBox.getSelectedIndex() == 0;
            useSpiral = buildOrderBox.getSelectedIndex() == 1; // Capture the spiral choice
            lockCenter = lockCenterBox.isSelected();
            useGpu = hardwareBox.getSelectedIndex() == 0;
            startClicked = true;
            setVisible(false); // Close the dialog
        });

        JPanel btnPanel = new JPanel();
        btnPanel.add(startBtn);
        add(btnPanel, BorderLayout.SOUTH);
    }

    /**
     * Checks if the user confirmed the settings by clicking the "Start Engine" button.
     * 
     * @return {@code true} if the user clicked "Start Engine"; {@code false} otherwise.
     */
    public boolean isStartClicked() {
        return startClicked;
    }

    /**
     * Determines whether the Piece-by-Piece strategy was selected.
     * 
     * @return {@code true} if Piece-by-Piece is selected; {@code false} otherwise.
     */
    public boolean isUsePbp() {
        return usePbp;
    }

    /**
     * Determines whether GPU acceleration (CUDA) was selected.
     * 
     * @return {@code true} if GPU (CUDA) is selected; {@code false} if CPU-only is selected.
     */
    public boolean isUseGpu() {
        return useGpu;
    }

    /**
     * Determines whether the Spiral build order was selected.
     * 
     * @return {@code true} if Spiral is selected; {@code false} if Layered (Typewriter) is selected.
     */
    public boolean isUseSpiral() {
        return useSpiral;
    }

    /**
     * Determines whether the official center piece should be locked in its predefined position.
     * 
     * @return {@code true} if the center piece is locked; {@code false} if it can move.
     */
    public boolean isLockCenter() {
        return lockCenter;
    }
}
package dk.roleplay;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;

/**
 * A modal configuration dialog presented at application startup.
 * This dialog allows the user to configure the solving strategy (e.g., Piece-by-Piece or Macro-tiles),
 * the specific build order, and select the hardware validator (e.g., GPU-accelerated CUDA or CPU-only).
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
     * Constructs a new StartupDialog.
     *
     * @param parent The parent JFrame to which this modal dialog is attached.
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
                "Piece-by-Piece",
                "Divide & Conquer (4x4 Macros)"
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
     */
    public boolean isStartClicked() {
        return startClicked;
    }

    /**
     * Determines whether the Piece-by-Piece strategy was selected.
     */
    public boolean isUsePbp() {
        return usePbp;
    }

    /**
     * Determines whether GPU acceleration (CUDA) was selected.
     */
    public boolean isUseGpu() {
        return useGpu;
    }

    /**
     * Determines whether the Spiral build order was selected.
     * * @return true if Spiral is selected; false if Layered (Typewriter) is selected.
     */
    public boolean isUseSpiral() {
        return useSpiral;
    }

    public boolean isLockCenter() {
        return lockCenter;
    }
}
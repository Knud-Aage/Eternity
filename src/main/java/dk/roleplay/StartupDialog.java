package dk.roleplay;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;

/**
 * A modal configuration dialog presented at application startup.
 * This dialog allows the user to configure the solving strategy (e.g., Piece-by-Piece or Macro-tiles)
 * and select the hardware validator (e.g., GPU-accelerated CUDA or CPU-only).
 */
public class StartupDialog extends JDialog {
    private final JComboBox<String> strategyBox;
    private final JComboBox<String> hardwareBox;
    private boolean startClicked = false;
    private boolean usePbp = false;
    private boolean useGpu = false;

    /**
     * Constructs a new StartupDialog.
     *
     * @param parent The parent JFrame to which this modal dialog is attached.
     */
    public StartupDialog(JFrame parent) {
        super(parent, "Eternity II Engine Setup", true);
        setLayout(new BorderLayout(10, 10));
        setSize(350, 200);
        setLocationRelativeTo(parent);

        // --- Create UI Elements ---
        JPanel panel = new JPanel(new GridLayout(2, 2, 10, 10));
        panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        panel.add(new JLabel("Solving Strategy:"));
        strategyBox = new JComboBox<>(new String[]{
                "Piece-by-Piece (Linear)",
                "Divide & Conquer (4x4 Macros)"
        });
        panel.add(strategyBox);

        panel.add(new JLabel("Hardware Validator:"));
        hardwareBox = new JComboBox<>(new String[]{
                "GPU Accelerated (CUDA)",
                "CPU Only"
        });
        panel.add(hardwareBox);

        add(panel, BorderLayout.CENTER);

        // --- Start Button ---
        JButton startBtn = new JButton("Start Engine");
        startBtn.setFont(new Font("Arial", Font.BOLD, 14));
        startBtn.addActionListener((ActionEvent e) -> {
            usePbp = strategyBox.getSelectedIndex() == 0;
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
     * @return true if the start button was clicked; false otherwise.
     */
    public boolean isStartClicked() {
        return startClicked;
    }

    /**
     * Determines whether the Piece-by-Piece strategy was selected.
     *
     * @return true if Piece-by-Piece (Linear) is selected; false if Divide & Conquer (Macros) is selected.
     */
    public boolean isUsePbp() {
        return usePbp;
    }

    /**
     * Determines whether GPU acceleration (CUDA) was selected for validation.
     *
     * @return true if GPU acceleration is enabled; false for CPU-only validation.
     */
    public boolean isUseGpu() {
        return useGpu;
    }
}
package dk.roleplay;

import javax.swing.*;
import java.awt.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * A JPanel component that visualizes the state of the Eternity II puzzle board.
 * It renders the 16x16 grid by iterating through the 4x4 macro-tile structures
 * stored in the main board reference.
 */
public class BoardVisualizer extends JPanel {
    /** The size in pixels for each square piece. */
    private static final int PIECE_SIZE = 40;
    
    /** Reference to the shared board state: 16 macro-tiles, each containing 16 pieces. */
    private final int[][] mainBoardRef;
    
    /** Mapping of color IDs (0-22) to specific AWT Color objects for rendering. */
    private static final Map<Integer, Color> COLORS = new HashMap<>();

    static {
        COLORS.put(0, Color.GRAY); // Border
        Random r = new Random(123);
        for (int i = 1; i <= 22; i++) {
            COLORS.put(i, new Color(r.nextInt(256), r.nextInt(256), r.nextInt(256)));
        }
    }

    /**
     * Constructs a new visualizer panel.
     * 
     * @param mainBoardRef The 2D array representing the 16 macro-tiles on the board.
     */
    public BoardVisualizer(int[][] mainBoardRef) {
        this.mainBoardRef = mainBoardRef;
        setPreferredSize(new Dimension(16 * PIECE_SIZE, 16 * PIECE_SIZE));
        setBackground(Color.BLACK);
    }

    /**
     * Paints the puzzle board by rendering each macro-tile and its constituent pieces.
     * 
     * @param g The Graphics context to use for painting.
     */
    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        for (int m = 0; m < 16; m++) {
            int[] tile = mainBoardRef[m];
            if (tile == null) continue;
            
            int mRow = m / 4;
            int mCol = m % 4;

            for (int p = 0; p < 16; p++) {
                int pRow = p / 4;
                int pCol = p % 4;
                int x = ((mCol * 4) + pCol) * PIECE_SIZE;
                int y = ((mRow * 4) + pRow) * PIECE_SIZE;
                drawPiece(g2d, x, y, tile[p]);
            }
        }
    }

    /**
     * Draws an individual puzzle piece as four colored triangles.
     * 
     * @param g2d   The Graphics2D context.
     * @param x     The x-coordinate of the piece's top-left corner.
     * @param y     The y-coordinate of the piece's top-left corner.
     * @param piece The bit-packed integer representing the piece's 4 edge colors 
     *              (North, East, South, West).
     */
    private void drawPiece(Graphics2D g2d, int x, int y, int piece) {
        if (piece == -1) return; // Wildcard or Empty

        int cx = x + (PIECE_SIZE / 2), cy = y + (PIECE_SIZE / 2);
        int n = PieceUtils.getNorth(piece);
        int e = PieceUtils.getEast(piece);
        int s = PieceUtils.getSouth(piece);
        int w = PieceUtils.getWest(piece);

        g2d.setColor(COLORS.getOrDefault(n, Color.BLACK));
        g2d.fillPolygon(new int[]{x, x + PIECE_SIZE, cx}, new int[]{y, y, cy}, 3);

        g2d.setColor(COLORS.getOrDefault(e, Color.BLACK));
        g2d.fillPolygon(new int[]{x + PIECE_SIZE, x + PIECE_SIZE, cx}, new int[]{y, y + PIECE_SIZE, cy}, 3);

        g2d.setColor(COLORS.getOrDefault(s, Color.BLACK));
        g2d.fillPolygon(new int[]{x, x + PIECE_SIZE, cx}, new int[]{y + PIECE_SIZE, y + PIECE_SIZE, cy}, 3);

        g2d.setColor(COLORS.getOrDefault(w, Color.BLACK));
        g2d.fillPolygon(new int[]{x, x, cx}, new int[]{y, y + PIECE_SIZE, cy}, 3);

        g2d.setColor(Color.BLACK);
        g2d.drawRect(x, y, PIECE_SIZE, PIECE_SIZE);
    }

    /**
     * Creates the main application window and starts a background timer to 
     * refresh the visualization every 100 milliseconds.
     * 
     * @param mainBoard The shared board state to be monitored and rendered.
     */
    public static void createAndShowGUI(int[][] mainBoard) {
        JFrame frame = new JFrame("Eternity II Solver Visualizer");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        BoardVisualizer viz = new BoardVisualizer(mainBoard);
        frame.add(viz);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        new Timer(100, e -> viz.repaint()).start();
    }
}

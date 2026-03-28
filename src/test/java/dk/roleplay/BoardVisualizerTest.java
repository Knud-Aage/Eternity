package dk.roleplay;

import org.junit.jupiter.api.Test;
import java.awt.*;
import java.awt.image.BufferedImage;
import static org.junit.jupiter.api.Assertions.*;

class BoardVisualizerTest {

    @Test
    void testPanelInitialization() {
        int[][] board = new int[16][16];
        BoardVisualizer visualizer = new BoardVisualizer(board);
        
        Dimension size = visualizer.getPreferredSize();
        // 16 pieces * 40 pixels = 640
        assertEquals(640, size.width);
        assertEquals(640, size.height);
        assertEquals(Color.BLACK, visualizer.getBackground());
    }

    @Test
    void testPaintComponentNoCrash() {
        int[][] board = new int[16][16];
        // Populate one piece
        board[0] = new int[16];
        board[0][0] = PieceUtils.pack(1, 2, 3, 4);
        
        BoardVisualizer visualizer = new BoardVisualizer(board);
        visualizer.setSize(640, 640);

        // Create a buffered image to act as the graphics context
        BufferedImage image = new BufferedImage(640, 640, BufferedImage.TYPE_INT_ARGB);
        Graphics g = image.getGraphics();
        
        // Assert that calling paint doesn't throw exceptions
        assertDoesNotThrow(() -> visualizer.paintAll(g));
    }

    @Test
    void testColorMapping() throws Exception {
        // Access private COLORS map via reflection
        java.lang.reflect.Field field = BoardVisualizer.class.getDeclaredField("COLORS");
        field.setAccessible(true);
        java.util.Map<Integer, Color> colors = (java.util.Map<Integer, Color>) field.get(null);
        
        assertTrue(colors.containsKey(0), "Should have border color 0");
        assertEquals(Color.GRAY, colors.get(0));
        assertTrue(colors.size() >= 23, "Should have mapped colors 0 through 22");
    }
}

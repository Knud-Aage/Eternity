package dk.puzzle.io;

import dk.puzzle.util.PieceUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link RecordManager}.
 *
 * <p>{@code saveRecord(...)} builds its output path from a hardcoded
 * {@code "records" + File.separator + strategyName} with no way to redirect
 * it, and {@code uploadToDrive(...)} performs a real Google Drive network
 * call - neither is invoked here. {@code saveImage} and the private
 * {@code saveText}/{@code rotateImage}/{@code getFallbackColor} helpers all
 * accept (or only touch) redirectable/pure inputs, so they are exercised
 * directly, always against a JUnit {@code @TempDir}. {@code drawPiece}/
 * {@code drawPatternTriangle} (per-triangle rendering) are only covered
 * indirectly via {@code saveImage} producing a valid, correctly-sized PNG -
 * asserting on individual pixel colors there would be a fragile test of
 * AWT rendering/antialiasing rather than of this class's own logic.</p>
 */
class RecordManagerTest {

    private int pieceSize() throws Exception {
        Field f = RecordManager.class.getDeclaredField("PIECE_SIZE");
        f.setAccessible(true);
        return f.getInt(null);
    }

    private void invokeSaveText(int[][] mainBoard, String filename) throws Exception {
        Method m = RecordManager.class.getDeclaredMethod("saveText", int[][].class, String.class);
        m.setAccessible(true);
        m.invoke(null, mainBoard, filename);
    }

    private Color invokeGetFallbackColor(int val) throws Exception {
        Method m = RecordManager.class.getDeclaredMethod("getFallbackColor", int.class);
        m.setAccessible(true);
        return (Color) m.invoke(null, val);
    }

    private BufferedImage invokeRotateImage(BufferedImage img, double angle) throws Exception {
        Method m = RecordManager.class.getDeclaredMethod("rotateImage", BufferedImage.class, double.class);
        m.setAccessible(true);
        return (BufferedImage) m.invoke(null, img, angle);
    }

    @Test
    void testSaveImageProducesReadablePngWithExpectedDimensions(@TempDir Path tempDir) throws Exception {
        int[][] mainBoard = new int[16][16];
        for (int[] row : mainBoard) {
            Arrays.fill(row, -1);
        }
        mainBoard[0][0] = PieceUtils.pack(1, 2, 3, 4);
        mainBoard[5][5] = PieceUtils.pack(6, 7, 8, 9);

        Path outFile = tempDir.resolve("board.png");
        RecordManager.saveImage(mainBoard, outFile.toString());

        assertTrue(Files.exists(outFile), "saveImage must create the PNG file at the given path");
        BufferedImage image = ImageIO.read(outFile.toFile());
        assertNotNull(image, "The written file must be a valid, readable PNG");
        int size = pieceSize();
        assertEquals(16 * size, image.getWidth());
        assertEquals(16 * size, image.getHeight());
    }

    @Test
    void testSaveImageToleratesNullRows(@TempDir Path tempDir) {
        int[][] mainBoard = new int[16][];
        mainBoard[0] = new int[16];
        Arrays.fill(mainBoard[0], -1);
        mainBoard[0][0] = PieceUtils.pack(1, 2, 3, 4);
        // rows 1..15 intentionally left null (unsolved board rows)

        Path outFile = tempDir.resolve("sparse.png");

        assertDoesNotThrow(() -> RecordManager.saveImage(mainBoard, outFile.toString()),
                "Null rows (unsolved board rows) must not cause saveImage to throw");
        assertTrue(Files.exists(outFile));
    }

    @Test
    void testSaveTextSkipsEmptyCellsAndNullRows(@TempDir Path tempDir) throws Exception {
        int[][] mainBoard = new int[16][];
        mainBoard[0] = new int[16];
        Arrays.fill(mainBoard[0], -1);
        mainBoard[0][0] = PieceUtils.pack(1, 2, 3, 4);
        mainBoard[0][5] = PieceUtils.pack(9, 8, 7, 6);
        // mainBoard[1..15] stay null

        Path outFile = tempDir.resolve("board.csv");
        invokeSaveText(mainBoard, outFile.toString());

        List<String> lines = Files.readAllLines(outFile);
        assertEquals("Row,Col,North,East,South,West", lines.get(0));
        assertEquals("0,0,1,2,3,4", lines.get(1));
        assertEquals("0,5,9,8,7,6", lines.get(2));
        assertEquals(3, lines.size(),
                "Only the two populated cells should produce data rows; -1 cells and null rows must be skipped");
    }

    @Test
    void testGetFallbackColorMapsKnownPatternGroupsAndDefault() throws Exception {
        assertEquals(new Color(100, 105, 110), invokeGetFallbackColor(0));
        assertEquals(new Color(100, 200, 230), invokeGetFallbackColor(1));
        assertEquals(invokeGetFallbackColor(1), invokeGetFallbackColor(17), "Colors 1 and 17 share the same fallback group");
        assertEquals(new Color(180, 60, 100), invokeGetFallbackColor(21));
        assertEquals(new Color(40, 40, 40), invokeGetFallbackColor(99), "Unmapped values must fall back to the default dark grey");
    }

    @Test
    void testRotateImagePreservesDimensionsAndCenterColor() throws Exception {
        BufferedImage img = new BufferedImage(20, 20, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = img.createGraphics();
        g.setColor(new Color(200, 30, 30, 255));
        g.fillRect(0, 0, 20, 20);
        g.dispose();

        for (double angle : new double[]{90, 180, 270}) {
            BufferedImage rotated = invokeRotateImage(img, angle);
            assertEquals(20, rotated.getWidth(), "Rotation must not change image width");
            assertEquals(20, rotated.getHeight(), "Rotation must not change image height");
            assertEquals(new Color(200, 30, 30, 255).getRGB(), rotated.getRGB(10, 10),
                    "Center pixel of a uniformly colored square must survive rotation about its own center");
        }
    }
}

package dk.puzzle.io;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the private parsing/conversion helpers of
 * {@link JBlackwoodToBucas}, reached via reflection.
 *
 * <p>{@code main(String[])} reads/writes hardcoded real project paths
 * (including {@code records\TYPEWRITER_LOCKED\Record_468Pieces.csv}) and is
 * never invoked here. All file I/O in these tests targets a JUnit
 * {@code @TempDir} instead.</p>
 */
class JBlackwoodToBucasTest {

    private int[][] invokeLoadHisPieces(String filepath) throws Exception {
        Method m = JBlackwoodToBucas.class.getDeclaredMethod("loadHisPieces", String.class);
        m.setAccessible(true);
        return (int[][]) m.invoke(null, filepath);
    }

    private void invokeGenerateCsv(String rawBoard, int[][] piecesDb, String outputPath) throws Exception {
        Method m = JBlackwoodToBucas.class.getDeclaredMethod("generateCsv", String.class, int[][].class, String.class);
        m.setAccessible(true);
        m.invoke(null, rawBoard, piecesDb, outputPath);
    }

    @Test
    void testLoadHisPiecesParsesMatchingLinesAndSkipsOthers(@TempDir Path tempDir) throws Exception {
        Path piecesFile = tempDir.resolve("pieces.txt");
        Files.writeString(piecesFile,
                "PieceNumber = 1, TopSide = 1, RightSide = 2, BottomSide = 3, LeftSide = 4\n" +
                        "// a comment line that does not match the pattern\n" +
                        "PieceNumber = 2, TopSide = 5, RightSide = 6, BottomSide = 7, LeftSide = 8\n");

        int[][] pieces = invokeLoadHisPieces(piecesFile.toString());

        assertArrayEquals(new int[]{1, 2, 3, 4}, pieces[0], "Piece 1 (index 0) must store Top/Right/Bottom/Left as N/E/S/W");
        assertArrayEquals(new int[]{5, 6, 7, 8}, pieces[1]);
        assertArrayEquals(new int[]{0, 0, 0, 0}, pieces[2], "Unmatched slots must remain at default zero values");
    }

    @Test
    void testGenerateCsvAppliesRotationsAndSkipsEmptySlots(@TempDir Path tempDir) throws Exception {
        int[][] piecesDb = new int[256][4];
        piecesDb[0] = new int[]{1, 2, 3, 4}; // physical piece 1
        piecesDb[1] = new int[]{5, 6, 7, 8}; // physical piece 2

        Path outFile = tempDir.resolve("out.csv");
        invokeGenerateCsv("1/0 2/1 ---/- 2/2", piecesDb, outFile.toString());

        List<String> lines = Files.readAllLines(outFile);
        assertEquals("Row,Col,North,East,South,West", lines.get(0));
        assertEquals("0,0,1,2,3,4", lines.get(1), "Rotation 0 leaves the base colors as-is");
        assertEquals("0,1,8,5,6,7", lines.get(2), "Rotation 1 (90 deg CW) maps N=W,E=N,S=E,W=S");
        assertEquals("0,3,7,8,5,6", lines.get(3), "Rotation 2 (180 deg) maps N=S,E=W,S=N,W=E; col 2 (---/-) skipped");
        assertEquals(4, lines.size());
    }

    @Test
    void testGenerateCsvAllEmptySlotsProducesHeaderOnly(@TempDir Path tempDir) throws Exception {
        int[][] piecesDb = new int[256][4];
        Path outFile = tempDir.resolve("empty.csv");

        invokeGenerateCsv("---/- ---/-", piecesDb, outFile.toString());

        List<String> lines = Files.readAllLines(outFile);
        assertEquals(1, lines.size(), "Only the header should be written when every slot is empty");
    }
}

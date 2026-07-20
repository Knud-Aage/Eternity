package dk.puzzle.io;

import dk.puzzle.model.PieceInventory;
import dk.puzzle.util.PieceUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link BoardImporter#generateCsvFromRawText}, the pure
 * conversion logic split out from {@code main(String[])}.
 *
 * <p>{@code main(String[])} reads/writes hardcoded paths under
 * {@code src/main/resources/} and is never invoked here. All file I/O in
 * these tests targets a JUnit {@code @TempDir}, never the project's real
 * resource files.</p>
 */
class BoardImporterTest {

    /**
     * A minimal, synthetic 256-piece inventory (NOT the real puzzle data)
     * used only to exercise {@code getBaseColors} lookups.
     */
    private PieceInventory buildInventory() {
        int[] basePieces = new int[256];
        basePieces[0] = PieceUtils.pack(1, 2, 3, 4); // physicalId 1
        basePieces[1] = PieceUtils.pack(5, 6, 7, 8); // physicalId 2
        for (int i = 2; i < 256; i++) {
            basePieces[i] = PieceUtils.pack(0, 0, 0, 0);
        }
        return new PieceInventory(basePieces);
    }

    @Test
    void testGenerateCsvFromRawTextAppliesRotationsAndSkipsEmptySlots(@TempDir Path tempDir) throws IOException {
        PieceInventory inventory = buildInventory();
        String rawBoardText = "1/0 2/1 ---/- 2/2";
        Path outFile = tempDir.resolve("out.csv");

        new BoardImporter().generateCsvFromRawText(rawBoardText, inventory, outFile.toString());

        List<String> lines = Files.readAllLines(outFile);
        assertEquals("Row,Col,North,East,South,West", lines.get(0));
        assertEquals("0,0,1,2,3,4", lines.get(1), "Rotation 0 must leave base colors unchanged");
        assertEquals("0,1,8,5,6,7", lines.get(2), "Rotation 1 (90 deg CW) must map N=W,E=N,S=E,W=S");
        assertEquals("0,3,7,8,5,6", lines.get(3), "Rotation 2 (180 deg) must map N=S,E=W,S=N,W=E; col 2 (---/-) skipped");
        assertEquals(4, lines.size(), "Only 3 data rows should be written; the empty slot must be skipped entirely");
    }

    @Test
    void testGenerateCsvFromRawTextStopsAtProvidedRowsAndColumns(@TempDir Path tempDir) throws IOException {
        PieceInventory inventory = buildInventory();
        // Only one row / one token supplied - the loop must break, not throw, for the missing rows/cols.
        Path outFile = tempDir.resolve("out2.csv");

        new BoardImporter().generateCsvFromRawText("1/0", inventory, outFile.toString());

        List<String> lines = Files.readAllLines(outFile);
        assertEquals(2, lines.size(), "Header plus exactly one data row for the single supplied token");
        assertEquals("0,0,1,2,3,4", lines.get(1));
    }

    @Test
    void testGenerateCsvFromRawTextAllSlotsEmptyWritesHeaderOnly(@TempDir Path tempDir) throws IOException {
        PieceInventory inventory = buildInventory();
        Path outFile = tempDir.resolve("out3.csv");

        new BoardImporter().generateCsvFromRawText("---/- ---/-", inventory, outFile.toString());

        List<String> lines = Files.readAllLines(outFile);
        assertEquals(1, lines.size(), "Only the header line should be written when every slot is empty");
    }

    @Test
    void testGenerateCsvFromRawTextSwallowsIOExceptionForUnwritableDirectory(@TempDir Path tempDir) {
        PieceInventory inventory = buildInventory();
        Path outFile = tempDir.resolve("missingSubdir").resolve("out.csv");

        assertDoesNotThrow(() ->
                        new BoardImporter().generateCsvFromRawText("1/0", inventory, outFile.toString()),
                "Method must catch its own IOException rather than propagate when the target directory doesn't exist");
        assertFalse(Files.exists(outFile), "No file should be produced when the write failed");
    }
}

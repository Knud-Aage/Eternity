package dk.puzzle.io;

import dk.puzzle.util.PieceUtils;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link BucasExporter#exportBoard(int[])}, which is pure
 * string-building logic (no disk I/O).
 *
 * <p>{@code BucasExporter.main(String[])} reads a hardcoded CSV under
 * {@code records/...} and is intentionally never invoked here - it touches
 * real solver output files, which this test suite must never read, write,
 * or overwrite (see the module-level testing guidelines).</p>
 */
class BucasExporterTest {

    private static final String URL_PREFIX =
            "https://e2.bucas.name/#puzzle=KnudHansen&board_w=16&board_h=16&board_edges=";

    private int[] remapTable() throws Exception {
        Field f = BucasExporter.class.getDeclaredField("THESIL_TO_THOMAS");
        f.setAccessible(true);
        return (int[]) f.get(null);
    }

    private String expectedChars(int[] table, int n, int e, int s, int w) {
        return "" + (char) ('a' + table[n]) + (char) ('a' + table[e])
                + (char) ('a' + table[s]) + (char) ('a' + table[w]);
    }

    @Test
    void testExportBoardEmptyBoardProducesAllAaaa() {
        int[] board = new int[256];
        Arrays.fill(board, -1);

        String result = BucasExporter.exportBoard(board);

        assertTrue(result.startsWith(URL_PREFIX), "Exported URL must use the KnudHansen Bucas prefix");
        assertEquals("a".repeat(1024), result.substring(URL_PREFIX.length()),
                "Every empty (-1) slot must be encoded as 'aaaa'");
    }

    @Test
    void testExportBoardTreatsSurgeonHoleSameAsEmpty() {
        int[] board = new int[256];
        Arrays.fill(board, -2);

        String result = BucasExporter.exportBoard(board);

        assertEquals("a".repeat(1024), result.substring(URL_PREFIX.length()),
                "Surgeon holes (-2) must be encoded identically to empty (-1) slots");
    }

    @Test
    void testExportBoardSinglePieceAppliesColorRemap() throws Exception {
        int[] table = remapTable();
        int[] board = new int[256];
        Arrays.fill(board, -1);
        board[0] = PieceUtils.pack(0, 1, 2, 3);

        String encoded = BucasExporter.exportBoard(board).substring(URL_PREFIX.length());

        assertEquals(expectedChars(table, 0, 1, 2, 3), encoded.substring(0, 4),
                "First piece's 4 chars must reflect the TheSil->Thomas color remap");
        assertEquals("a".repeat(1020), encoded.substring(4), "All other slots must remain empty");
    }

    @Test
    void testExportBoardIsPositionSensitive() throws Exception {
        int[] table = remapTable();
        int[] board = new int[256];
        Arrays.fill(board, -1);
        board[255] = PieceUtils.pack(4, 5, 6, 7);

        String encoded = BucasExporter.exportBoard(board).substring(URL_PREFIX.length());

        assertEquals("a".repeat(1020), encoded.substring(0, 1020), "Only the last slot should be non-empty");
        assertEquals(expectedChars(table, 4, 5, 6, 7), encoded.substring(1020, 1024),
                "Last board index must map to the last 4 characters of the encoded string");
    }

    @Test
    void testExportBoardMultiplePiecesEncodedIndependently() throws Exception {
        int[] table = remapTable();
        int[] board = new int[256];
        Arrays.fill(board, -1);
        board[0] = PieceUtils.pack(1, 2, 3, 4);
        board[1] = PieceUtils.pack(5, 6, 7, 8);
        board[16] = PieceUtils.pack(9, 10, 11, 12);

        String encoded = BucasExporter.exportBoard(board).substring(URL_PREFIX.length());

        assertEquals(expectedChars(table, 1, 2, 3, 4), encoded.substring(0, 4));
        assertEquals(expectedChars(table, 5, 6, 7, 8), encoded.substring(4, 8));
        assertEquals("a".repeat(56), encoded.substring(8, 64), "Slots 2..15 on row 0 must stay empty");
        assertEquals(expectedChars(table, 9, 10, 11, 12), encoded.substring(64, 68),
                "Index 16 (row 1, col 0) must occupy characters 64..67");
    }

    @Test
    void testExportBoardTotalLengthMatchesFullGrid() {
        int[] board = new int[256];
        Arrays.fill(board, -1);

        String result = BucasExporter.exportBoard(board);

        assertEquals(URL_PREFIX.length() + 1024, result.length(),
                "256 pieces * 4 chars each must always produce exactly 1024 encoded characters");
    }
}

package dk.puzzle.tools;

import dk.puzzle.model.PieceInventory;
import dk.puzzle.util.PieceUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;

/**
 * Unit tests for {@link ConflictScanner}. {@code main(String[])} (real
 * directory args, stdout formatting) is intentionally never invoked here;
 * {@link ConflictScanner#scan} is exercised directly against a
 * {@code @TempDir} instead, matching this project's established convention
 * (see HoleSolverTest, JBlackwoodToBucasTest).
 */
class ConflictScannerTest {

    private int invokeParsePieceCount(String filename) throws Exception {
        Method m = ConflictScanner.class.getDeclaredMethod("parsePieceCountFromFilename", String.class);
        m.setAccessible(true);
        return (int) m.invoke(null, filename);
    }

    @Test
    void testParsePieceCountFromFilename() throws Exception {
        assertEquals(247, invokeParsePieceCount("247_6aef0a10907d8aefc50d783c51f278a2_812326.txt"));
        assertEquals(0, invokeParsePieceCount("0_abc_1.txt"));
        assertEquals(-1, invokeParsePieceCount("no-underscore.txt"), "malformed filename must be rejected, not guessed");
        assertEquals(-1, invokeParsePieceCount("abc_123_456.txt"), "non-numeric leading token must be rejected");
    }

    /**
     * Builds a minimal inventory the same way HoleSolverTest does: a Mockito mock
     * with only the first few allOrientations/physicalMapping slots populated, so
     * decoding doesn't require loading the real 256-piece pieces.csv. Every listed
     * piece resolves to a distinct physical id, so decodeBoardAuto's resolved-
     * fraction check never has to fall back on ambiguous tie-breaking.
     */
    private static PieceInventory multiPieceInventory(int... packedPieces) {
        PieceInventory inv = mock(PieceInventory.class);
        inv.allOrientations = new int[1024];
        inv.physicalMapping = new int[1024];
        for (int i = 0; i < packedPieces.length; i++) {
            inv.allOrientations[i] = packedPieces[i];
            inv.physicalMapping[i] = i;
        }
        return inv;
    }

    /**
     * A Blackwood-format board_edges string with exactly one real cell (index 0)
     * and "aaaa" (empty sentinel) everywhere else. Uses raw colours 0 and 1 only,
     * both of which BLACKWOOD_TO_THESIL maps to themselves (identity), so the
     * resulting TheSil piece is exactly pack(rawN,rawE,rawS,rawW) without needing
     * access to HoleSolver's private colour table.
     */
    private static String singleCellBlackwoodLink(int rawN, int rawE, int rawS, int rawW) {
        StringBuilder edges = new StringBuilder();
        edges.append((char) ('a' + rawN)).append((char) ('a' + rawE)).append((char) ('a' + rawS)).append((char) ('a' + rawW));
        edges.append("aaaa".repeat(255));
        return "https://e2.bucas.name/#puzzle=Joshua_Blackwood&board_w=16&board_h=16&board_edges=" + edges + "&motifs_order=jblackwood";
    }

    private static void writeSaveFile(Path dir, String filename, String link) throws IOException {
        Files.writeString(dir.resolve(filename), "some grid text\n\n" + link + "\n");
    }

    @Test
    void testScanFindsConflictsAndSkipsUnparseableFiles(@TempDir Path dir) throws Exception {
        // Cell 0 = (row 0, col 0): North=1 violates the row-0 border requirement
        // (must be BORDER_COLOR=0), West=0 satisfies the col-0 border requirement --
        // exactly one conflict, easy to hand-verify against ConflictReducer's own rules.
        int packedPiece = PieceUtils.pack(1, 1, 0, 0);
        PieceInventory inventory = multiPieceInventory(packedPiece);
        String link = singleCellBlackwoodLink(1, 1, 0, 0);

        writeSaveFile(dir, "5_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_123.txt", link);
        writeSaveFile(dir, "not-a-save-file.txt", "irrelevant content");
        writeSaveFile(dir, "10_noLinkHere_999.txt", "grid text with no link at all");

        List<ConflictScanner.ScanRow> rows = ConflictScanner.scan(dir, inventory);

        assertEquals(1, rows.size(), "only the one well-formed, linked, correctly-named file should produce a row");
        ConflictScanner.ScanRow row = rows.get(0);
        assertEquals("5_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_123.txt", row.filename());
        assertEquals(5, row.pieces());
        assertEquals(1, row.conflicts(), "north border violation at (0,0) is the only expected conflict");
    }

    @Test
    void testScanDistinguishesConflictCountsAcrossFiles(@TempDir Path dir) throws Exception {
        // Both pieces present, each resolving to its own physical id, so this test isn't
        // coupled to decodeBoardAuto's fallback tie-breaking when a piece fails to resolve.
        PieceInventory inventory = multiPieceInventory(PieceUtils.pack(0, 0, 0, 0), PieceUtils.pack(1, 0, 0, 0));

        writeSaveFile(dir, "200_hashone_1.txt", singleCellBlackwoodLink(0, 0, 0, 0)); // 0 conflicts, shallower
        writeSaveFile(dir, "250_hashtwo_2.txt", singleCellBlackwoodLink(1, 0, 0, 0)); // 1 conflict (north), deeper

        List<ConflictScanner.ScanRow> rows = ConflictScanner.scan(dir, inventory);
        rows.sort((a, b) -> Integer.compare(a.conflicts(), b.conflicts()));

        assertEquals(2, rows.size());
        assertEquals(0, rows.get(0).conflicts());
        assertEquals(200, rows.get(0).pieces(), "the shallower, cleaner board must still be reported -- not shadowed by the deeper one");
        assertEquals(1, rows.get(1).conflicts());
        assertEquals(250, rows.get(1).pieces());
    }

    @Test
    void testMinPiecesFiltersOutShallowTriviallyConflictFreeBoards(@TempDir Path dir) throws Exception {
        // Steps 1-200 never allow a break at all, so any board with roughly 190-200 pieces is
        // GUARANTEED 0 conflicts by construction -- trivial, not a genuine achievement. Without
        // a depth filter, "lowest conflict count" would always latch onto one of these and never
        // reflect whether genuinely deep attempts are improving.
        PieceInventory inventory = multiPieceInventory(PieceUtils.pack(0, 0, 0, 0), PieceUtils.pack(1, 0, 0, 0));

        writeSaveFile(dir, "195_shallow_1.txt", singleCellBlackwoodLink(0, 0, 0, 0)); // trivially 0 conflicts, shallow
        writeSaveFile(dir, "245_deep_2.txt", singleCellBlackwoodLink(1, 0, 0, 0)); // 1 conflict, but genuinely deep

        List<ConflictScanner.ScanRow> unfiltered = ConflictScanner.scan(dir, inventory, 0);
        assertEquals(2, unfiltered.size(), "no filter must still see both boards");

        List<ConflictScanner.ScanRow> filtered = ConflictScanner.scan(dir, inventory, 240);
        assertEquals(1, filtered.size(), "the shallow trivially-conflict-free board must be excluded");
        assertEquals(245, filtered.get(0).pieces());
        assertEquals(1, filtered.get(0).conflicts());
    }

    @Test
    void testHolesIsDerivedFromBoardSize() {
        var row = new ConflictScanner.ScanRow("x.txt", 206, 0);
        assertEquals(50, row.holes(), "256 - 206 placed pieces = 50 empty cells");
    }
}

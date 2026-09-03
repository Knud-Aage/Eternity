package dk.puzzle.tools;

import dk.puzzle.ai.ConflictAnnealer;
import dk.puzzle.io.BucasExporter;
import dk.puzzle.model.PieceInventory;
import dk.puzzle.util.PieceUtils;
import org.junit.jupiter.api.Test;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;

/**
 * Unit tests for HoleSolver's reusable logic.
 *
 * <p>{@code main(String[])} is CLI argv parsing plus writing
 * {@code physical_layout.txt} to the working directory and is intentionally
 * never invoked here (per project rules: no main(), no real-file I/O). The
 * private {@code writePhysicalLayoutFile} helper is skipped for the same
 * reason (hardcoded relative file path, no injectable seam); the pure
 * {@code writePhysicalLayout(PrintWriter, ...)} it delegates to IS covered
 * below, using an in-memory {@link StringWriter} so no file is touched.</p>
 *
 * <p>{@link HoleSolver#solveConflicts} is covered for its two
 * deterministic, cheap-to-verify outcomes: the already-solved short-circuit,
 * and a tiny hand-crafted 3-cell conflict region that the exact
 * backtracking search can clear near-instantly. The heuristic MCV-fallback
 * branch (triggered only when the exact search proves a region truly has no
 * zero-conflict rearrangement) is NOT covered: forcing genuine
 * unsatisfiability deterministically on a small fixture — without reading
 * real piece data — would require fragile, hard-to-verify engineering of
 * exact color collisions, so per this task's guidance to prefer bounded
 * deterministic helpers over open-ended search plumbing, it's left
 * untested. The private backtracking search internals (RegionSolver) are
 * exercised indirectly through the public solveConflicts entry point rather
 * than reflectively, since they have no other externally meaningful
 * contract.</p>
 */
class HoleSolverTest {

    // Colors used to build small self-consistent board fixtures. Both are
    // within CompatibilityIndex's legal palette (0 = border, 1-22 usable).
    private static final int FILLER = 1;
    private static final int SPECIAL = 9;

    /**
     * A fully self-consistent, conflict-free 16x16 board: every interior-
     * facing edge is FILLER, every board-facing edge is BORDER_COLOR. Every
     * adjacent pair of cells therefore matches trivially.
     */
    private static int[] buildUniformBoard() {
        int[] board = new int[256];
        for (int row = 0; row < 16; row++) {
            for (int col = 0; col < 16; col++) {
                int n = row == 0 ? PieceUtils.BORDER_COLOR : FILLER;
                int e = col == 15 ? PieceUtils.BORDER_COLOR : FILLER;
                int s = row == 15 ? PieceUtils.BORDER_COLOR : FILLER;
                int w = col == 0 ? PieceUtils.BORDER_COLOR : FILLER;
                board[row * 16 + col] = PieceUtils.pack(n, e, s, w);
            }
        }
        return board;
    }

    /**
     * Same as {@link #buildUniformBoard()} but cells 100 (row6,col4) and 101
     * (row6,col5) — both deep interior, away from any border — carry a
     * unique SPECIAL-colored interlock on their shared edge instead of
     * FILLER, so they can be told apart from the rest of the board.
     */
    private static int[] buildBoardWithSignatureRegion() {
        int[] board = buildUniformBoard();
        board[100] = PieceUtils.pack(FILLER, SPECIAL, FILLER, FILLER); // East facing 101
        board[101] = PieceUtils.pack(FILLER, FILLER, FILLER, SPECIAL); // West facing 100
        return board;
    }

    private static PieceInventory dummyInventory() {
        PieceInventory inv = mock(PieceInventory.class);
        inv.allOrientations = new int[1024];
        inv.physicalMapping = new int[1024];
        for (int i = 0; i < 1024; i++) inv.physicalMapping[i] = i / 4;
        return inv;
    }

    // ------------------------------------------------------------------
    // extractBoardEdges
    // ------------------------------------------------------------------

    @Test
    void testExtractBoardEdgesFromFullBucasUrl() {
        String url = "https://e2.bucas.name/#puzzle=KnudHansen&board_w=16&board_h=16&board_edges=abcd1234&motifs_order=x";
        assertEquals("abcd1234", HoleSolver.extractBoardEdges(url));
    }

    @Test
    void testExtractBoardEdgesFromUrlWithNoTrailingParams() {
        String url = "https://e2.bucas.name/#puzzle=KnudHansen&board_edges=abcd1234";
        assertEquals("abcd1234", HoleSolver.extractBoardEdges(url));
    }

    @Test
    void testExtractBoardEdgesRawStringPassesThroughTrimmed() {
        assertEquals("rawboardedges", HoleSolver.extractBoardEdges("  rawboardedges  "),
                "Input with no board_edges= marker must be returned trimmed as-is");
    }

    // ------------------------------------------------------------------
    // decodeBoard
    // ------------------------------------------------------------------

    @Test
    void testDecodeBoardAllEmptySentinel() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 256; i++) sb.append("aaaa");

        int[] board = HoleSolver.decodeBoard(sb.toString());

        int[] expected = new int[256];
        Arrays.fill(expected, -1);
        assertArrayEquals(expected, board, "The 'aaaa' sentinel must decode to -1 (empty), not an all-grey piece");
    }

    @Test
    void testDecodeBoardConvertsBucasLettersToThesilColors() {
        StringBuilder sb = new StringBuilder("baaa"); // cell 0: N='b', E=W=S='a'
        for (int i = 1; i < 256; i++) sb.append("aaaa");

        int[] board = HoleSolver.decodeBoard(sb.toString());

        // 'b' (bucas index 1) maps back to TheSil color 1 (identity at index 1
        // in the THESIL_TO_BUCAS table); 'a' (index 0) maps to color 0.
        assertEquals(PieceUtils.pack(1, 0, 0, 0), board[0]);
        for (int i = 1; i < 256; i++) {
            assertEquals(-1, board[i], "Every other cell must remain the empty sentinel");
        }
    }

    @Test
    void testDecodeBoardRoundTripsThroughBucasExporter() {
        int[] original = buildBoardWithSignatureRegion();

        String url = BucasExporter.exportBoard(original);
        String edges = HoleSolver.extractBoardEdges(url);
        int[] decoded = HoleSolver.decodeBoard(edges);

        assertArrayEquals(original, decoded,
                "decodeBoard must be the exact inverse of BucasExporter.exportBoard's colour remap");
    }

    // ------------------------------------------------------------------
    // decodeBoardBlackwood / decodeBoardAuto
    //
    // Blackwood's own solver writes bucas links using his raw internal
    // colour IDs directly, not bucas-standard numbering -- decoding one with
    // the bucas table silently identifies every placed piece as "unknown"
    // (see the 2026-08-02 investigation: all 251 placed cells of a real
    // board failed findPhysicalId this way), letting hole-filling "solve"
    // using pieces already used elsewhere on the board. decodeBoardAuto
    // exists to catch that before it happens.
    // ------------------------------------------------------------------

    @Test
    void testDecodeBoardBlackwoodUsesDifferentColorMapThanBucasStandard() {
        // Letter index 2 ('c'): BUCAS_TO_THESIL maps it to TheSil colour 4,
        // BLACKWOOD_TO_THESIL maps it to TheSil colour 6 -- a genuinely
        // distinguishing case, not a coincidental match between the tables.
        StringBuilder sb = new StringBuilder("caaa");
        for (int i = 1; i < 256; i++) sb.append("aaaa");

        int[] bucasDecoded = HoleSolver.decodeBoard(sb.toString());
        int[] blackwoodDecoded = HoleSolver.decodeBoardBlackwood(sb.toString());

        assertEquals(PieceUtils.pack(4, 0, 0, 0), bucasDecoded[0]);
        assertEquals(PieceUtils.pack(6, 0, 0, 0), blackwoodDecoded[0]);
        assertNotEquals(bucasDecoded[0], blackwoodDecoded[0],
                "The two colour maps must genuinely disagree for this test to prove anything");
    }

    /** An inventory containing exactly one real piece/orientation, at physical id 0. */
    private static PieceInventory singlePieceInventory(int packedPiece) {
        PieceInventory inv = mock(PieceInventory.class);
        inv.allOrientations = new int[1024];
        inv.physicalMapping = new int[1024];
        inv.allOrientations[0] = packedPiece;
        inv.physicalMapping[0] = 0;
        for (int i = 1; i < 1024; i++) inv.physicalMapping[i] = i / 4;
        return inv;
    }

    private static String singleCellBoardEdges(char n, char e, char s, char w) {
        StringBuilder sb = new StringBuilder();
        sb.append(n).append(e).append(s).append(w);
        for (int i = 1; i < 256; i++) sb.append("aaaa");
        return sb.toString();
    }

    @Test
    void testDecodeBoardAutoUsesBlackwoodMapWhenPuzzleNameSaysSo() {
        // 'c' at cell 0 only identifies as a real piece under the Blackwood
        // interpretation (colour 6) -- the bucas-standard interpretation
        // (colour 4) isn't in this inventory at all.
        String edges = singleCellBoardEdges('c', 'a', 'a', 'a');
        PieceInventory inventory = singlePieceInventory(PieceUtils.pack(6, 0, 0, 0));
        String link = "https://e2.bucas.name/#puzzle=Joshua_Blackwood&board_w=16&board_h=16&board_edges=" + edges;

        int[] decoded = HoleSolver.decodeBoardAuto(link, inventory, false);

        assertEquals(PieceUtils.pack(6, 0, 0, 0), decoded[0],
                "puzzle=Joshua_Blackwood must select the Blackwood colour map on the first guess");
    }

    @Test
    void testDecodeBoardAutoUsesBlackwoodMapForRenamedPuzzleName() {
        // BwUtil.buildBoardString was renamed puzzle=Joshua_Blackwood -> puzzle=Knud_Hansen
        // (2026-08-04, cosmetic) -- the detection hint must recognize the new name too, not just
        // old already-saved links that still say Joshua_Blackwood.
        String edges = singleCellBoardEdges('c', 'a', 'a', 'a');
        PieceInventory inventory = singlePieceInventory(PieceUtils.pack(6, 0, 0, 0));
        String link = "https://e2.bucas.name/#puzzle=Knud_Hansen&board_w=16&board_h=16&board_edges=" + edges;

        int[] decoded = HoleSolver.decodeBoardAuto(link, inventory, false);

        assertEquals(PieceUtils.pack(6, 0, 0, 0), decoded[0],
                "puzzle=Knud_Hansen must select the Blackwood colour map on the first guess");
    }

    @Test
    void testDecodeBoardAutoDefaultsToBucasStandardWithoutPuzzleMarker() {
        // No puzzle= at all -- must default to (and stay with) bucas-standard
        // when that guess already resolves fine, matching every link this
        // project's own BucasExporter has ever produced.
        String edges = singleCellBoardEdges('c', 'a', 'a', 'a');
        PieceInventory inventory = singlePieceInventory(PieceUtils.pack(4, 0, 0, 0));

        int[] decoded = HoleSolver.decodeBoardAuto(edges, inventory, false);

        assertEquals(PieceUtils.pack(4, 0, 0, 0), decoded[0]);
    }

    @Test
    void testDecodeBoardAutoSwitchesToBlackwoodWhenBucasGuessResolvesPoorly() {
        // No puzzle= marker (so the initial guess is bucas-standard), but
        // the inventory only contains the Blackwood-interpreted piece --
        // this is the actual failure mode from the 2026-08-02 investigation,
        // and must self-correct rather than silently returning an
        // unidentifiable board.
        String edges = singleCellBoardEdges('c', 'a', 'a', 'a');
        PieceInventory inventory = singlePieceInventory(PieceUtils.pack(6, 0, 0, 0));

        int[] decoded = HoleSolver.decodeBoardAuto(edges, inventory, false);

        assertEquals(PieceUtils.pack(6, 0, 0, 0), decoded[0],
                "Must detect the bucas-standard guess doesn't resolve and fall back to the Blackwood map");
    }

    // ------------------------------------------------------------------
    // ConflictSolveResult.bestBoard()
    // ------------------------------------------------------------------

    @Test
    void testBestBoardPrefersRepairedWhenPresent() {
        int[] finalBoard = new int[256];
        int[] repaired = new int[256];
        repaired[0] = 42;

        HoleSolver.ConflictSolveResult result = new HoleSolver.ConflictSolveResult(finalBoard, repaired, false);

        assertSame(repaired, result.bestBoard(), "bestBoard must prefer the heuristic repair when present");
    }

    @Test
    void testBestBoardFallsBackToFinalWhenRepairedNull() {
        int[] finalBoard = new int[256];

        HoleSolver.ConflictSolveResult result = new HoleSolver.ConflictSolveResult(finalBoard, null, false);

        assertSame(finalBoard, result.bestBoard(), "bestBoard must fall back to finalBoard when no repair exists");
    }

    // ------------------------------------------------------------------
    // findConflicts (private)
    // ------------------------------------------------------------------

    @SuppressWarnings("unchecked")
    private List<int[]> invokeFindConflicts(int[] board) throws Exception {
        Method m = HoleSolver.class.getDeclaredMethod("findConflicts", int[].class);
        m.setAccessible(true);
        return (List<int[]>) m.invoke(null, (Object) board);
    }

    @Test
    void testFindConflictsIgnoresFullyEmptyBoard() throws Exception {
        int[] board = new int[256];
        Arrays.fill(board, -1);

        assertTrue(invokeFindConflicts(board).isEmpty(), "An all-empty board has nothing to check FROM");
    }

    @Test
    void testFindConflictsCleanCompleteBoardHasNoConflicts() throws Exception {
        assertTrue(invokeFindConflicts(buildUniformBoard()).isEmpty(),
                "A fully self-consistent board must report zero conflicts");
    }

    @Test
    void testFindConflictsDetectsInternalEastWestMismatch() throws Exception {
        int[] board = new int[256];
        Arrays.fill(board, -1);
        board[100] = PieceUtils.pack(1, 2, 1, 1); // East = 2
        board[101] = PieceUtils.pack(1, 1, 1, 9); // West = 9, mismatches East above

        List<int[]> conflicts = invokeFindConflicts(board);

        assertEquals(1, conflicts.size());
        assertArrayEquals(new int[]{100, 101}, conflicts.get(0));
    }

    @Test
    void testFindConflictsDetectsBorderViolation() throws Exception {
        int[] board = new int[256];
        Arrays.fill(board, -1);
        board[0] = PieceUtils.pack(5, 0, 0, 0); // North should be BORDER_COLOR at row 0, but is 5

        List<int[]> conflicts = invokeFindConflicts(board);

        assertEquals(1, conflicts.size());
        assertArrayEquals(new int[]{0, -1}, conflicts.get(0));
    }

    // ------------------------------------------------------------------
    // connectedComponents (private)
    // ------------------------------------------------------------------

    @SuppressWarnings("unchecked")
    private List<List<Integer>> invokeConnectedComponents(boolean[] inHole) throws Exception {
        Method m = HoleSolver.class.getDeclaredMethod("connectedComponents", boolean[].class);
        m.setAccessible(true);
        return (List<List<Integer>>) m.invoke(null, (Object) inHole);
    }

    @Test
    void testConnectedComponentsEmptyReturnsNoComponents() throws Exception {
        assertTrue(invokeConnectedComponents(new boolean[256]).isEmpty());
    }

    @Test
    void testConnectedComponentsGroupsOrthogonallyAdjacentCells() throws Exception {
        boolean[] inHole = new boolean[256];
        inHole[100] = true;
        inHole[101] = true; // horizontally adjacent to 100
        inHole[117] = true; // vertically adjacent to 101 (101 + 16)
        inHole[5] = true;   // isolated

        List<List<Integer>> components = invokeConnectedComponents(inHole);

        assertEquals(2, components.size());
        List<Integer> big = components.get(0).size() == 3 ? components.get(0) : components.get(1);
        List<Integer> small = components.get(0).size() == 3 ? components.get(1) : components.get(0);
        assertEquals(List.of(100, 101, 117), big, "Component cells must be sorted ascending");
        assertEquals(List.of(5), small);
    }

    // ------------------------------------------------------------------
    // findPhysicalId / physicalPieceAndRotation (private)
    // ------------------------------------------------------------------

    private int invokeFindPhysicalId(PieceInventory inv, int piece) throws Exception {
        Method m = HoleSolver.class.getDeclaredMethod("findPhysicalId", PieceInventory.class, int.class);
        m.setAccessible(true);
        return (int) m.invoke(null, inv, piece);
    }

    private int[] invokePhysicalPieceAndRotation(PieceInventory inv, int piece) throws Exception {
        Method m = HoleSolver.class.getDeclaredMethod("physicalPieceAndRotation", PieceInventory.class, int.class);
        m.setAccessible(true);
        return (int[]) m.invoke(null, inv, piece);
    }

    private static PieceInventory smallDistinctInventory() {
        int[] baseFixture = new int[256];
        Arrays.fill(baseFixture, PieceUtils.pack(1, 1, 1, 1));
        baseFixture[7] = PieceUtils.pack(2, 3, 4, 5); // the only asymmetric, uniquely-identifiable piece
        return new PieceInventory(baseFixture);
    }

    @Test
    void testFindPhysicalIdMatchesRotatedOrientation() throws Exception {
        PieceInventory inv = smallDistinctInventory();
        int rotatedOnce = PieceUtils.rotate(PieceUtils.pack(2, 3, 4, 5));

        assertEquals(7, invokeFindPhysicalId(inv, rotatedOnce));
    }

    @Test
    void testFindPhysicalIdReturnsMinusOneForUnknownPiece() throws Exception {
        PieceInventory inv = smallDistinctInventory();

        assertEquals(-1, invokeFindPhysicalId(inv, PieceUtils.pack(20, 21, 22, 8)));
    }

    @Test
    void testPhysicalPieceAndRotationReturnsOneBasedIdAndDegrees() throws Exception {
        PieceInventory inv = smallDistinctInventory();
        int base = PieceUtils.pack(2, 3, 4, 5);
        int rotatedTwice = PieceUtils.rotate(PieceUtils.rotate(base));

        int[] result = invokePhysicalPieceAndRotation(inv, rotatedTwice);

        assertEquals(8, result[0], "Physical piece number is 1-based (physId 7 -> 8)");
        assertEquals(180, result[1], "Two 90-degree rotations from reference orientation");
    }

    @Test
    void testPhysicalPieceAndRotationReturnsMinusOnesWhenNotFound() throws Exception {
        PieceInventory inv = smallDistinctInventory();

        assertArrayEquals(new int[]{-1, -1},
                invokePhysicalPieceAndRotation(inv, PieceUtils.pack(20, 21, 22, 8)));
    }

    // ------------------------------------------------------------------
    // countPlaced (private)
    // ------------------------------------------------------------------

    private int invokeCountPlaced(int[] board) throws Exception {
        Method m = HoleSolver.class.getDeclaredMethod("countPlaced", int[].class);
        m.setAccessible(true);
        return (int) m.invoke(null, (Object) board);
    }

    @Test
    void testCountPlacedIgnoresEmptyAndSurgeonHoles() throws Exception {
        int[] board = new int[256];
        Arrays.fill(board, -1);
        board[0] = 500;
        board[1] = -2; // surgeon hole
        board[2] = 600;

        assertEquals(2, invokeCountPlaced(board));
    }

    // ------------------------------------------------------------------
    // writePhysicalLayout (private, but pure given a PrintWriter — no file I/O)
    // ------------------------------------------------------------------

    @Test
    void testWritePhysicalLayoutFormatsEmptyAndPlacedCells() throws Exception {
        int[] baseFixture = new int[256];
        Arrays.fill(baseFixture, PieceUtils.pack(1, 1, 1, 1));
        PieceInventory inv = new PieceInventory(baseFixture);

        int[] board = new int[256];
        Arrays.fill(board, -1);
        board[0] = baseFixture[0]; // physical piece 0 (1-based "1"), rotation 0

        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);
        Method m = HoleSolver.class.getDeclaredMethod("writePhysicalLayout",
                PrintWriter.class, String.class, int[].class, PieceInventory.class);
        m.setAccessible(true);
        m.invoke(null, pw, "Test board", board, inv);
        pw.flush();

        String output = sw.toString();
        assertTrue(output.contains("Test board"), "Header must include the caller-supplied label");
        assertTrue(output.contains("EMPTY"), "Cells holding -1 must render as EMPTY");
        assertTrue(output.contains("1@0"), "Placed cell must render as <1-based physical id>@<rotation degrees>");
    }

    // ------------------------------------------------------------------
    // solveConflicts (public API)
    // ------------------------------------------------------------------

    @Test
    void testSolveConflictsAlreadyCompleteReturnsUnchanged() {
        int[] board = buildUniformBoard();

        HoleSolver.ConflictSolveResult result = HoleSolver.solveConflicts(board, dummyInventory(), false);

        assertArrayEquals(board, result.finalBoard(), "An already conflict-free board must pass through unchanged");
        assertNull(result.repairedBoard(), "No repair should be attempted when there is nothing to fix");
        assertArrayEquals(board, result.bestBoard());
    }

    @Test
    void testSolveConflictsFourArgOverloadAlsoShortCircuitsOnCleanBoard() {
        int[] board = buildUniformBoard();

        HoleSolver.ConflictSolveResult result = HoleSolver.solveConflicts(board, dummyInventory(), false, 1);

        assertArrayEquals(board, result.finalBoard());
        assertNull(result.repairedBoard());
    }

    @Test
    void testSolveConflictsSolvesSmallRegionExactly() {
        int[] correctBoard = buildBoardWithSignatureRegion();
        PieceInventory inventory = new PieceInventory(Arrays.copyOf(correctBoard, 256));

        // Corrupt cell 100 by rotating it in place: this breaks its match with
        // cell 101 (east/west) AND with cell 116 (south/north), forming a
        // small 3-cell connected conflict region {100, 101, 116} — solvable
        // exactly by simply rotating piece 100 back.
        int[] brokenBoard = Arrays.copyOf(correctBoard, 256);
        brokenBoard[100] = PieceUtils.rotate(correctBoard[100]);

        HoleSolver.ConflictSolveResult result = HoleSolver.solveConflicts(brokenBoard, inventory, false, 50);

        assertNull(result.repairedBoard(),
                "The exact search must clear this tiny, genuinely solvable region without needing fallback repair");
        int[] finalBoard = result.finalBoard();
        assertEquals(0, ConflictAnnealer.countInternalConflicts(finalBoard), "No internal edge conflicts should remain");
        assertEquals(0, ConflictAnnealer.countBorderViolations(finalBoard), "No frame violations should remain");
        for (int p : finalBoard) {
            assertTrue(p != -1 && p != -2, "Every cell must be filled after an exact region solve");
        }
    }

    @Test
    void testResolveLinkInputPassesThroughPlainArgUnchanged() throws Exception {
        assertEquals("https://e2.bucas.name/#board_edges=abcd", HoleSolver.resolveLinkInput("https://e2.bucas.name/#board_edges=abcd"));
    }

    @Test
    void testResolveLinkInputReadsAtFile(@org.junit.jupiter.api.io.TempDir java.nio.file.Path tempDir) throws Exception {
        java.nio.file.Path linkFile = tempDir.resolve("link.txt");
        // Trailing newline (as a real text editor/echo would leave) must be trimmed, and the '&'
        // characters -- the whole reason this @file form exists -- must survive untouched.
        java.nio.file.Files.writeString(linkFile, "https://e2.bucas.name/#puzzle=Joshua_Blackwood&board_edges=abcd&motifs_order=jblackwood\n");

        String resolved = HoleSolver.resolveLinkInput("@" + linkFile);

        assertEquals("https://e2.bucas.name/#puzzle=Joshua_Blackwood&board_edges=abcd&motifs_order=jblackwood", resolved);
    }

    @Test
    void testResolveLinkInputThrowsOnMissingFile() {
        assertThrows(java.io.IOException.class, () -> HoleSolver.resolveLinkInput("@does/not/exist.txt"));
    }
}
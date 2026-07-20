package dk.puzzle.io;

import dk.puzzle.util.PieceUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link CheckpointGenerator}'s underlying logic methods.
 *
 * <p>{@code main(String[])} is a standalone command-line tool that reads and
 * writes hardcoded real project paths (JBlackwood_Pieces.txt, Raw_468_Board.txt,
 * and a {@code profiles} folder) - it is never invoked here. Instead, the
 * private color-cracking/parsing/translation helpers are reached via
 * reflection, and any file I/O is redirected to a JUnit {@code @TempDir}.</p>
 *
 * <p>{@code jbToUserColor} and {@code userColorUsed} are package-private
 * static fields shared by several of these helpers, so they are reset before
 * every test to avoid cross-test pollution.</p>
 */
class CheckpointGeneratorTest {

    @BeforeEach
    void resetSharedColorMapState() {
        Arrays.fill(CheckpointGenerator.jbToUserColor, -1);
        Arrays.fill(CheckpointGenerator.userColorUsed, false);
    }

    private boolean invokeCrackColorCode(int currentColorId, int[][] jbPieces, int[][] userPieces) throws Exception {
        Method m = CheckpointGenerator.class.getDeclaredMethod("crackColorCode", int.class, int[][].class, int[][].class);
        m.setAccessible(true);
        return (boolean) m.invoke(null, currentColorId, jbPieces, userPieces);
    }

    private boolean invokeIsValidPartialMap(int[][] jbPieces, int[][] userPieces) throws Exception {
        Method m = CheckpointGenerator.class.getDeclaredMethod("isValidPartialMap", int[][].class, int[][].class);
        m.setAccessible(true);
        return (boolean) m.invoke(null, jbPieces, userPieces);
    }

    private boolean invokePieceExistsInUserSet(int n, int e, int s, int w, int[][] userPieces) throws Exception {
        Method m = CheckpointGenerator.class.getDeclaredMethod("pieceExistsInUserSet",
                int.class, int.class, int.class, int.class, int[][].class);
        m.setAccessible(true);
        return (boolean) m.invoke(null, n, e, s, w, userPieces);
    }

    private int[][] invokeExtractUserBasePieces(int[] allOrientations) throws Exception {
        Method m = CheckpointGenerator.class.getDeclaredMethod("extractUserBasePieces", int[].class);
        m.setAccessible(true);
        return (int[][]) m.invoke(null, (Object) allOrientations);
    }

    private int invokeFindPieceByColors(int[] allOrientations, int n, int e, int s, int w) throws Exception {
        Method m = CheckpointGenerator.class.getDeclaredMethod("findPieceByColors",
                int[].class, int.class, int.class, int.class, int.class);
        m.setAccessible(true);
        return (int) m.invoke(null, allOrientations, n, e, s, w);
    }

    private int[][] invokeLoadJBlackwoodPieces(String filepath) throws Exception {
        Method m = CheckpointGenerator.class.getDeclaredMethod("loadJBlackwoodPieces", String.class);
        m.setAccessible(true);
        return (int[][]) m.invoke(null, filepath);
    }

    private int[][] invokeCreateTranslatedBoard(String rawBoardFile, int[][] jbPieces, int[] allOrientations) throws Exception {
        Method m = CheckpointGenerator.class.getDeclaredMethod("createTranslatedBoard",
                String.class, int[][].class, int[].class);
        m.setAccessible(true);
        return (int[][]) m.invoke(null, rawBoardFile, jbPieces, allOrientations);
    }

    @Test
    void testExtractUserBasePiecesUnpacksBaseOrientationOnly() throws Exception {
        int[] allOrientations = new int[1024];
        allOrientations[0] = PieceUtils.pack(1, 2, 3, 4);       // physical piece 0, orientation 0 (base)
        allOrientations[1] = PieceUtils.pack(99, 99, 99, 99);   // orientation 1 of piece 0, must be ignored
        allOrientations[4] = PieceUtils.pack(5, 6, 7, 8);       // physical piece 1, orientation 0 (base)

        int[][] bases = invokeExtractUserBasePieces(allOrientations);

        assertArrayEquals(new int[]{1, 2, 3, 4}, bases[0], "extractUserBasePieces must read only the rotation-0 slot (index i*4)");
        assertArrayEquals(new int[]{5, 6, 7, 8}, bases[1]);
    }

    @Test
    void testFindPieceByColorsReturnsPackedPieceOnMatch() throws Exception {
        int p = PieceUtils.pack(1, 2, 3, 4);
        int[] allOrientations = new int[]{PieceUtils.pack(9, 9, 9, 9), p, PieceUtils.pack(0, 0, 0, 0)};

        assertEquals(p, invokeFindPieceByColors(allOrientations, 1, 2, 3, 4));
        assertEquals(-1, invokeFindPieceByColors(allOrientations, 8, 8, 8, 8), "No matching orientation must yield -1");
    }

    @Test
    void testPieceExistsInUserSetChecksAllFourRotations() throws Exception {
        int[][] userPieces = {{1, 2, 3, 4}};

        assertTrue(invokePieceExistsInUserSet(1, 2, 3, 4, userPieces), "Direct (0 deg) orientation must match");
        assertTrue(invokePieceExistsInUserSet(4, 1, 2, 3, userPieces), "90 deg rotation must match");
        assertTrue(invokePieceExistsInUserSet(3, 4, 1, 2, userPieces), "180 deg rotation must match");
        assertTrue(invokePieceExistsInUserSet(2, 3, 4, 1, userPieces), "270 deg rotation must match");
        assertFalse(invokePieceExistsInUserSet(9, 9, 9, 9, userPieces), "Unrelated color combination must not match");
    }

    @Test
    void testIsValidPartialMapSkipsPiecesNotYetFullyMapped() throws Exception {
        int[][] jbPieces = {{1, 2, 3, 4}};
        int[][] userPieces = new int[0][4]; // deliberately empty: would fail pieceExistsInUserSet if actually checked
        // jbToUserColor entries for 1..4 are left at -1 (unmapped) by @BeforeEach

        assertTrue(invokeIsValidPartialMap(jbPieces, userPieces),
                "A piece whose colors are not all mapped yet must be skipped, not treated as invalid");
    }

    @Test
    void testIsValidPartialMapValidatesFullyMappedPieces() throws Exception {
        int[][] jbPieces = {{1, 2, 3, 4}};
        int[][] userPieces = {{1, 2, 3, 4}};
        for (int i = 1; i <= 4; i++) {
            CheckpointGenerator.jbToUserColor[i] = i; // identity mapping
        }

        assertTrue(invokeIsValidPartialMap(jbPieces, userPieces), "Fully mapped piece present in userPieces must validate");

        CheckpointGenerator.jbToUserColor[4] = 9; // break the mapping so it no longer matches any userPiece rotation
        assertFalse(invokeIsValidPartialMap(jbPieces, userPieces), "Fully mapped piece absent from userPieces must invalidate");
    }

    @Test
    void testCrackColorCodeSucceedsWithNoConstraints() throws Exception {
        int[][] emptyJbPieces = new int[0][4];
        int[][] anyUserPieces = new int[0][4];

        assertTrue(invokeCrackColorCode(1, emptyJbPieces, anyUserPieces),
                "With no jb pieces to validate against, every candidate mapping is valid");
        for (int i = 1; i <= 22; i++) {
            assertEquals(i, CheckpointGenerator.jbToUserColor[i], "Unconstrained search assigns colors in ascending order");
        }
    }

    @Test
    void testCrackColorCodeFailsWhenNoUserPieceCanEverMatch() throws Exception {
        int[][] jbPieces = {{1, 2, 3, 4}};
        int[][] emptyUserPieces = new int[0][4]; // no piece can ever satisfy pieceExistsInUserSet

        assertFalse(invokeCrackColorCode(1, jbPieces, emptyUserPieces),
                "No permutation can succeed when the user piece set is empty");
    }

    @Test
    void testLoadJBlackwoodPiecesParsesMatchingLinesOnly(@TempDir Path tempDir) throws Exception {
        Path piecesFile = tempDir.resolve("pieces.txt");
        Files.writeString(piecesFile,
                "PieceNumber = 1, TopSide = 1, RightSide = 2, BottomSide = 3, LeftSide = 4\n" +
                        "this line matches nothing\n" +
                        "PieceNumber = 2, TopSide = 5, RightSide = 6, BottomSide = 7, LeftSide = 8\n");

        int[][] pieces = invokeLoadJBlackwoodPieces(piecesFile.toString());

        assertArrayEquals(new int[]{1, 2, 3, 4}, pieces[0]);
        assertArrayEquals(new int[]{5, 6, 7, 8}, pieces[1]);
        assertArrayEquals(new int[]{0, 0, 0, 0}, pieces[2], "Unmatched piece slots must remain at their default zero values");
    }

    @Test
    void testCreateTranslatedBoardMapsColorsAndFindsMatchingPiece(@TempDir Path tempDir) throws Exception {
        Path boardFile = tempDir.resolve("board.txt");
        Files.writeString(boardFile, "1/0 ---/-\n");

        int[][] jbPieces = new int[256][4];
        jbPieces[0] = new int[]{1, 2, 3, 4}; // physicalId 1

        for (int i = 1; i <= 4; i++) {
            CheckpointGenerator.jbToUserColor[i] = i; // identity mapping
        }

        int userPiece = PieceUtils.pack(1, 2, 3, 4);
        int[] allOrientations = new int[]{userPiece};

        int[][] board = invokeCreateTranslatedBoard(boardFile.toString(), jbPieces, allOrientations);

        assertEquals(userPiece, board[0][0], "Row 0/Col 0 must resolve to the matching packed user piece");
        assertEquals(-1, board[0][1], "The '---/-' slot must be left untouched (-1)");
        assertEquals(-1, board[5][5], "Untouched cells elsewhere on the board must stay at their -1 default");
    }

    @Test
    void testSaveRecordCheckpointWritesReadableSerializedBoard(@TempDir Path tempDir) throws Exception {
        int[][] board = new int[16][16];
        for (int[] row : board) {
            Arrays.fill(row, -1);
        }
        board[3][3] = 555;
        String profileFolder = tempDir.resolve("profile").toString();

        CheckpointGenerator.saveRecordCheckpoint(board, 254, profileFolder);

        File expectedFile = new File(profileFolder, "checkpoint_254.dat");
        assertTrue(expectedFile.exists(), "saveRecordCheckpoint must create checkpoint_<score>.dat in the profile folder");

        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(expectedFile))) {
            int[][] loaded = (int[][]) ois.readObject();
            assertEquals(555, loaded[3][3]);
        }
    }
}

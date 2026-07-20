package dk.puzzle.core;

import dk.puzzle.util.PieceUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Eternity core logic.
 * main() itself is Swing/thread/System.exit wiring and is intentionally not
 * exercised here; these tests cover the pure helpers and the shared display
 * state that solver threads publish to.
 */
class EternityTest {

    @BeforeEach
    void resetStaticState() throws Exception {
        Eternity.currentScore.set(0);
        getHighScoreField().set(0);
        int[][] board = getCurrentDisplayBoardField();
        Arrays.fill(board, null);
    }

    private int[][] getCurrentDisplayBoardField() throws Exception {
        Field field = Eternity.class.getDeclaredField("currentDisplayBoard");
        field.setAccessible(true);
        return (int[][]) field.get(null);
    }

    private AtomicInteger getHighScoreField() throws Exception {
        Field field = Eternity.class.getDeclaredField("highScore");
        field.setAccessible(true);
        return (AtomicInteger) field.get(null);
    }

    private int invokeParseColorField(String value) throws Exception {
        Method method = Eternity.class.getDeclaredMethod("parseColorField", String.class);
        method.setAccessible(true);
        return (int) method.invoke(null, value);
    }

    private int[] invokeGenerateMock() throws Exception {
        Method method = Eternity.class.getDeclaredMethod("generateMock");
        method.setAccessible(true);
        return (int[]) method.invoke(null);
    }

    @Test
    void testParseColorFieldNullAndEmptyReturnZero() throws Exception {
        assertEquals(0, invokeParseColorField(null), "Null field must default to border color 0");
        assertEquals(0, invokeParseColorField(""), "Empty field must default to border color 0");
        assertEquals(0, invokeParseColorField("   "), "Whitespace-only field must default to border color 0");
    }

    @Test
    void testParseColorFieldParsesValidIntegers() throws Exception {
        assertEquals(5, invokeParseColorField("5"));
        assertEquals(7, invokeParseColorField(" 7 "), "Surrounding whitespace must be trimmed before parsing");
        assertEquals(0, invokeParseColorField("0"));
    }

    @Test
    void testParseColorFieldNonNumericReturnsZero() throws Exception {
        assertEquals(0, invokeParseColorField("abc"), "Non-numeric garbage must fall back to 0 rather than throw");
        assertEquals(0, invokeParseColorField("12abc"));
    }

    @Test
    void testGenerateMockIsDeterministic() throws Exception {
        int[] first = invokeGenerateMock();
        int[] second = invokeGenerateMock();

        assertEquals(256, first.length, "Mock board must always produce exactly 256 pieces");
        assertArrayEquals(first, second, "generateMock uses a fixed seed and must be reproducible run-to-run");
    }

    @Test
    void testGenerateMockIsInternallyConsistent() throws Exception {
        int[] pieces = invokeGenerateMock();

        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                int p = pieces[r * 16 + c];

                if (c < 15) {
                    int east = PieceUtils.getEast(p);
                    int neighborWest = PieceUtils.getWest(pieces[r * 16 + c + 1]);
                    assertEquals(east, neighborWest,
                            String.format("East edge of (%d,%d) must match West edge of (%d,%d)", r, c, r, c + 1));
                }
                if (r < 15) {
                    int south = PieceUtils.getSouth(p);
                    int neighborNorth = PieceUtils.getNorth(pieces[(r + 1) * 16 + c]);
                    assertEquals(south, neighborNorth,
                            String.format("South edge of (%d,%d) must match North edge of (%d,%d)", r, c, r + 1, c));
                }
            }
        }

        // Outer border must be the grey/border color (0) on all four sides.
        for (int c = 0; c < 16; c++) {
            assertEquals(0, PieceUtils.getNorth(pieces[c]), "Top row North edges must be border color");
            assertEquals(0, PieceUtils.getSouth(pieces[15 * 16 + c]), "Bottom row South edges must be border color");
        }
        for (int r = 0; r < 16; r++) {
            assertEquals(0, PieceUtils.getWest(pieces[r * 16]), "Left column West edges must be border color");
            assertEquals(0, PieceUtils.getEast(pieces[r * 16 + 15]), "Right column East edges must be border color");
        }
    }

    @Test
    void testUpdateDisplaySetsScoreAndRecord() throws Exception {
        int[][] board = new int[16][];
        for (int i = 0; i < 16; i++) {
            board[i] = new int[16];
        }

        Eternity.updateDisplay(120, 215, board);

        assertEquals(120, Eternity.currentScore.get());
        assertEquals(215, getHighScoreField().get());
    }

    @Test
    void testUpdateDisplayDefensivelyClonesRows() throws Exception {
        int[][] board = new int[16][];
        board[0] = new int[]{1, 2, 3};

        Eternity.updateDisplay(1, 1, board);

        // Mutate the caller's array after publishing; the stored state must be unaffected.
        board[0][0] = 999;

        int[][] stored = getCurrentDisplayBoardField();
        assertEquals(1, stored[0][0], "updateDisplay must clone each row, not alias the caller's array");
    }

    @Test
    void testUpdateDisplayHandlesNullRows() throws Exception {
        int[][] board = new int[16][];
        board[3] = new int[]{7, 8};
        // All other rows stay null.

        Eternity.updateDisplay(1, 1, board);

        int[][] stored = getCurrentDisplayBoardField();
        assertNotNull(stored[3]);
        assertArrayEquals(new int[]{7, 8}, stored[3]);
        assertNull(stored[0], "Null rows in the source board must be stored as null, not skipped/stale");
    }

}
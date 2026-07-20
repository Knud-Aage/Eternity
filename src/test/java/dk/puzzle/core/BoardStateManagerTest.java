package dk.puzzle.core;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for BoardStateManager.
 * This class is currently unused/dead code (part of an in-progress extraction
 * from the EternitySolver monolith) but its logic is deterministic and worth
 * locking down independently of EternitySolver's own behavior.
 */
class BoardStateManagerTest {

    private int[] sampleBoard(int firstValue) {
        int[] board = new int[256];
        Arrays.fill(board, -1);
        board[0] = firstValue;
        return board;
    }

    @Test
    void testConstructorInitializesBothBoardsToNegativeOne() {
        BoardStateManager manager = new BoardStateManager();

        int[] resume = manager.getFlatResumeBoardCopy();
        int[] best = manager.getGlobalBestBoardCopy();

        assertEquals(256, resume.length);
        assertEquals(256, best.length);
        for (int i = 0; i < 256; i++) {
            assertEquals(-1, resume[i], "flatResumeBoard must start fully empty (-1)");
            assertEquals(-1, best[i], "globalBestBoard must start fully empty (-1)");
        }
    }

    @Test
    void testInitialAbsoluteHighScoreIsZero() {
        BoardStateManager manager = new BoardStateManager();
        assertEquals(0, manager.getAbsoluteHighScore());
    }

    @Test
    void testGetFlatResumeBoardCopyReturnsDefensiveCopy() {
        BoardStateManager manager = new BoardStateManager();

        int[] copy = manager.getFlatResumeBoardCopy();
        copy[0] = 12345;

        int[] second = manager.getFlatResumeBoardCopy();
        assertEquals(-1, second[0], "Mutating a returned copy must not affect internal state");
    }

    @Test
    void testGetGlobalBestBoardCopyReturnsDefensiveCopy() {
        BoardStateManager manager = new BoardStateManager();
        manager.updateGlobalBestIfHigher(sampleBoard(7), 10);

        int[] copy = manager.getGlobalBestBoardCopy();
        copy[0] = 99999;

        int[] second = manager.getGlobalBestBoardCopy();
        assertEquals(7, second[0], "Mutating a returned copy must not affect internal state");
    }

    @Test
    void testUpdateGlobalBestIfHigherUpdatesWhenScoreIsHigher() {
        BoardStateManager manager = new BoardStateManager();
        int[] board = sampleBoard(42);

        manager.updateGlobalBestIfHigher(board, 50);

        assertEquals(50, manager.getAbsoluteHighScore());
        assertArrayEquals(board, manager.getGlobalBestBoardCopy());
    }

    @Test
    void testUpdateGlobalBestIfHigherIgnoresEqualScore() {
        BoardStateManager manager = new BoardStateManager();
        manager.updateGlobalBestIfHigher(sampleBoard(1), 50);

        manager.updateGlobalBestIfHigher(sampleBoard(2), 50);

        assertEquals(50, manager.getAbsoluteHighScore());
        assertEquals(1, manager.getGlobalBestBoardCopy()[0], "Equal score must not overwrite the stored board (strict > check)");
    }

    @Test
    void testUpdateGlobalBestIfHigherIgnoresLowerScore() {
        BoardStateManager manager = new BoardStateManager();
        manager.updateGlobalBestIfHigher(sampleBoard(1), 50);

        manager.updateGlobalBestIfHigher(sampleBoard(2), 20);

        assertEquals(50, manager.getAbsoluteHighScore());
        assertEquals(1, manager.getGlobalBestBoardCopy()[0]);
    }

    @Test
    void testUpdateGlobalBestIfHigherCopiesValuesNotReference() {
        BoardStateManager manager = new BoardStateManager();
        int[] board = sampleBoard(3);

        manager.updateGlobalBestIfHigher(board, 10);
        board[0] = 999; // mutate the caller's array after the call

        assertEquals(3, manager.getGlobalBestBoardCopy()[0], "Internal state must not alias the caller's array");
    }

    @Test
    void testSetAbsoluteHighScoreOverridesDirectly() {
        BoardStateManager manager = new BoardStateManager();
        manager.setAbsoluteHighScore(777);
        assertEquals(777, manager.getAbsoluteHighScore());
    }

    @Test
    void testMultipleUpdatesOnlyKeepStrictlyIncreasingScores() {
        BoardStateManager manager = new BoardStateManager();

        manager.updateGlobalBestIfHigher(sampleBoard(1), 10);
        manager.updateGlobalBestIfHigher(sampleBoard(2), 30);
        manager.updateGlobalBestIfHigher(sampleBoard(3), 20); // lower, must be ignored

        assertEquals(30, manager.getAbsoluteHighScore());
        assertEquals(2, manager.getGlobalBestBoardCopy()[0]);
    }
}

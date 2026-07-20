package dk.puzzle.core;

import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the standalone dk.puzzle.core.TopBoardRegistry.
 * Not to be confused with EternitySolver's private inner class of the same
 * name (covered by EternitySolverTest#testTopBoardRegistry) -- this is a
 * distinct top-level class with extra behavior (score-sorted insertion,
 * getBest(), bestScore(), size(), getRawRegistry()) that the inner class does
 * not have. It is currently unused/dead code but deterministic and worth
 * locking down independently.
 */
class TopBoardRegistryTest {

    private int[] board(int firstValue) {
        int[] b = new int[256];
        b[0] = firstValue;
        return b;
    }

    @Test
    void testGetBestOnEmptyRegistryReturnsNull() {
        TopBoardRegistry registry = new TopBoardRegistry();
        assertNull(registry.getBest());
    }

    @Test
    void testNextForRepairOnEmptyRegistryReturnsNull() {
        TopBoardRegistry registry = new TopBoardRegistry();
        assertNull(registry.nextForRepair());
    }

    @Test
    void testBestScoreOnEmptyRegistryReturnsZero() {
        TopBoardRegistry registry = new TopBoardRegistry();
        assertEquals(0, registry.bestScore());
    }

    @Test
    void testSizeOnEmptyRegistryIsZero() {
        TopBoardRegistry registry = new TopBoardRegistry();
        assertEquals(0, registry.size());
    }

    @Test
    void testOfferAndGetBestReturnsHighestScoringBoard() {
        TopBoardRegistry registry = new TopBoardRegistry();
        registry.offer(board(1), 10);
        registry.offer(board(2), 30);
        registry.offer(board(3), 20);

        assertEquals(3, registry.size());
        assertEquals(30, registry.bestScore());
        assertEquals(2, registry.getBest()[0], "getBest must return the highest-scoring board");
    }

    @Test
    void testOfferDeduplicatesIdenticalBoardAndScore() {
        TopBoardRegistry registry = new TopBoardRegistry();
        registry.offer(board(5), 10);
        registry.offer(board(5), 10); // exact duplicate: same board content and same score

        assertEquals(1, registry.size(), "Offering an identical board+score pair must not add a duplicate entry");
    }

    @Test
    void testOfferAllowsSameBoardContentWithDifferentScore() {
        TopBoardRegistry registry = new TopBoardRegistry();
        registry.offer(board(5), 10);
        registry.offer(board(5), 20); // same board contents, different score: NOT a duplicate per the offer() check

        assertEquals(2, registry.size(),
                "Dedup check requires matching score AND board contents; a different score must be treated as a new entry");
    }

    @Test
    void testOfferAllowsDifferentBoardsWithSameScore() {
        TopBoardRegistry registry = new TopBoardRegistry();
        registry.offer(board(5), 10);
        registry.offer(board(6), 10); // same score, different board contents

        assertEquals(2, registry.size());
    }

    @Test
    void testGetAllReturnsEntriesSortedByScoreDescending() {
        TopBoardRegistry registry = new TopBoardRegistry();
        registry.offer(board(1), 5);
        registry.offer(board(2), 9);
        registry.offer(board(3), 1);
        registry.offer(board(4), 3);

        List<int[]> all = registry.getAll();
        assertEquals(4, all.size());
        assertEquals(2, all.get(0)[0]);
        assertEquals(1, all.get(1)[0]);
        assertEquals(4, all.get(2)[0]);
        assertEquals(3, all.get(3)[0]);
    }

    @Test
    void testCapacityLimitEnforcedAndKeepsHighestScores() {
        TopBoardRegistry registry = new TopBoardRegistry();
        for (int i = 0; i < 30; i++) {
            registry.offer(board(i), i);
        }

        assertEquals(TopBoardRegistry.MAX_SIZE, registry.size(), "Registry must never exceed MAX_SIZE entries");
        assertEquals(29, registry.bestScore(), "Highest-scoring board must survive capacity trimming");

        List<int[]> all = registry.getAll();
        Set<Integer> keptValues = new HashSet<>();
        for (int[] b : all) keptValues.add(b[0]);

        for (int i = 10; i < 30; i++) {
            assertTrue(keptValues.contains(i), "Top-20 by score (10..29) must be retained: missing " + i);
        }
        for (int i = 0; i < 10; i++) {
            assertFalse(keptValues.contains(i), "Lowest-scoring boards must be evicted once capacity is exceeded: " + i + " should have been dropped");
        }
    }

    @Test
    void testNextForRepairRoundRobinsThroughAllEntriesAndWrapsAround() {
        TopBoardRegistry registry = new TopBoardRegistry();
        registry.offer(board(1), 30);
        registry.offer(board(2), 20);
        registry.offer(board(3), 10);

        int[] first = registry.nextForRepair();
        int[] second = registry.nextForRepair();
        int[] third = registry.nextForRepair();
        int[] fourth = registry.nextForRepair(); // wraps back to the first entry

        assertEquals(1, first[0]);
        assertEquals(2, second[0]);
        assertEquals(3, third[0]);
        assertEquals(1, fourth[0], "Round-robin index must wrap back to the first entry after exhausting the list");
    }

    @Test
    void testGetBestReturnsDefensiveCopy() {
        TopBoardRegistry registry = new TopBoardRegistry();
        registry.offer(board(7), 10);

        int[] copy = registry.getBest();
        copy[0] = 999;

        assertEquals(7, registry.getBest()[0], "Mutating a returned board must not affect internal state");
    }

    @Test
    void testGetAllReturnsDefensiveCopies() {
        TopBoardRegistry registry = new TopBoardRegistry();
        registry.offer(board(7), 10);

        List<int[]> all = registry.getAll();
        all.get(0)[0] = 999;

        assertEquals(7, registry.getAll().get(0)[0], "Mutating a board from getAll() must not affect internal state");
    }

    @Test
    void testNextForRepairReturnsDefensiveCopy() {
        TopBoardRegistry registry = new TopBoardRegistry();
        registry.offer(board(7), 10);

        int[] copy = registry.nextForRepair();
        copy[0] = 999;

        assertEquals(7, registry.nextForRepair()[0], "Mutating a board from nextForRepair() must not affect internal state");
    }

    @Test
    void testOfferCopiesInputBoardNotReference() {
        TopBoardRegistry registry = new TopBoardRegistry();
        int[] source = board(7);
        registry.offer(source, 10);

        source[0] = 999; // mutate caller's array after offering

        assertEquals(7, registry.getBest()[0], "offer() must copy the board contents, not alias the caller's array");
    }

    @Test
    void testGetRawRegistrySizeMatchesEntryCount() {
        TopBoardRegistry registry = new TopBoardRegistry();
        registry.offer(board(1), 10);
        registry.offer(board(2), 20);

        assertEquals(2, registry.getRawRegistry().size());
    }
}

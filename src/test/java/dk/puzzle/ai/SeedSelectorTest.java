package dk.puzzle.ai;

import dk.puzzle.util.PieceUtils;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for SeedSelector's stratified seed-selection strategy.
 *
 * <p>selectBest is only randomized in its Tier-3 (random-restart) shuffle;
 * everything else (scoring, sorting, elite/diverse filtering) is fully
 * deterministic given fixed inputs, so most tests here use fixed-seed
 * {@link Random} instances and assert exact/structural outcomes rather than
 * loose statistical bounds.</p>
 */
class SeedSelectorTest {

    // 20 flat board indices, pairwise non edge-adjacent (rows 0/3/6, columns
    // spaced by 2), so boards built purely from these indices never trigger
    // any internal adjacency comparison and are trivially edge-consistent.
    private static final int[] SHARED_INDICES = {
            0, 2, 4, 6, 8, 10, 12, 14,
            48, 50, 52, 54, 56, 58, 60, 62,
            96, 98, 100, 102
    };

    private int originalElitePct;
    private int originalDiversePct;

    @BeforeEach
    void captureDefaults() throws Exception {
        originalElitePct = readPrivateStaticInt("elitePct");
        originalDiversePct = readPrivateStaticInt("diversePct");
    }

    @AfterEach
    void restoreDefaults() {
        // elitePct/diversePct are static, so leaking a mutation here could
        // silently affect any other test class that runs later in the JVM.
        SeedSelector.setElitePct(originalElitePct);
        SeedSelector.setDiversePct(originalDiversePct);
    }

    private int readPrivateStaticInt(String fieldName) throws Exception {
        Field field = SeedSelector.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        return (int) field.get(null);
    }

    private int[] identityBuildOrder() {
        int[] order = new int[256];
        for (int i = 0; i < 256; i++) order[i] = i;
        return order;
    }

    /** Builds {@code count} boards, each placing a unique value at every SHARED_INDICES slot. */
    private List<int[]> buildDiverseBoardSet(int count) {
        List<int[]> boards = new ArrayList<>(count);
        for (int b = 0; b < count; b++) {
            int[] board = new int[256];
            Arrays.fill(board, -1);
            for (int slot = 0; slot < SHARED_INDICES.length; slot++) {
                board[SHARED_INDICES[slot]] = 1 + b * 1000 + slot;
            }
            boards.add(board);
        }
        return boards;
    }

    /** A reference board whose values at SHARED_INDICES never collide with buildDiverseBoardSet output. */
    private int[] buildOutOfRangeReferenceBoard() {
        int[] ref = new int[256];
        Arrays.fill(ref, -1);
        for (int slot = 0; slot < SHARED_INDICES.length; slot++) {
            ref[SHARED_INDICES[slot]] = 90000 + slot;
        }
        return ref;
    }

    // ── setElitePct / setDiversePct ──

    @Test
    void testSetElitePctClampsToLowerBound() throws Exception {
        SeedSelector.setElitePct(1);
        assertEquals(5, readPrivateStaticInt("elitePct"), "elitePct must clamp to the documented minimum of 5");
    }

    @Test
    void testSetElitePctClampsToUpperBound() throws Exception {
        SeedSelector.setElitePct(999);
        assertEquals(60, readPrivateStaticInt("elitePct"), "elitePct must clamp to the documented maximum of 60");
    }

    @Test
    void testSetElitePctAcceptsValueWithinRange() throws Exception {
        SeedSelector.setElitePct(30);
        assertEquals(30, readPrivateStaticInt("elitePct"));
    }

    @Test
    void testSetDiversePctClampsToLowerBound() throws Exception {
        SeedSelector.setDiversePct(0);
        assertEquals(10, readPrivateStaticInt("diversePct"), "diversePct must clamp to the documented minimum of 10");
    }

    @Test
    void testSetDiversePctClampsToUpperBound() throws Exception {
        SeedSelector.setDiversePct(1000);
        assertEquals(70, readPrivateStaticInt("diversePct"), "diversePct must clamp to the documented maximum of 70");
    }

    @Test
    void testSetDiversePctAcceptsValueWithinRange() throws Exception {
        SeedSelector.setDiversePct(40);
        assertEquals(40, readPrivateStaticInt("diversePct"));
    }

    // ── scoreBoard ──

    @Test
    void testScoreBoardOnEmptyBoardUsesDepthOnly() {
        int[] board = new int[256];
        Arrays.fill(board, -1);

        assertEquals(500, SeedSelector.scoreBoard(board, 5),
                "An all-empty board contributes zero position score and zero danger penalty");
    }

    @Test
    void testScoreBoardWeightsCenterHigherThanCorner() {
        int[] cornerBoard = new int[256];
        Arrays.fill(cornerBoard, -1);
        cornerBoard[0] = 111; // row0,col0 -> distFromEdge 0 -> weight 1

        int[] centerBoard = new int[256];
        Arrays.fill(centerBoard, -1);
        centerBoard[119] = 111; // row7,col7 -> distFromEdge 7 -> weight 8

        int cornerScore = SeedSelector.scoreBoard(cornerBoard, 0);
        int centerScore = SeedSelector.scoreBoard(centerBoard, 0);

        assertEquals(2, cornerScore, "Corner piece score = 0*100 + weight(1)*2 - 0");
        assertEquals(16, centerScore, "Center piece score = 0*100 + weight(8)*2 - 0");
        assertTrue(centerScore > cornerScore, "Interior placements must score higher than edge placements");
    }

    @Test
    void testScoreBoardAppliesDangerPenaltyForEnclosedEmptyCell() {
        int[] board = new int[256];
        Arrays.fill(board, -1);
        // Index 85 (row5,col5) is left empty but surrounded on all 4 sides.
        board[69] = 123;  // north
        board[101] = 123; // south
        board[84] = 123;  // west
        board[86] = 123;  // east

        // posScore = weight(69)=5 + weight(101)=6 + weight(84)=5 + weight(86)=6 = 22
        // dangerPenalty = (filled(4) - 2) * 5 = 10, only from index 85
        // score = depth(7)*100 + 22*2 - 10 = 734
        assertEquals(734, SeedSelector.scoreBoard(board, 7),
                "An empty cell surrounded on all 4 sides must incur the danger penalty");
    }

    // ── isEdgeConsistent ──

    @Test
    void testIsEdgeConsistentTrueForEmptyBoard() {
        int[] board = new int[256];
        Arrays.fill(board, -1);

        assertTrue(SeedSelector.isEdgeConsistent(board));
    }

    @Test
    void testIsEdgeConsistentTrueForMatchingAdjacentPieces() {
        int[] board = new int[256];
        Arrays.fill(board, -1);
        board[0] = PieceUtils.pack(1, 2, 3, 4);   // East=2
        board[1] = PieceUtils.pack(5, 6, 7, 2);   // West=2 matches
        board[16] = PieceUtils.pack(3, 8, 9, 10); // North=3 matches South of index 0

        assertTrue(SeedSelector.isEdgeConsistent(board));
    }

    @Test
    void testIsEdgeConsistentFalseForHorizontalMismatch() {
        int[] board = new int[256];
        Arrays.fill(board, -1);
        board[0] = PieceUtils.pack(1, 2, 3, 4); // East=2
        board[1] = PieceUtils.pack(5, 6, 7, 9); // West=9, mismatches

        assertFalse(SeedSelector.isEdgeConsistent(board));
    }

    @Test
    void testIsEdgeConsistentFalseForVerticalMismatch() {
        int[] board = new int[256];
        Arrays.fill(board, -1);
        board[0] = PieceUtils.pack(1, 2, 3, 4);  // South=3
        board[16] = PieceUtils.pack(9, 6, 7, 8); // North=9, mismatches

        assertFalse(SeedSelector.isEdgeConsistent(board));
    }

    @Test
    void testIsEdgeConsistentTreatsSurgeonHoleAsEmpty() {
        // -2 "surgeon hole" markers must be skipped like -1, matching
        // EternitySolver.verifyBoardStrict's convention, not bit-masked and
        // compared as if they were a real piece's packed colors.
        int[] board = new int[256];
        Arrays.fill(board, -1);
        board[0] = PieceUtils.pack(1, 2, 3, 4);
        board[1] = -2;

        assertTrue(SeedSelector.isEdgeConsistent(board),
                "A -2 hole marker must be treated as empty, not compared as a real piece's colors");
    }

    @Test
    void testIsEdgeConsistentTreatsSurgeonHoleOnSouthNeighborAsEmpty() {
        int[] board = new int[256];
        Arrays.fill(board, -1);
        board[0] = PieceUtils.pack(1, 2, 3, 4);
        board[16] = -2;

        assertTrue(SeedSelector.isEdgeConsistent(board),
                "A -2 hole marker on the south neighbor must be treated as empty, not compared as a real piece's colors");
    }

    // ── hammingDistance ──

    @Test
    void testHammingDistanceCountsOnlyCellsPlacedInBoth() {
        int[] a = new int[256];
        int[] b = new int[256];
        Arrays.fill(a, -1);
        Arrays.fill(b, -1);

        a[0] = 100; b[0] = 100; // both placed, equal -> no diff
        a[1] = 100; b[1] = 200; // both placed, differ -> diff
        a[2] = 100; b[2] = -1;  // only a placed -> no diff
        a[3] = -1;  b[3] = 100; // only b placed -> no diff
        a[5] = 300; b[5] = 301; // both placed, differ -> diff

        assertEquals(2, SeedSelector.hammingDistance(a, b));
    }

    @Test
    void testHammingDistanceZeroForIdenticalEmptyBoards() {
        int[] a = new int[256];
        int[] b = new int[256];
        Arrays.fill(a, -1);
        Arrays.fill(b, -1);

        assertEquals(0, SeedSelector.hammingDistance(a, b));
    }

    // ── selectBest ──

    @Test
    void testSelectBestReturnsEmptyListForEmptyInput() {
        List<int[]> result = SeedSelector.selectBest(
                new ArrayList<>(), new int[0], 5, identityBuildOrder(), null, new Random(1));

        assertTrue(result.isEmpty());
    }

    @Test
    void testSelectBestAllInconsistentBoardsReturnsEmptyResult() {
        int[] badBoard1 = new int[256];
        Arrays.fill(badBoard1, -1);
        badBoard1[0] = PieceUtils.pack(1, 2, 3, 4);
        badBoard1[1] = PieceUtils.pack(5, 6, 7, 9); // mismatches badBoard1[0]'s East

        int[] badBoard2 = new int[256];
        Arrays.fill(badBoard2, -1);
        badBoard2[16] = PieceUtils.pack(1, 2, 3, 4);
        badBoard2[0] = PieceUtils.pack(9, 6, 7, 8); // mismatches badBoard2[16]'s North

        List<int[]> allBoards = Arrays.asList(badBoard1, badBoard2);
        int[] threadDepths = {0, 0};

        List<int[]> result = SeedSelector.selectBest(
                allBoards, threadDepths, 3, identityBuildOrder(), null, new Random(1));

        assertTrue(result.isEmpty(), "No tier ever admits an edge-inconsistent board, including the fallback pass");
    }

    @Test
    void testSelectBestReturnsOnlyEdgeConsistentBoards() {
        List<int[]> goodBoards = buildDiverseBoardSet(3);

        int[] badBoard = new int[256];
        Arrays.fill(badBoard, -1);
        badBoard[0] = PieceUtils.pack(1, 2, 3, 4);
        badBoard[1] = PieceUtils.pack(5, 6, 7, 9); // mismatches

        List<int[]> allBoards = new ArrayList<>(goodBoards);
        allBoards.add(badBoard);
        allBoards.add(badBoard);
        int[] threadDepths = {0, 1, 2, 0, 0};

        List<int[]> result = SeedSelector.selectBest(
                allBoards, threadDepths, 3, identityBuildOrder(), null, new Random(1));

        assertFalse(result.isEmpty());
        for (int[] board : result) {
            assertTrue(SeedSelector.isEdgeConsistent(board), "Every returned board must be edge-consistent");
        }
    }

    @Test
    void testSelectBestRespectsTargetCountWhenSupplyIsAmple() {
        SeedSelector.setElitePct(15);
        SeedSelector.setDiversePct(55);

        List<int[]> boards = buildDiverseBoardSet(8);
        int[] threadDepths = {0, 1, 2, 3, 4, 5, 6, 7};

        // With targetCount=10: eliteCount=1, diverseCount=5, restartCount=4 -> sums to exactly 10.
        List<int[]> result = SeedSelector.selectBest(
                boards, threadDepths, 10, identityBuildOrder(), buildOutOfRangeReferenceBoard(), new Random(2));

        assertEquals(10, result.size());
        for (int[] board : result) {
            assertNotNull(board);
            assertEquals(256, board.length);
        }
    }

    @Test
    void testSelectBestSingleCandidateWithLargeTargetCountReturnsFewerThanRequested() {
        List<int[]> boards = buildDiverseBoardSet(1);
        int[] threadDepths = {0};

        List<int[]> result = SeedSelector.selectBest(
                boards, threadDepths, 5, identityBuildOrder(), null, new Random(3));

        assertFalse(result.isEmpty(), "The single candidate should still be selected as the elite pick");
        assertTrue(result.size() < 5,
                "With only one supply board, selectBest cannot manufacture 5 distinct results");
        for (int[] board : result) {
            assertArrayEquals(boards.get(0), board);
        }
    }

    @Test
    void testSelectBestEliteTierPicksHighestScoringBoardFirst() {
        List<int[]> boards = buildDiverseBoardSet(3);
        int[] threadDepths = {0, 1, 2}; // board index 2 has the highest depth -> highest score

        List<int[]> result = SeedSelector.selectBest(
                boards, threadDepths, 1, identityBuildOrder(), null, new Random(4));

        assertFalse(result.isEmpty());
        assertArrayEquals(boards.get(2), result.get(0),
                "The first (elite) slot must be the highest-scoring board");
    }

    @Test
    void testSelectBestIsDeterministicForSameRandomSeed() {
        List<int[]> boards = buildDiverseBoardSet(6);
        int[] threadDepths = {0, 1, 2, 3, 4, 5};
        int[] buildOrder = identityBuildOrder();
        int[] reference = buildOutOfRangeReferenceBoard();

        List<int[]> resultA = SeedSelector.selectBest(boards, threadDepths, 4, buildOrder, reference, new Random(777));
        List<int[]> resultB = SeedSelector.selectBest(boards, threadDepths, 4, buildOrder, reference, new Random(777));

        assertEquals(resultA.size(), resultB.size());
        for (int i = 0; i < resultA.size(); i++) {
            assertArrayEquals(resultA.get(i), resultB.get(i),
                    "Same seed and same inputs must produce the same output board at position " + i);
        }
    }

    @Test
    void testSelectBestNeverExceedsTargetCountEvenWhenTierQuotasEachFloorToOne() {
        // Each tier count is Math.max(1, targetCount * pct / 100), so for a
        // small targetCount, elite+diverse+restart can each floor up to 1 and
        // sum to more than targetCount. Every tier's loop is capped on
        // result.size() < targetCount as a hard ceiling, so selectBest must
        // never return more boards than requested regardless of how the
        // individual tier quotas add up.
        SeedSelector.setElitePct(15);
        SeedSelector.setDiversePct(55);

        List<int[]> boards = buildDiverseBoardSet(5);
        int[] threadDepths = {0, 1, 2, 3, 4};

        List<int[]> result = SeedSelector.selectBest(
                boards, threadDepths, 2, identityBuildOrder(), buildOutOfRangeReferenceBoard(), new Random(5));

        assertEquals(2, result.size(),
                "eliteCount(1) + diverseCount(1) + restartCount(1) = 3 would exceed targetCount(2) without the hard cap");
    }

    @Test
    void testSelectBestDoesNotMutateInputBoards() {
        List<int[]> boards = buildDiverseBoardSet(3);
        List<int[]> snapshots = new ArrayList<>();
        for (int[] board : boards) {
            snapshots.add(Arrays.copyOf(board, 256));
        }
        int[] threadDepths = {0, 1, 2};

        SeedSelector.selectBest(boards, threadDepths, 2, identityBuildOrder(), null, new Random(5));

        for (int i = 0; i < boards.size(); i++) {
            assertArrayEquals(snapshots.get(i), boards.get(i), "selectBest must not mutate its input boards");
        }
    }

    // ── STAGE 2: conflict-evaluated elite selection ──

    /** Counts how many of the first `filledSlots` SHARED_INDICES entries are actually placed. */
    private int[] buildBoardWithDepth(int boardId, int filledSlots) {
        int[] board = new int[256];
        Arrays.fill(board, -1);
        for (int slot = 0; slot < filledSlots; slot++) {
            board[SHARED_INDICES[slot]] = 1 + boardId * 1000 + slot;
        }
        return board;
    }

    @Test
    void testEvaluatedPoolIsDepthStratifiedNotFlatTopN() {
        SeedSelector.setElitePct(15);
        SeedSelector.setDiversePct(55);

        // 20 "deep" seeds (depth 20, all SHARED_INDICES filled) that fill badly,
        // plus 1 "shallow" seed (depth 10) that fills cleanly. A flat top-N pool
        // (evalPoolSize=5) ranked by scoreBoard() would be entirely deep seeds,
        // since depthReached*100 swamps everything else -- the shallow-but-clean
        // seed would never even be evaluated, let alone win the elite slot.
        List<int[]> boards = new ArrayList<>();
        int[] threadDepths = new int[21];
        for (int i = 0; i < 20; i++) {
            boards.add(buildBoardWithDepth(i, 20));
            threadDepths[i] = 20;
        }
        int[] shallowCleanSeed = buildBoardWithDepth(999, 10);
        boards.add(shallowCleanSeed);
        threadDepths[20] = 10;

        SeedSelector.ConflictEvaluator evaluator = seedBoard -> {
            int filled = 0;
            for (int idx : SHARED_INDICES) if (seedBoard[idx] != -1) filled++;
            return filled >= 20 ? 50 : 3; // deep seeds fill badly; the shallow one fills cleanly
        };

        // targetCount=10, elitePct=15 -> eliteCount = max(1, 1) = 1: exactly one
        // elite slot, so the winner is unambiguous.
        List<int[]> result = SeedSelector.selectBest(
                boards, threadDepths, 10, identityBuildOrder(), null, new Random(1),
                evaluator, 5);

        assertTrue(SeedSelector.getLastEvaluatedCount() >= 2,
                "The depth-10 seed must be part of the evaluated pool despite 20 higher-scoring depth-20 seeds");
        assertEquals(3, SeedSelector.getLastBestConflicts(),
                "The shallow-but-clean seed's conflict count must be visible to the evaluator");
        assertArrayEquals(shallowCleanSeed, result.get(0),
                "The single elite slot must go to the shallow-but-clean seed, not one of the deeper-but-dirtier ones");
    }

    @Test
    void testEvaluatedEliteTierDoesNotContainDuplicateBoards() {
        SeedSelector.setElitePct(15);
        SeedSelector.setDiversePct(55);

        // Two seeds with IDENTICAL content (same board content can legitimately
        // appear twice in a GPU batch) both tie for the best evaluated conflict
        // count, plus one distinct seed with a worse-but-still-evaluated count.
        // With eliteCount=2, a correct implementation must not burn both elite
        // slots on the same board twice - it should dedupe and pull in the
        // distinct third seed for the second slot. Elite entries are always the
        // first `eliteCount` entries of result (added before Diverse/Restart/
        // Fallback ever run), so checking result.get(0)/(1) directly isolates
        // the elite tier from those later, unrelated tiers.
        int[] boardA = buildBoardWithDepth(1, 20);
        int[] boardADuplicate = buildBoardWithDepth(1, 20); // identical content, different array instance
        int[] boardB = buildBoardWithDepth(2, 20);

        List<int[]> boards = List.of(boardA, boardADuplicate, boardB);
        int[] threadDepths = {20, 20, 20};

        SeedSelector.ConflictEvaluator evaluator = seedBoard ->
                Arrays.equals(seedBoard, boardB) ? 10 : 3; // A/A' tie for best; B is worse but usable

        // targetCount=14, elitePct=15 -> eliteCount = 14*15/100 = 2 exactly.
        List<int[]> result = SeedSelector.selectBest(
                boards, threadDepths, 14, identityBuildOrder(), null, new Random(1),
                evaluator, 10);

        assertTrue(result.size() >= 2, "Expected at least the 2 elite slots to be filled");
        assertArrayEquals(boardA, result.get(0), "First elite slot should be the (tied-for-best) duplicate-content board");
        assertArrayEquals(boardB, result.get(1),
                "Second elite slot must be the genuinely distinct board, not a second copy of the first");
    }
}

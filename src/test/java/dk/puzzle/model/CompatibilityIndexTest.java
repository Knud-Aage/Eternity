package dk.puzzle.model;

import dk.puzzle.util.PieceUtils;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.BitSet;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for CompatibilityIndex's bitmask-based candidate lookup.
 *
 * The index is built from two parallel 1024-length arrays (orientation ->
 * packed piece, orientation -> physical id), mirroring the layout
 * PieceInventory produces (256 physical pieces x 4 rotations). Rather than
 * relying on real rotation data, each test builds a controlled 1024-slot
 * array where every slot defaults to a "filler" piece and only a handful of
 * slots are overridden with distinctive edge colors, so expected matches can
 * be pinned down exactly.
 */
class CompatibilityIndexTest {

    private static final int FILLER_COLOR = 1;

    /** A 1024-slot orientation array where every slot is the filler piece (N=E=S=W=1). */
    private int[] newFilledOrientations() {
        int[] orientations = new int[1024];
        Arrays.fill(orientations, PieceUtils.pack(FILLER_COLOR, FILLER_COLOR, FILLER_COLOR, FILLER_COLOR));
        return orientations;
    }

    /** Maps orientation index i to physical id i/4, matching PieceInventory's real layout. */
    private int[] newPhysicalMapping() {
        int[] mapping = new int[1024];
        for (int i = 0; i < 1024; i++) {
            mapping[i] = i / 4;
        }
        return mapping;
    }

    private boolean[] noneUsed() {
        return new boolean[256];
    }

    @Test
    void testConstantsMatchDocumentedEncoding() {
        assertEquals(-1, CompatibilityIndex.WILDCARD, "WILDCARD must be -1 per javadoc");
        assertEquals(0, CompatibilityIndex.N);
        assertEquals(1, CompatibilityIndex.E);
        assertEquals(2, CompatibilityIndex.S);
        assertEquals(3, CompatibilityIndex.W);
    }

    @Test
    void testAllWildcardReturnsEveryOrientationWhenNoneUsed() {
        CompatibilityIndex index = new CompatibilityIndex(newFilledOrientations(), newPhysicalMapping());

        BitSet all = index.candidatesFor(CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD,
                CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD);

        assertEquals(1024, all.cardinality(), "With no constraints, all 1024 orientation slots are candidates");
    }

    @Test
    void testCandidatesForFiltersOnSingleNorthConstraint() {
        int[] orientations = newFilledOrientations();
        orientations[0] = PieceUtils.pack(2, 3, 4, 5); // physical 0, orientation slot 0
        CompatibilityIndex index = new CompatibilityIndex(orientations, newPhysicalMapping());

        BitSet north2 = index.candidatesFor(2, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD);

        assertEquals(1, north2.cardinality(), "Only orientation slot 0 has North=2");
        assertTrue(north2.get(0));
    }

    @Test
    void testCandidatesForFiltersOnEastSouthWestConstraintsIndependently() {
        int[] orientations = newFilledOrientations();
        orientations[8] = PieceUtils.pack(9, 6, 7, 8); // physical 2, orientation slot 8
        CompatibilityIndex index = new CompatibilityIndex(orientations, newPhysicalMapping());

        assertTrue(index.candidatesFor(CompatibilityIndex.WILDCARD, 6, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD).get(8),
                "East=6 constraint must find slot 8");
        assertTrue(index.candidatesFor(CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, 7, CompatibilityIndex.WILDCARD).get(8),
                "South=7 constraint must find slot 8");
        assertTrue(index.candidatesFor(CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, 8).get(8),
                "West=8 constraint must find slot 8");
    }

    @Test
    void testCandidatesForCombinesAllFourConstraintsWithLogicalAnd() {
        int[] orientations = newFilledOrientations();
        orientations[0] = PieceUtils.pack(2, 3, 4, 5);
        orientations[4] = PieceUtils.pack(2, 10, 11, 12); // shares North=2 with slot 0 but differs elsewhere
        CompatibilityIndex index = new CompatibilityIndex(orientations, newPhysicalMapping());

        BitSet north2Only = index.candidatesFor(2, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD);
        assertEquals(2, north2Only.cardinality(), "Two orientation slots share North=2 before narrowing further");

        BitSet exact = index.candidatesFor(2, 3, 4, 5);
        assertEquals(1, exact.cardinality(), "Adding East/South/West constraints must narrow to the single exact match");
        assertTrue(exact.get(0));
    }

    @Test
    void testCandidatesForReturnsEmptyWhenNoOrientationMatches() {
        CompatibilityIndex index = new CompatibilityIndex(newFilledOrientations(), newPhysicalMapping());

        BitSet none = index.candidatesFor(21, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD);

        assertTrue(none.isEmpty(), "No filler slot uses color 21, so this constraint must match nothing");
    }

    @Test
    void testCandidatesForSupportsBorderColorAndMaxValidColorIndex() {
        int[] orientations = newFilledOrientations();
        orientations[12] = PieceUtils.pack(0, 0, 0, 0);  // border color on all sides, physical 3
        orientations[16] = PieceUtils.pack(22, 1, 1, 1); // highest valid inner color, physical 4
        CompatibilityIndex index = new CompatibilityIndex(orientations, newPhysicalMapping());

        assertTrue(index.candidatesFor(0, 0, 0, 0).get(12), "Border color (0) must be indexable like any other color");
        assertTrue(index.candidatesFor(22, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD).get(16),
                "Highest documented valid inner color (22) must be indexable without overflow");
    }

    @Test
    void testCandidatesForReturnsIndependentBitSetsAcrossCalls() {
        CompatibilityIndex index = new CompatibilityIndex(newFilledOrientations(), newPhysicalMapping());

        BitSet first = index.candidatesFor(CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD);
        first.clear();
        BitSet second = index.candidatesFor(CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD);

        assertEquals(1024, second.cardinality(), "Mutating a previously returned BitSet must not leak into later independent calls");
    }

    @Test
    void testAndNotUsedRemovesOnlyOrientationsOfTheUsedPhysicalPiece() {
        int[] orientations = newFilledOrientations();
        orientations[0] = PieceUtils.pack(2, 3, 4, 5); // physical 0
        orientations[4] = PieceUtils.pack(2, 6, 7, 8); // physical 1
        CompatibilityIndex index = new CompatibilityIndex(orientations, newPhysicalMapping());

        BitSet candidates = index.candidatesFor(2, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD);
        boolean[] used = noneUsed();
        used[0] = true; // mark physical piece 0 as used

        index.andNotUsed(candidates, used);

        assertFalse(candidates.get(0), "Orientation slot belonging to a used physical piece must be removed");
        assertTrue(candidates.get(4), "Orientation slot of an unused physical piece must remain");
    }

    @Test
    void testAndNotUsedWithNothingUsedIsANoOp() {
        CompatibilityIndex index = new CompatibilityIndex(newFilledOrientations(), newPhysicalMapping());

        BitSet candidates = index.candidatesFor(CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD);
        int before = candidates.cardinality();
        index.andNotUsed(candidates, noneUsed());

        assertEquals(before, candidates.cardinality(), "With nothing marked used, andNotUsed must not remove anything");
    }

    @Test
    void testAndNotUsedWithEveryPhysicalPieceUsedEmptiesCandidates() {
        CompatibilityIndex index = new CompatibilityIndex(newFilledOrientations(), newPhysicalMapping());

        BitSet candidates = index.candidatesFor(CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD);
        boolean[] used = new boolean[256];
        Arrays.fill(used, true);
        index.andNotUsed(candidates, used);

        assertTrue(candidates.isEmpty(), "Marking every physical piece as used must empty a full candidate set");
    }

    @Test
    void testHasAnyCandidateTrueWhenAnUnusedMatchExists() {
        int[] orientations = newFilledOrientations();
        orientations[0] = PieceUtils.pack(2, 3, 4, 5);
        CompatibilityIndex index = new CompatibilityIndex(orientations, newPhysicalMapping());

        assertTrue(index.hasAnyCandidate(2, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, noneUsed()));
    }

    @Test
    void testHasAnyCandidateFalseWhenTheOnlyMatchIsUsed() {
        int[] orientations = newFilledOrientations();
        orientations[0] = PieceUtils.pack(2, 3, 4, 5); // physical 0, orientation slots 0-3
        CompatibilityIndex index = new CompatibilityIndex(orientations, newPhysicalMapping());

        boolean[] used = noneUsed();
        used[0] = true;

        assertFalse(index.hasAnyCandidate(2, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, used),
                "The only piece with North=2 is used, so no candidate should remain");
    }

    @Test
    void testHasAnyCandidateFalseWhenColorHasNoMatchAtAll() {
        CompatibilityIndex index = new CompatibilityIndex(newFilledOrientations(), newPhysicalMapping());

        assertFalse(index.hasAnyCandidate(21, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, CompatibilityIndex.WILDCARD, noneUsed()));
    }

}

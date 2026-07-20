package dk.puzzle.model;

import dk.puzzle.util.PieceUtils;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for PieceInventory's construction of the 1024-orientation index,
 * corner/edge/interior classification, edge-color maps and color-frequency
 * table.
 *
 * A single hand-crafted 256-piece base array is reused across most tests so
 * expected results can be computed by hand and asserted exactly rather than
 * loosely:
 *   - physical piece 0: pack(0,1,2,0)  -> two border edges (North, West)   -> "corner"
 *   - physical piece 1: pack(0,1,2,3)  -> one border edge (North)          -> "edge"
 *   - physical piece 2: pack(1,2,3,4)  -> no border edges                  -> "interior"
 *   - physical piece 7: pack(50,1,2,3) -> no border edges, but color 50 is
 *                                          otherwise unused, isolating the
 *                                          colorFrequency assertion
 *   - every other physical piece (3-6, 8-255): pack(1,2,3,4) filler, same
 *     shape as physical piece 2 (also "interior"), never touches color 0 or 50
 */
class PieceInventoryTest {

    private int[] buildCraftedBasePieces() {
        int[] basePieces = new int[256];
        Arrays.fill(basePieces, PieceUtils.pack(1, 2, 3, 4));
        basePieces[0] = PieceUtils.pack(0, 1, 2, 0);
        basePieces[1] = PieceUtils.pack(0, 1, 2, 3);
        basePieces[2] = PieceUtils.pack(1, 2, 3, 4);
        basePieces[7] = PieceUtils.pack(50, 1, 2, 3);
        return basePieces;
    }

    @Test
    void testConstructorSizesOrientationAndMappingArraysTo1024() {
        PieceInventory inventory = new PieceInventory(buildCraftedBasePieces());

        assertEquals(1024, inventory.allOrientations.length);
        assertEquals(1024, inventory.physicalMapping.length);
    }

    @Test
    void testPhysicalMappingAssignsFourOrientationSlotsPerPhysicalPiece() {
        PieceInventory inventory = new PieceInventory(buildCraftedBasePieces());

        for (int physId : new int[]{0, 1, 2, 7, 255}) {
            for (int r = 0; r < 4; r++) {
                assertEquals(physId, inventory.physicalMapping[physId * 4 + r],
                        "Orientation slot " + (physId * 4 + r) + " must map back to physical piece " + physId);
            }
        }
    }

    @Test
    void testAllOrientationsAppliesRotateChainStartingFromTheBasePiece() {
        int[] basePieces = buildCraftedBasePieces();
        PieceInventory inventory = new PieceInventory(basePieces);

        for (int physId : new int[]{0, 1, 7}) {
            int expected = basePieces[physId];
            for (int r = 0; r < 4; r++) {
                assertEquals(expected, inventory.allOrientations[physId * 4 + r],
                        "Orientation slot " + (physId * 4 + r) + " must be physical piece " + physId + " rotated " + r + " time(s)");
                expected = PieceUtils.rotate(expected);
            }
        }
    }

    @Test
    void testCornersContainsExactlyTheOrientationsOfTwoBorderPieces() {
        PieceInventory inventory = new PieceInventory(buildCraftedBasePieces());

        // Physical piece 0 is the only crafted piece with 2 border edges; border
        // count is rotation-invariant, so all 4 of its orientation slots qualify.
        assertEquals(List.of(0, 1, 2, 3), inventory.corners);
    }

    @Test
    void testEdgesContainsExactlyTheOrientationsOfOneBorderPiece() {
        PieceInventory inventory = new PieceInventory(buildCraftedBasePieces());

        // Physical piece 1 is the only crafted piece with exactly 1 border edge.
        assertEquals(List.of(4, 5, 6, 7), inventory.edges);
    }

    @Test
    void testClassificationIsExhaustiveAndMutuallyExclusive() {
        PieceInventory inventory = new PieceInventory(buildCraftedBasePieces());

        assertEquals(4, inventory.corners.size());
        assertEquals(4, inventory.edges.size());
        assertEquals(1024 - 4 - 4, inventory.interior.size(), "Every orientation slot not classified as corner/edge must be interior");

        Set<Integer> corners = new HashSet<>(inventory.corners);
        Set<Integer> edges = new HashSet<>(inventory.edges);
        Set<Integer> interior = new HashSet<>(inventory.interior);

        assertTrue(java.util.Collections.disjoint(corners, edges), "Corners and edges must not overlap");
        assertTrue(java.util.Collections.disjoint(corners, interior), "Corners and interior must not overlap");
        assertTrue(java.util.Collections.disjoint(edges, interior), "Edges and interior must not overlap");

        // Physical piece 2's 4 orientation slots (8-11) and physical piece 7's (28-31)
        // both have 0 border edges and must land in interior.
        assertTrue(interior.containsAll(List.of(8, 9, 10, 11)));
        assertTrue(interior.containsAll(List.of(28, 29, 30, 31)));
    }

    @Test
    void testEdgeToOrientationMapsAreExactForEachOfTheFourSides() {
        PieceInventory inventory = new PieceInventory(buildCraftedBasePieces());

        // Hand-computed rotations of physical pieces 0 and 1 (see class javadoc):
        //   orientation 0: N=0 E=1 S=2 W=0   orientation 4: N=0 E=1 S=2 W=3
        //   orientation 1: N=0 E=0 S=1 W=2   orientation 5: N=3 E=0 S=1 W=2
        //   orientation 2: N=2 E=0 S=0 W=1   orientation 6: N=2 E=3 S=0 W=1
        //   orientation 3: N=1 E=2 S=0 W=0   orientation 7: N=1 E=2 S=3 W=0
        // No other physical piece ever touches border color 0.
        assertEquals(Set.of(0, 1, 4), new HashSet<>(inventory.northEdgeToOrientations.get(0)), "North=0 map");
        assertEquals(Set.of(1, 2, 5), new HashSet<>(inventory.eastEdgeToOrientations.get(0)), "East=0 map");
        assertEquals(Set.of(2, 3, 6), new HashSet<>(inventory.southEdgeToOrientations.get(0)), "South=0 map");
        assertEquals(Set.of(0, 3, 7), new HashSet<>(inventory.westEdgeToOrientations.get(0)), "West=0 map");
    }

    @Test
    void testCompatibilityArrayMirrorsTheEdgeToOrientationMaps() {
        PieceInventory inventory = new PieceInventory(buildCraftedBasePieces());

        // compatibility[0..3] correspond to North/East/South/West respectively,
        // populated from the same per-orientation loop as the *EdgeToOrientations maps.
        assertEquals(new HashSet<>(inventory.northEdgeToOrientations.get(0)), new HashSet<>(inventory.compatibility[0][0]));
        assertEquals(new HashSet<>(inventory.eastEdgeToOrientations.get(0)), new HashSet<>(inventory.compatibility[1][0]));
        assertEquals(new HashSet<>(inventory.southEdgeToOrientations.get(0)), new HashSet<>(inventory.compatibility[2][0]));
        assertEquals(new HashSet<>(inventory.westEdgeToOrientations.get(0)), new HashSet<>(inventory.compatibility[3][0]));
    }

    @Test
    void testColorFrequencyCountsAUniqueColorAcrossAllFourRotations() {
        PieceInventory inventory = new PieceInventory(buildCraftedBasePieces());

        // Color 50 appears exactly once among physical piece 7's 4 edge values
        // (50,1,2,3); rotation only permutes which side it lands on, so across
        // the 4 rotations it is counted exactly 4 times, and nowhere else.
        assertEquals(4, inventory.colorFrequency[50]);
    }

    @Test
    void testColorCountSizingMatchesMaxColorPlusOne() {
        PieceInventory inventory = new PieceInventory(buildCraftedBasePieces());

        // Highest color used anywhere in the crafted base pieces is 50, so
        // colorFrequency/compatibility must be sized for indices [0..50].
        assertEquals(51, inventory.colorFrequency.length);
        assertEquals(51, inventory.compatibility[0].length);
    }

    @Test
    void testGetBaseColorsAtLowerBoundaryId() {
        int[] basePieces = buildCraftedBasePieces();
        PieceInventory inventory = new PieceInventory(basePieces);

        // physicalId=1 (1-based) -> arrayIndex 0 -> physical piece 0 (0,1,2,0)
        int[] colors = inventory.getBaseColors(1);

        assertArrayEquals(new int[]{0, 1, 2, 0}, colors);
    }

    @Test
    void testGetBaseColorsAtUpperBoundaryId() {
        int[] basePieces = buildCraftedBasePieces();
        PieceInventory inventory = new PieceInventory(basePieces);

        // physicalId=256 (max, 1-based) -> arrayIndex 255 -> filler piece (1,2,3,4)
        int[] colors = inventory.getBaseColors(256);

        assertArrayEquals(new int[]{1, 2, 3, 4}, colors);
    }

    @Test
    void testGetBaseColorsMatchesTheUniquelyMarkedPiece() {
        int[] basePieces = buildCraftedBasePieces();
        PieceInventory inventory = new PieceInventory(basePieces);

        // physicalId=8 -> arrayIndex 7 -> physical piece 7 (50,1,2,3)
        int[] colors = inventory.getBaseColors(8);

        assertArrayEquals(new int[]{50, 1, 2, 3}, colors);
    }
}

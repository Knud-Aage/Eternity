package dk.puzzle.util;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for PieceUtils' bit-packing, rotation and border-counting logic.
 * These are pure functions over 32-bit packed integers, so every test is a
 * plain input/output check with no fixtures or mocking required.
 */
class PieceUtilsTest {

    @Test
    void testConstantsHaveDocumentedValues() {
        assertEquals(0, PieceUtils.BORDER_COLOR, "Border color must be 0");
        assertEquals(0xFF, PieceUtils.WILDCARD, "Wildcard must be the max single-byte value 0xFF");
    }

    @Test
    void testPackAndUnpackRoundTripForTypicalValues() {
        int p = PieceUtils.pack(1, 2, 3, 4);

        assertEquals(1, PieceUtils.getNorth(p));
        assertEquals(2, PieceUtils.getEast(p));
        assertEquals(3, PieceUtils.getSouth(p));
        assertEquals(4, PieceUtils.getWest(p));
    }

    @Test
    void testPackAndUnpackRoundTripForAllZero() {
        int p = PieceUtils.pack(0, 0, 0, 0);

        assertEquals(0, PieceUtils.getNorth(p));
        assertEquals(0, PieceUtils.getEast(p));
        assertEquals(0, PieceUtils.getSouth(p));
        assertEquals(0, PieceUtils.getWest(p));
    }

    @Test
    void testPackAndUnpackRoundTripForMaxByteValues() {
        int p = PieceUtils.pack(255, 255, 255, 255);

        assertEquals(255, PieceUtils.getNorth(p));
        assertEquals(255, PieceUtils.getEast(p));
        assertEquals(255, PieceUtils.getSouth(p));
        assertEquals(255, PieceUtils.getWest(p));
        assertEquals(PieceUtils.WILDCARD, PieceUtils.getNorth(p), "0xFF round-trips to the WILDCARD sentinel value");
    }

    @Test
    void testPackAndUnpackRoundTripForDistinctBoundaryValuesPerEdge() {
        int p = PieceUtils.pack(0, 255, 1, 254);

        assertEquals(0, PieceUtils.getNorth(p));
        assertEquals(255, PieceUtils.getEast(p));
        assertEquals(1, PieceUtils.getSouth(p));
        assertEquals(254, PieceUtils.getWest(p));
    }

    @Test
    void testPackMasksValuesOutsideByteRangeToLowByte() {
        // 256 == 0x100, masked with 0xFF becomes 0. -1 masked with 0xFF becomes 255.
        int p = PieceUtils.pack(256, -1, 511, -2);

        assertEquals(0, PieceUtils.getNorth(p), "256 & 0xFF must wrap to 0");
        assertEquals(255, PieceUtils.getEast(p), "-1 & 0xFF must mask to 255");
        assertEquals(255, PieceUtils.getSouth(p), "511 (0x1FF) & 0xFF must mask to 255");
        assertEquals(254, PieceUtils.getWest(p), "-2 & 0xFF must mask to 254");
    }

    @Test
    void testGettersExtractCorrectByteSegmentsFromRawInt() {
        // 0x01020304: North=0x01, East=0x02, South=0x03, West=0x04
        int raw = 0x01020304;

        assertEquals(1, PieceUtils.getNorth(raw));
        assertEquals(2, PieceUtils.getEast(raw));
        assertEquals(3, PieceUtils.getSouth(raw));
        assertEquals(4, PieceUtils.getWest(raw));
    }

    @Test
    void testGettersDoNotSignExtendHighBitBytes() {
        // 0x80 in any byte segment must read back as 128, not a negative value,
        // since getters use unsigned shifts (>>>) and mask with 0xFF.
        int p = PieceUtils.pack(0x80, 0x81, 0x82, 0x83);

        assertEquals(128, PieceUtils.getNorth(p));
        assertEquals(129, PieceUtils.getEast(p));
        assertEquals(130, PieceUtils.getSouth(p));
        assertEquals(131, PieceUtils.getWest(p));
    }

    @Test
    void testRotateAppliesDocumentedEdgeMapping() {
        // Javadoc: New North = Old West, New East = Old North,
        //          New South = Old East, New West = Old South.
        int original = PieceUtils.pack(1, 2, 3, 4); // N=1 E=2 S=3 W=4
        int rotated = PieceUtils.rotate(original);

        assertEquals(4, PieceUtils.getNorth(rotated), "New North must equal old West");
        assertEquals(1, PieceUtils.getEast(rotated), "New East must equal old North");
        assertEquals(2, PieceUtils.getSouth(rotated), "New South must equal old East");
        assertEquals(3, PieceUtils.getWest(rotated), "New West must equal old South");
        assertEquals(PieceUtils.pack(4, 1, 2, 3), rotated);
    }

    @Test
    void testRotateFourTimesReturnsToOriginalValueForAsymmetricPiece() {
        int p = PieceUtils.pack(1, 2, 3, 4);

        int r1 = PieceUtils.rotate(p);
        int r2 = PieceUtils.rotate(r1);
        int r3 = PieceUtils.rotate(r2);
        int r4 = PieceUtils.rotate(r3);

        assertNotEquals(p, r1, "A single rotation of an asymmetric piece must change its packed value");
        assertEquals(p, r4, "Four rotations (360 degrees) must return to the original packed value");
    }

    @Test
    void testRotateFourTimesReturnsToOriginalValueForUniformPiece() {
        int p = PieceUtils.pack(7, 7, 7, 7);

        int r4 = PieceUtils.rotate(PieceUtils.rotate(PieceUtils.rotate(PieceUtils.rotate(p))));

        assertEquals(p, r4, "Four rotations must be identity regardless of whether edges are uniform");
    }

    @Test
    void testRotateFourTimesReturnsToOriginalValueForBorderAndWildcardEdges() {
        int p = PieceUtils.pack(PieceUtils.BORDER_COLOR, PieceUtils.WILDCARD, PieceUtils.BORDER_COLOR, PieceUtils.WILDCARD);

        int r4 = PieceUtils.rotate(PieceUtils.rotate(PieceUtils.rotate(PieceUtils.rotate(p))));

        assertEquals(p, r4, "Four rotations must be identity for border/wildcard edge values too");
    }

    @Test
    void testRotatePreservesTheMultisetOfEdgeColors() {
        int p = PieceUtils.pack(10, 20, 30, 40);
        int rotated = PieceUtils.rotate(p);

        int[] before = {PieceUtils.getNorth(p), PieceUtils.getEast(p), PieceUtils.getSouth(p), PieceUtils.getWest(p)};
        int[] after = {PieceUtils.getNorth(rotated), PieceUtils.getEast(rotated), PieceUtils.getSouth(rotated), PieceUtils.getWest(rotated)};

        java.util.Arrays.sort(before);
        java.util.Arrays.sort(after);
        assertArrayEquals(before, after, "Rotation must permute edge colors, never add/drop/change one");
    }

    @Test
    void testGetBorderCountZeroBorders() {
        int p = PieceUtils.pack(1, 2, 3, 4);
        assertEquals(0, PieceUtils.getBorderCount(p));
    }

    @Test
    void testGetBorderCountOneBorderOnEachSideIndividually() {
        assertEquals(1, PieceUtils.getBorderCount(PieceUtils.pack(0, 1, 2, 3)), "Border on North only");
        assertEquals(1, PieceUtils.getBorderCount(PieceUtils.pack(1, 0, 2, 3)), "Border on East only");
        assertEquals(1, PieceUtils.getBorderCount(PieceUtils.pack(1, 2, 0, 3)), "Border on South only");
        assertEquals(1, PieceUtils.getBorderCount(PieceUtils.pack(1, 2, 3, 0)), "Border on West only");
    }

    @Test
    void testGetBorderCountTwoBordersAdjacentAndOpposite() {
        assertEquals(2, PieceUtils.getBorderCount(PieceUtils.pack(0, 1, 2, 0)), "Adjacent borders North+West (corner-shaped)");
        assertEquals(2, PieceUtils.getBorderCount(PieceUtils.pack(0, 1, 0, 3)), "Opposite borders North+South");
        assertEquals(2, PieceUtils.getBorderCount(PieceUtils.pack(1, 0, 2, 0)), "Opposite borders East+West");
    }

    @Test
    void testGetBorderCountThreeBorders() {
        assertEquals(3, PieceUtils.getBorderCount(PieceUtils.pack(0, 0, 0, 3)));
        assertEquals(3, PieceUtils.getBorderCount(PieceUtils.pack(3, 0, 0, 0)));
    }

    @Test
    void testGetBorderCountFourBorders() {
        assertEquals(4, PieceUtils.getBorderCount(PieceUtils.pack(0, 0, 0, 0)));
    }
}

package dk.roleplay;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.lang.reflect.Method;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.*;

class EternityIITest {

    @BeforeEach
    void setup() throws Exception {
        // Initialize private static fields via reflection if necessary, 
        // or ensure public setup methods are called.
        // For this mock-based test, we assume generateMockPieces is available.
        Method m = EternityII.class.getDeclaredMethod("generateMockPieces");
        m.setAccessible(true);
        m.invoke(null);
    }

    @Test
    void testConstraintBuilding() throws Exception {
        // Test corner macro-tile constraints (0,0)
        Method m = EternityII.class.getDeclaredMethod("buildConstraints", int.class);
        m.setAccessible(true);
        int[] constraints = (int[]) m.invoke(null, 0);

        // Piece at (0,0) of Macro (0,0) should have North and West as 0 (Border)
        int p0 = constraints[0];
        assertEquals(0, PieceUtils.getNorth(p0));
        assertEquals(0, PieceUtils.getWest(p0));

        // Piece at (3,3) of Macro (0,0) should be wildcard as it's internal to the macro
        int p15 = constraints[15];
        assertEquals(PieceUtils.WILDCARD, PieceUtils.getNorth(p15));
        assertEquals(PieceUtils.WILDCARD, PieceUtils.getEast(p15));
    }

    @Test
    void testScoring() throws Exception {
        Method calcFreq = EternityII.class.getDeclaredMethod("calculateFrequencies");
        calcFreq.setAccessible(true);
        calcFreq.invoke(null);

        Method scoreMethod = EternityII.class.getDeclaredMethod("scoreTile", int[].class);
        scoreMethod.setAccessible(true);

        int[] tile = new int[16];
        Arrays.fill(tile, PieceUtils.pack(1, 1, 1, 1));
        
        double score = (double) scoreMethod.invoke(null, (Object) tile);
        assertTrue(score > 0);
    }

    @Test
    void testInventoryConsistency() throws Exception {
        Method shuffle = EternityII.class.getDeclaredMethod("shuffleInventory");
        shuffle.setAccessible(true);
        
        int[] before = getInventory().clone();
        shuffle.invoke(null);
        int[] after = getInventory();
        
        assertFalse(Arrays.equals(before, after), "Shuffle should change order");
        assertEquals(before.length, after.length);
        
        // Ensure pieces are still valid packed integers
        for (int p : after) {
            assertTrue(PieceUtils.getNorth(p) <= 23);
        }
    }

    private int[] getInventory() throws Exception {
        java.lang.reflect.Field f = EternityII.class.getDeclaredField("inventory");
        f.setAccessible(true);
        return (int[]) f.get(null);
    }
}

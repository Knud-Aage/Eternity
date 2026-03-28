package dk.roleplay;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class PieceUtilsTest {

    @Test
    void testPackAndUnpack() {
        int n = 1, e = 2, s = 3, w = 4;
        int p = PieceUtils.pack(n, e, s, w);
        
        assertEquals(n, PieceUtils.getNorth(p));
        assertEquals(e, PieceUtils.getEast(p));
        assertEquals(s, PieceUtils.getSouth(p));
        assertEquals(w, PieceUtils.getWest(p));
    }

    @Test
    void testRotation() {
        // Original: N=1, E=2, S=3, W=4
        int p = PieceUtils.pack(1, 2, 3, 4);
        
        // After 90 deg clockwise: N=4, E=1, S=2, W=3
        int pRot = PieceUtils.rotate(p);
        
        assertEquals(4, PieceUtils.getNorth(pRot));
        assertEquals(1, PieceUtils.getEast(pRot));
        assertEquals(2, PieceUtils.getSouth(pRot));
        assertEquals(3, PieceUtils.getWest(pRot));
        
        // Full circle
        int pFull = PieceUtils.rotate(PieceUtils.rotate(PieceUtils.rotate(PieceUtils.rotate(p))));
        assertEquals(p, pFull);
    }
}

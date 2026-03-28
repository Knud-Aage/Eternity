package dk.roleplay;

import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class ValidatorTest {

    @Test
    void testCpuValidator() {
        CpuValidator validator = new CpuValidator();
        runValidatorTest(validator);
    }

    private void runValidatorTest(CandidateValidator validator) {
        // Create a 4x4 macro-tile where internal edges match
        // 1-1, 1-1, 1-1 ...
        int[] validTile = new int[16];
        for (int i = 0; i < 16; i++) {
            validTile[i] = PieceUtils.pack(1, 1, 1, 1);
        }

        // Create an invalid macro-tile (East-West mismatch)
        int[] invalidTile = new int[16];
        for (int i = 0; i < 16; i++) {
            invalidTile[i] = PieceUtils.pack(1, 1, 1, 1);
        }
        invalidTile[0] = PieceUtils.pack(1, 2, 1, 1); // East is 2, neighbor West is 1

        int[] batch = new int[32];
        System.arraycopy(validTile, 0, batch, 0, 16);
        System.arraycopy(invalidTile, 0, batch, 16, 16);

        List<int[]> results = validator.validate(batch);
        
        assertEquals(1, results.size(), "Should find exactly one valid tile");
        assertArrayEquals(validTile, results.get(0));
    }
}

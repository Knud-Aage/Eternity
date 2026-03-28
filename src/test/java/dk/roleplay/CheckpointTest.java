package dk.roleplay;

import org.junit.jupiter.api.Test;
import java.io.File;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.*;

class CheckpointTest {

    @Test
    void testSaveAndLoad() {
        int[] inventory = new int[1024];
        Arrays.fill(inventory, 42);
        
        int[][] board = new int[16][16];
        for(int i=0; i<16; i++) board[i] = new int[]{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

        CheckpointManager.save(inventory, board);
        
        CheckpointManager.CheckpointData data = CheckpointManager.load();
        
        assertNotNull(data);
        assertArrayEquals(inventory, data.inventory);
        for(int i=0; i<16; i++) {
            assertArrayEquals(board[i], data.mainBoard[i]);
        }
        
        // Cleanup
        new File("checkpoint.txt").delete();
    }
}

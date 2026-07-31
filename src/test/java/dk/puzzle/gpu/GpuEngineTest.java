package dk.puzzle.gpu;

import dk.puzzle.model.PieceInventory;
import dk.puzzle.util.PieceUtils;
import org.junit.jupiter.api.Test;
import sun.misc.Unsafe;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for GpuEngine's hardware-independent logic.
 *
 * <p>Every other code path in this class (initCUDA, runDeepDfs/runRepairMode
 * beyond the empty-input guard) calls directly into the CUDA
 * driver via JCuda static natives with no abstraction seam, and the
 * constructor performs real device initialization (cuInit, cuCtxCreate,
 * cuModuleLoad, VRAM allocation) as a side effect of construction. That
 * makes it hardware/integration-test territory, not unit-test territory —
 * exercising it here would require a physical GPU and risks colliding with
 * an already-running solver's CUDA context. It is intentionally not covered.</p>
 *
 * <p>The only logic that runs before any CUDA call is the "no elements"
 * early-return guard in runDeepDfs/runRepairMode. To reach it without
 * paying the constructor's real GPU-init cost, the instance below is
 * allocated via {@link Unsafe#allocateInstance}, which skips the
 * constructor entirely — safe here only because the guarded return path
 * never touches any of the (therefore still-null) device buffer fields.</p>
 */
class GpuEngineTest {

    private GpuEngine newUninitializedEngine() throws Exception {
        Field theUnsafe = Unsafe.class.getDeclaredField("theUnsafe");
        theUnsafe.setAccessible(true);
        Unsafe unsafe = (Unsafe) theUnsafe.get(null);
        return (GpuEngine) unsafe.allocateInstance(GpuEngine.class);
    }

    @Test
    void testRunDeepDfsWithNoSeedsSkipsGpuAndReturnsCurrentHighScore() throws Exception {
        GpuEngine engine = newUninitializedEngine();
        int[] bestBoardOut = new int[256];

        GpuEngine.GpuResult result = engine.runDeepDfs(Collections.emptyList(), 50, 187, bestBoardOut);

        assertEquals(187, result.newHighScore(), "Empty seed batch must pass the current high score straight through");
        assertFalse(result.solved(), "Empty seed batch cannot have solved the puzzle");
        assertEquals(0, result.stepsTaken(), "No GPU work was launched, so zero steps were taken");
        assertEquals(0, result.threadDepths().length, "No threads ran, so there are no per-thread depths");
    }

    @Test
    void testRunRepairModeWithNoBoardsSkipsGpuAndReturnsCurrentHighScore() throws Exception {
        GpuEngine engine = newUninitializedEngine();
        int[] bestBoardOut = new int[256];

        GpuEngine.GpuResult result = engine.runRepairMode(Collections.emptyList(), 202, bestBoardOut);

        assertEquals(202, result.newHighScore(), "Empty variation batch must pass the current high score straight through");
        assertFalse(result.solved(), "Empty variation batch cannot have solved the puzzle");
        assertEquals(0, result.stepsTaken(), "No GPU work was launched, so zero steps were taken");
        assertEquals(0, result.threadDepths().length, "Repair mode never populates per-thread depths");
    }

    /**
     * blackwoodHeuristicSortedOrder() is pure host-side logic (no CUDA call),
     * unlike everything else in this class -- worth locking down directly
     * since a wrong sort direction or an incomplete permutation would silently
     * defeat the whole point of re-enabling c_heuristicRequired (see
     * initCUDA's 2026-07-31 comment): the kernel would go back to seeing
     * candidates in an order that doesn't honour the heuristic preference,
     * with no test failure to say so -- only a live GPU throughput collapse.
     */
    @Test
    void testBlackwoodHeuristicSortedOrderIsAPermutationSortedByDescendingHeuristicCount() throws Exception {
        int[] basePieces = new int[256];
        Arrays.fill(basePieces, PieceUtils.pack(1, 2, 3, 4)); // no heuristic colours (10, 13, 16) -> count 0
        basePieces[0] = PieceUtils.pack(10, 13, 16, 10); // all 4 edges heuristic -> count 4
        basePieces[1] = PieceUtils.pack(10, 2, 3, 4);    // 1 heuristic edge -> count 1
        PieceInventory inventory = new PieceInventory(basePieces);

        Method sortedOrderMethod = GpuEngine.class.getDeclaredMethod("blackwoodHeuristicSortedOrder", PieceInventory.class);
        sortedOrderMethod.setAccessible(true);
        int[] order = (int[]) sortedOrderMethod.invoke(null, inventory);

        assertEquals(1024, order.length);
        Set<Integer> seen = new HashSet<>();
        for (int idx : order) {
            assertTrue(idx >= 0 && idx < 1024, "Every entry must be a valid orientation index");
            seen.add(idx);
        }
        assertEquals(1024, seen.size(), "Must be a full permutation of 0..1023 with no duplicates");

        Method sideCountMethod = GpuEngine.class.getDeclaredMethod("blackwoodHeuristicSideCount", PieceInventory.class);
        sideCountMethod.setAccessible(true);
        int[] sideCount = (int[]) sideCountMethod.invoke(null, inventory);

        int prev = Integer.MAX_VALUE;
        for (int idx : order) {
            int count = sideCount[inventory.physicalMapping[idx]];
            assertTrue(count <= prev, "Order must be non-increasing by heuristic side count");
            prev = count;
        }
        assertEquals(4, sideCount[inventory.physicalMapping[order[0]]],
                "Highest-heuristic physical piece (count 4) must sort first");
    }

}
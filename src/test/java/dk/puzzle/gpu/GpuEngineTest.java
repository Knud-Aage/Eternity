package dk.puzzle.gpu;

import org.junit.jupiter.api.Test;
import sun.misc.Unsafe;

import java.lang.reflect.Field;
import java.util.Collections;

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

}
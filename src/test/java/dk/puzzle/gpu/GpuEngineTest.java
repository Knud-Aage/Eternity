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

    // ── exampleEdgeSlipBudget ──

    @Test
    void testExampleEdgeSlipBudgetIsZeroBeforeStep190() {
        int[] budget = GpuEngine.exampleEdgeSlipBudget();
        for (int step = 0; step < 190; step++) {
            assertEquals(0, budget[step], "Step " + step + " must permit no slips -- conflicts concentrate at index >= ~192");
        }
    }

    @Test
    void testExampleEdgeSlipBudgetIsMonotonicallyNonDecreasing() {
        int[] budget = GpuEngine.exampleEdgeSlipBudget();
        for (int step = 1; step < 256; step++) {
            assertTrue(budget[step] >= budget[step - 1],
                    "Slip budget must never decrease with depth (step " + step + ": " + budget[step]
                            + " < step " + (step - 1) + ": " + budget[step - 1] + ")");
        }
    }

    @Test
    void testExampleEdgeSlipBudgetNeverExceedsTheDocumentedCap() {
        // The formula's natural growth (+1 every 8 steps from 190) only reaches
        // 9 by step 255 -- the 25 cap is a safety ceiling, not a value the
        // curve is expected to actually hit within 256 steps.
        int[] budget = GpuEngine.exampleEdgeSlipBudget();
        for (int step = 0; step < 256; step++) {
            assertTrue(budget[step] <= 25, "Step " + step + " exceeds the documented cap of 25: " + budget[step]);
        }
    }

    @Test
    void testExampleEdgeSlipBudgetStartsAtOneImmediatelyAtStep190() {
        int[] budget = GpuEngine.exampleEdgeSlipBudget();
        assertEquals(1, budget[190], "The first slip-eligible step must permit exactly one slip");
    }

}
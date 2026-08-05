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

    // ------------------------------------------------------------------
    // Marathon-thread persistence state machine
    //
    // The kernel-side resume/checkpoint logic needs a real GPU and is covered
    // by GpuMarathonEquivalenceHarness (a standalone main(), deliberately named
    // *Harness so Surefire won't collect it). What IS unit-testable is the host-side
    // gate deciding whether a resume happens at all: persistResumeFlag is
    // (seedPersistenceEnabled && persistSlotValid), so if either half of that
    // state machine breaks, a launch either silently stops resuming (losing the
    // whole feature) or resumes into a slot that a teardown was supposed to have
    // invalidated (resuming a search whose startingStep floor no longer applies).
    // ------------------------------------------------------------------

    private static boolean persistSlotValid(GpuEngine engine) throws Exception {
        Field f = GpuEngine.class.getDeclaredField("persistSlotValid");
        f.setAccessible(true);
        return f.getBoolean(engine);
    }

    private static void setPersistSlotValid(GpuEngine engine, boolean value) throws Exception {
        Field f = GpuEngine.class.getDeclaredField("persistSlotValid");
        f.setAccessible(true);
        f.setBoolean(engine, value);
    }

    @Test
    void testSeedPersistenceIsDisabledByDefault() throws Exception {
        GpuEngine engine = newUninitializedEngine();

        assertFalse(engine.isSeedPersistenceEnabled(),
                "Persistence must default OFF so the live pipeline behaves exactly as before until deliberately enabled");
    }

    @Test
    void testDisablingPersistenceAlsoInvalidatesTheSlot() throws Exception {
        GpuEngine engine = newUninitializedEngine();
        engine.setSeedPersistenceEnabled(true);
        setPersistSlotValid(engine, true);

        engine.setSeedPersistenceEnabled(false);

        assertFalse(persistSlotValid(engine),
                "Turning persistence off must drop the checkpoint too -- otherwise re-enabling it later would "
                        + "resume a stale search captured under a possibly-different startingStep floor");
    }

    @Test
    void testInvalidatePersistedStateClearsTheSlotButLeavesTheToggleAlone() throws Exception {
        GpuEngine engine = newUninitializedEngine();
        engine.setSeedPersistenceEnabled(true);
        setPersistSlotValid(engine, true);

        engine.invalidatePersistedState();

        assertFalse(persistSlotValid(engine),
                "triggerBranchScrap's teardown must force the next launch to fresh-init");
        assertTrue(engine.isSeedPersistenceEnabled(),
                "A teardown invalidates the checkpoint, it does not switch the feature off");
    }

    @Test
    void testEnablingPersistenceDoesNotImmediatelyMarkASlotValid() throws Exception {
        GpuEngine engine = newUninitializedEngine();

        engine.setSeedPersistenceEnabled(true);

        assertFalse(persistSlotValid(engine),
                "Nothing has been checkpointed yet, so the first launch after enabling must fresh-init "
                        + "rather than resume whatever bytes happen to be in the slot buffers");
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
        // Read the REAL (post-BLACKWOOD_TO_THESIL-translation) heuristic colours rather than
        // hardcoding Blackwood's raw {13,16,10} -- those raw IDs are NOT the TheSil colours this
        // test's PieceInventory actually gets checked against (fixed 2026-08-02; translated values
        // are {5,20,7}, not {13,16,10} -- a test hardcoding the untranslated values would have
        // silently kept passing against the OLD, buggy, untranslated production code too).
        Field heuristicColorsField = GpuEngine.class.getDeclaredField("BLACKWOOD_HEURISTIC_COLORS");
        heuristicColorsField.setAccessible(true);
        int[] heuristicColors = (int[]) heuristicColorsField.get(null);
        int h0 = heuristicColors[0], h1 = heuristicColors[1], h2 = heuristicColors[2];
        int nonHeuristicColor = 1;
        while (nonHeuristicColor == h0 || nonHeuristicColor == h1 || nonHeuristicColor == h2) {
            nonHeuristicColor++;
        }

        int[] basePieces = new int[256];
        Arrays.fill(basePieces, PieceUtils.pack(nonHeuristicColor, nonHeuristicColor, nonHeuristicColor, nonHeuristicColor)); // no heuristic colours -> count 0
        basePieces[0] = PieceUtils.pack(h0, h1, h2, h0); // all 4 edges heuristic -> count 4
        basePieces[1] = PieceUtils.pack(h0, nonHeuristicColor, nonHeuristicColor, nonHeuristicColor); // 1 heuristic edge -> count 1
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

    /**
     * Direct regression test for the 2026-08-02 fix: BLACKWOOD_SIDE_COLORS and
     * BLACKWOOD_HEURISTIC_COLORS must be Blackwood's raw colours translated
     * through BLACKWOOD_TO_THESIL, not applied as-is. Before the fix these
     * fields held the untranslated {1,5,9,13,17}/{13,16,10} and were applied
     * directly against TheSil-numbered PieceInventory data -- silently
     * excluding/weighting the wrong colours with no test failure.
     */
    @Test
    void testBlackwoodColorConstantsAreTranslatedNotRaw() throws Exception {
        Field sideColorsField = GpuEngine.class.getDeclaredField("BLACKWOOD_SIDE_COLORS");
        sideColorsField.setAccessible(true);
        int[] sideColors = (int[]) sideColorsField.get(null);

        Field heuristicColorsField = GpuEngine.class.getDeclaredField("BLACKWOOD_HEURISTIC_COLORS");
        heuristicColorsField.setAccessible(true);
        int[] heuristicColors = (int[]) heuristicColorsField.get(null);

        // BLACKWOOD_TO_THESIL[{1,5,9,13,17}] = {1,3,4,5,2}, BLACKWOOD_TO_THESIL[{13,16,10}] = {5,20,7}
        // -- verified independently against dk.puzzle.blackwood.BwUtil.BLACKWOOD_TO_THESIL's own
        // documented derivation, not re-derived here.
        int[] expectedSideColors = {1, 3, 4, 5, 2};
        int[] expectedHeuristicColors = {5, 20, 7};

        assertArrayEquals(expectedSideColors, sideColors,
                "side colours must be translated to TheSil numbering, not left as Blackwood's raw {1,5,9,13,17}");
        assertArrayEquals(expectedHeuristicColors, heuristicColors,
                "heuristic colours must be translated to TheSil numbering, not left as Blackwood's raw {13,16,10}");
    }

    /**
     * Direct regression test for the second 2026-08-02 fix: the last branch's
     * divisor must stay double-precision (matching Blackwood's actual C#
     * source, which leaves 4.4615 as an unsuffixed double literal) rather
     * than the uniform-float approximation (4.4615f) this method previously
     * used. Reimplements the formula independently (not copy-pasted) so this
     * can't just repeat the same mistake twice, mirroring BwUtilTest's own
     * test for the CPU port's identical formula.
     */
    @Test
    void testBlackwoodHeuristicRequiredMatchesIndependentReimplementation() throws Exception {
        int[] expected = new int[256];
        for (int i = 0; i <= 160; i++) {
            if (i <= 16) expected[i] = 0;
            else if (i <= 26) expected[i] = (int) (((float) i - 16) * 2.8f);
            else if (i <= 56) expected[i] = (int) ((((float) i - 26) * 1.43333f) + 28);
            else if (i <= 76) expected[i] = (int) ((((float) i - 56) * 0.9f) + 71);
            else if (i <= 102) expected[i] = (int) ((((float) i - 76) * 0.6538f) + 89);
            else expected[i] = (int) ((((float) i - 102) / 4.4615) + 106);
        }

        Method method = GpuEngine.class.getDeclaredMethod("blackwoodHeuristicRequired");
        method.setAccessible(true);
        int[] actual = (int[]) method.invoke(null);

        assertArrayEquals(expected, actual);
    }

    /**
     * The kernel-side resume/checkpoint logic needs a real GPU and is covered by
     * GpuMarathonEquivalenceHarness. What is unit-testable here is the host-side
     * gate that decides whether a resume happens at all: persistResumeFlag is
     * (seedPersistenceEnabled AND persistSlotValid). If either half breaks, a
     * launch either silently stops resuming (losing the feature outright) or
     * resumes into a slot a teardown was supposed to have invalidated -- i.e.
     * continues a search whose startingStep floor no longer applies.
     */
    @Test
    void testSeedPersistenceAndInvalidationStateMachine() throws Exception {
        GpuEngine engine = newUninitializedEngine();
        assertFalse(engine.isSeedPersistenceEnabled(), "Seed persistence must default to OFF");

        Field persistSlotValidField = GpuEngine.class.getDeclaredField("persistSlotValid");
        persistSlotValidField.setAccessible(true);
        assertFalse((boolean) persistSlotValidField.get(engine), "persistSlotValid must default to false");

        engine.setSeedPersistenceEnabled(true);
        assertTrue(engine.isSeedPersistenceEnabled(), "Seed persistence should be enabled");

        // Manually simulate slot valid = true
        persistSlotValidField.set(engine, true);
        assertTrue((boolean) persistSlotValidField.get(engine), "persistSlotValid should be set to true");

        // Invalidate state
        engine.invalidatePersistedState();
        assertFalse((boolean) persistSlotValidField.get(engine), "invalidatePersistedState must clear persistSlotValid");

        // Set persistence = false clears valid slot
        persistSlotValidField.set(engine, true);
        engine.setSeedPersistenceEnabled(false);
        assertFalse(engine.isSeedPersistenceEnabled());
        assertFalse((boolean) persistSlotValidField.get(engine), "setSeedPersistenceEnabled(false) must clear persistSlotValid");
    }

}
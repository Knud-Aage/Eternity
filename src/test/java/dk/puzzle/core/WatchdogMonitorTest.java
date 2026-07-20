package dk.puzzle.core;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for WatchdogMonitor.
 * This class is currently unused/dead code -- similar logic lives inline in
 * EternitySolver's checkPoisonAndRetreat/triggerBranchScrap methods -- but it
 * is a distinct, deterministic, standalone unit worth testing on its own
 * terms. A real Logger is used since logging to console during a test run is
 * harmless; no need to mock it.
 */
class WatchdogMonitorTest {

    private static final Logger LOGGER = LogManager.getLogger("test");

    private WatchdogMonitor monitor;

    @BeforeEach
    void setUp() {
        monitor = new WatchdogMonitor(LOGGER);
    }

    // ---- checkPoison ----

    @Test
    void testCheckPoisonBelowDepthThresholdAlwaysFalseAndDoesNotAccumulateStrikes() {
        int hash = 123;
        assertFalse(monitor.checkPoison(hash, 0));
        assertFalse(monitor.checkPoison(hash, 100));
        assertFalse(monitor.checkPoison(hash, 199));

        // If those sub-200-depth calls had silently accumulated strikes, the
        // very next call at depth >= 200 would already be the 3rd strike.
        // It must instead be only the 1st real strike.
        assertFalse(monitor.checkPoison(hash, 200), "Calls below depth 200 must not count toward strikes");
        assertFalse(monitor.checkPoison(hash, 200), "Second real strike must still not poison");
        assertTrue(monitor.checkPoison(hash, 200), "Third real strike must poison the hash");
    }

    @Test
    void testCheckPoisonThreeStrikesTriggersPoisoning() {
        int hash = 555;
        assertFalse(monitor.checkPoison(hash, 200), "1st strike must not poison yet");
        assertFalse(monitor.checkPoison(hash, 250), "2nd strike must not poison yet");
        assertTrue(monitor.checkPoison(hash, 300), "3rd strike must poison the hash");
    }

    @Test
    void testCheckPoisonPoisonedHashAlwaysReturnsTrueOnSubsequentCalls() {
        int hash = 777;
        monitor.checkPoison(hash, 200);
        monitor.checkPoison(hash, 200);
        assertTrue(monitor.checkPoison(hash, 200), "3rd strike poisons");

        assertTrue(monitor.checkPoison(hash, 200), "Already-poisoned hash must keep returning true");
        assertTrue(monitor.checkPoison(hash, 999), "Already-poisoned hash must keep returning true regardless of depth (as long as >=200)");
    }

    @Test
    void testCheckPoisonDifferentHashesTrackedIndependently() {
        int hashA = 1;
        int hashB = 2;

        assertFalse(monitor.checkPoison(hashA, 200));
        assertFalse(monitor.checkPoison(hashA, 200));
        assertFalse(monitor.checkPoison(hashB, 200), "Unrelated hash must have its own independent strike count");

        assertTrue(monitor.checkPoison(hashA, 200), "hashA reaches its 3rd strike and must poison");
        assertFalse(monitor.checkPoison(hashB, 200), "hashB has only 2 strikes and must not be poisoned yet");
    }

    // ---- calculateTeardownDepth: GPU branch ----

    @Test
    void testCalculateTeardownDepthGpuLowStagnationDropFormula() {
        int result = monitor.calculateTeardownDepth(100, 10, true);

        assertEquals(100, monitor.absolutePeakDepth, "New peak depth must be recorded");
        assertEquals(0, monitor.trueStagnationCounter, "Stagnation counter resets when a new peak is reached");
        assertEquals(85, result, "dropAmount = 15 + (0 * 2) = 15 -> 100 - 15 = 85");
        assertEquals(1, monitor.consecutiveExtinctions);
    }

    @Test
    void testCalculateTeardownDepthGpuModerateStagnationUsesFixedDrop() {
        monitor.absolutePeakDepth = 200;
        monitor.trueStagnationCounter = 2;

        int result = monitor.calculateTeardownDepth(50, 1, true);

        assertEquals(3, monitor.trueStagnationCounter, "Counter increments because deepestStep did not beat the peak");
        assertEquals(10, result, "trueStagnationCounter (3) > 2 -> fixed dropAmount of 40 -> 50 - 40 = 10");
    }

    @Test
    void testCalculateTeardownDepthGpuHighStagnationResetsCounterAndClampsResultAtZero() {
        monitor.absolutePeakDepth = 1000;
        monitor.trueStagnationCounter = 4;

        int result = monitor.calculateTeardownDepth(10, 5, true);

        assertEquals(0, monitor.trueStagnationCounter, "Counter resets once the >4 branch fires");
        assertEquals(0, result, "dropAmount = (10 - 5) + 40 = 45, exceeds deepestStep so result clamps to 0");
    }

    // ---- calculateTeardownDepth: CPU branch ----

    @Test
    void testCalculateTeardownDepthCpuLowStagnationDropFormula() {
        int result = monitor.calculateTeardownDepth(10, 0, false);

        assertEquals(1, monitor.cpuStagnationCounter, "abs(10 - lastPeakDepth(0)) = 10 <= 15 -> increments");
        assertEquals(8, result, "cpuStagnationCounter (1) <= 5 -> dropAmount 2 -> 10 - 2 = 8");
    }

    @Test
    void testCalculateTeardownDepthCpuLastPeakResetsOnBigJump() {
        monitor.calculateTeardownDepth(100, 0, false); // lastPeakDepth becomes 100 (jump from 0 is > 15)
        int counterAfterFirstCall = monitor.cpuStagnationCounter;

        monitor.calculateTeardownDepth(105, 0, false); // abs(105-100)=5 <= 15 -> counter increments
        assertEquals(counterAfterFirstCall + 1, monitor.cpuStagnationCounter);

        monitor.calculateTeardownDepth(50, 0, false); // abs(50-100)=50 > 15 -> counter resets, lastPeakDepth becomes 50
        assertEquals(0, monitor.cpuStagnationCounter,
                "A jump greater than 15 from the last recorded peak must reset the CPU stagnation counter");
    }

    @Test
    void testCalculateTeardownDepthCpuStagnationOver5ThresholdIncreasesDrop() {
        monitor.cpuStagnationCounter = 5;

        int result = monitor.calculateTeardownDepth(10, 0, false); // counter becomes 6

        assertEquals(6, monitor.cpuStagnationCounter);
        assertEquals(6, result, "cpuStagnationCounter (6) > 5 -> dropAmount 4 -> 10 - 4 = 6");
    }

    @Test
    void testCalculateTeardownDepthCpuStagnationOver10ThresholdIncreasesDropFurther() {
        monitor.cpuStagnationCounter = 10;

        int result = monitor.calculateTeardownDepth(10, 0, false); // counter becomes 11

        assertEquals(11, monitor.cpuStagnationCounter);
        assertEquals(2, result, "cpuStagnationCounter (11) > 10 -> dropAmount 8 -> 10 - 8 = 2");
    }

    @Test
    void testCalculateTeardownDepthCpuStagnationOver20ThresholdResetsCounter() {
        monitor.cpuStagnationCounter = 20;

        int result = monitor.calculateTeardownDepth(10, 0, false); // counter becomes 21 -> triggers reset

        assertEquals(0, monitor.cpuStagnationCounter, "Counter resets to 0 once it exceeds 20");
        assertEquals(0, result, "cpuStagnationCounter (21) > 20 -> dropAmount 15 -> max(0, 10 - 15) = 0");
    }

    // ---- shared bookkeeping ----

    @Test
    void testCalculateTeardownDepthIncrementsConsecutiveExtinctionsEveryCall() {
        monitor.calculateTeardownDepth(10, 0, true);
        monitor.calculateTeardownDepth(20, 0, false);
        monitor.calculateTeardownDepth(30, 0, true);

        assertEquals(3, monitor.consecutiveExtinctions, "consecutiveExtinctions must increment on every call regardless of branch");
    }

    @Test
    void testCalculateTeardownDepthUpdatesAbsolutePeakDepthOnlyWhenExceeded() {
        monitor.absolutePeakDepth = 50;

        monitor.calculateTeardownDepth(30, 0, true); // does not beat the peak
        assertEquals(50, monitor.absolutePeakDepth);
        assertEquals(1, monitor.trueStagnationCounter);

        monitor.calculateTeardownDepth(80, 0, true); // beats the peak
        assertEquals(80, monitor.absolutePeakDepth);
        assertEquals(0, monitor.trueStagnationCounter, "Stagnation counter must reset once a new peak is reached");
    }
}

package dk.puzzle.core;

import org.apache.logging.log4j.Logger;

import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public class WatchdogMonitor {
    private final Logger logger;

    private final ConcurrentHashMap<Integer, Integer> hashStrikeCount = new ConcurrentHashMap<>();
    private final Set<Integer> poisonedHashes = ConcurrentHashMap.newKeySet();

    public int consecutiveExtinctions = 0;
    public int trueStagnationCounter = 0;
    public int cpuStagnationCounter = 0;
    public int absolutePeakDepth = 0;
    private int lastPeakDepth = 0;

    public WatchdogMonitor(Logger logger) {
        this.logger = logger;
    }

    public boolean checkPoison(int currentBoardHash, int currentDepth) {
        if (currentDepth < 200) return false;

        if (poisonedHashes.contains(currentBoardHash)) {
            logger.warn(">>> POISONED BOARD DETECTED! (" + currentBoardHash + "). Executing Nuclear Retreat!");
            return true;
        }

        int strikes = hashStrikeCount.getOrDefault(currentBoardHash, 0) + 1;
        hashStrikeCount.put(currentBoardHash, strikes);

        if (strikes >= 3) {
            logger.error(">>> GRAVITY WELL DETECTED! Poisoning hash: " + currentBoardHash);
            poisonedHashes.add(currentBoardHash);
            return true;
        }
        return false;
    }

    /**
     * Calculates how many pieces to drop based on stagnation history.
     * Returns the new depth target.
     */
    public int calculateTeardownDepth(int deepestStep, int baseCampPieces, boolean useGpu) {
        consecutiveExtinctions++;

        if (deepestStep > absolutePeakDepth) {
            absolutePeakDepth = deepestStep;
            trueStagnationCounter = 0;
        } else {
            trueStagnationCounter++;
        }

        if (!useGpu) {
            if (Math.abs(deepestStep - lastPeakDepth) <= 15) {
                cpuStagnationCounter++;
            } else {
                cpuStagnationCounter = 0;
                lastPeakDepth = deepestStep;
            }
        }

        int dropAmount;
        if (useGpu) {
            if (trueStagnationCounter > 4) {
                dropAmount = (deepestStep - baseCampPieces) + 40;
                trueStagnationCounter = 0;
            } else if (trueStagnationCounter > 2) {
                dropAmount = 40;
            } else {
                dropAmount = 15 + (trueStagnationCounter * 2);
            }
        } else {
            if (cpuStagnationCounter > 20) {
                dropAmount = 15; cpuStagnationCounter = 0;
            } else if (cpuStagnationCounter > 10) {
                dropAmount = 8;
            } else if (cpuStagnationCounter > 5) {
                dropAmount = 4;
            } else {
                dropAmount = 2;
            }
        }

        return Math.max(0, deepestStep - dropAmount);
    }
}

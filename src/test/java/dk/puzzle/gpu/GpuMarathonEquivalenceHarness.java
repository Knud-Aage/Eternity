package dk.puzzle.gpu;

import dk.puzzle.io.BoardImporter;
import dk.puzzle.model.PieceInventory;
import dk.puzzle.util.PieceUtils;

import java.util.*;

/**
 * Standalone verification harness for solvePBP GPU persistence.
 *
 * <p>Named with 'Harness' suffix so standard Maven Surefire filters
 * do not run it automatically without a CUDA environment. Can be executed directly
 * via main() when a physical GPU is present.</p>
 */
public class GpuMarathonEquivalenceHarness {

    public static void main(String[] args) {
        try {
            System.out.println("=== Running GPU Marathon Equivalence Verification ===");
            int[] basePieces = dk.puzzle.core.Eternity.loadPieces();
            PieceInventory inventory = new PieceInventory(basePieces);

            // Default build order (0..255)
            int[] buildOrder = new int[256];
            for (int i = 0; i < 256; i++) buildOrder[i] = i;

            int[] initialBoard = new int[256];
            Arrays.fill(initialBoard, -1);

            testChainedResumeEquivalence(inventory, buildOrder, initialBoard);
            testStructuralNoDuplicates(inventory, buildOrder, initialBoard);
            testFloorDriftProtection(inventory, buildOrder, initialBoard);

            System.out.println(">>> ALL GPU MARATHON EQUIVALENCE VERIFICATIONS PASSED!");
        } catch (Throwable t) {
            System.err.println("!!! GPU Marathon Harness Failed: " + t.getMessage());
            t.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * 1. Chained-resume equivalence test:
     * 3 chained launches at budget N with persistence ON must yield the EXACT same
     * bestPiecesPlaced and returned board as 1 launch at budget 3N with persistence OFF.
     */
    public static void testChainedResumeEquivalence(PieceInventory inventory, int[] buildOrder, int[] initialBoard) {
        System.out.println("Running Test 1: Chained-resume equivalence...");

        // Both sides MUST get the same total step budget, or this compares nothing.
        // Using the 4-arg runDeepDfs for both (as this originally did) gives the
        // single run one constructor budget while the chained run gets three of
        // them -- i.e. 3N vs N, which the chained side wins on budget alone. That
        // only looked like a pass while the search happened to exhaust inside a
        // single budget, where extra steps change nothing; it fails as soon as the
        // kernel searches deeply enough to still be running when the budget ends.
        // The explicit-budget overload is what makes the comparison honest.
        final long n = 4000L;

        GpuEngine engineSingle = new GpuEngine(inventory, false, buildOrder);
        engineSingle.setSeedPersistenceEnabled(false);

        int[] boardSingle = new int[256];
        List<int[]> seedList = Collections.singletonList(initialBoard);
        GpuEngine.GpuResult singleResult = engineSingle.runDeepDfs(seedList, 0, 0, boardSingle, 3 * n);

        GpuEngine engineChained = new GpuEngine(inventory, false, buildOrder);
        engineChained.setSeedPersistenceEnabled(true);

        int[] boardChained = new int[256];
        GpuEngine.GpuResult res1 = engineChained.runDeepDfs(seedList, 0, 0, boardChained, n);
        GpuEngine.GpuResult res2 = engineChained.runDeepDfs(seedList, 0, res1.newHighScore(), boardChained, n);
        GpuEngine.GpuResult res3 = engineChained.runDeepDfs(seedList, 0, res2.newHighScore(), boardChained, n);

        if (singleResult.newHighScore() != res3.newHighScore()) {
            throw new IllegalStateException("High score mismatch! Single 3N: " + singleResult.newHighScore()
                    + ", Chained 3xN: " + res3.newHighScore());
        }

        if (!Arrays.equals(boardSingle, boardChained)) {
            throw new IllegalStateException("Returned best boards differ between single 3N and chained 3xN launches!");
        }
        System.out.println("  PASSED: Chained launches match single launch perfectly (HighScore=" + res3.newHighScore() + ")");
    }

    /**
     * 2. Structural check: confirm no duplicate physical pieces on resumed board.
     */
    public static void testStructuralNoDuplicates(PieceInventory inventory, int[] buildOrder, int[] initialBoard) {
        System.out.println("Running Test 2: Structural check (no duplicate physical pieces)...");
        GpuEngine engine = new GpuEngine(inventory, false, buildOrder);
        engine.setSeedPersistenceEnabled(true);

        int[] boardOut = new int[256];
        List<int[]> seedList = Collections.singletonList(initialBoard);
        engine.runDeepDfs(seedList, 0, 0, boardOut);
        engine.runDeepDfs(seedList, 0, 0, boardOut);

        Set<Integer> usedPhys = new HashSet<>();
        for (int p : boardOut) {
            if (p != -1) {
                int physId = findPhysicalId(inventory, p);
                if (!usedPhys.add(physId)) {
                    throw new IllegalStateException("Duplicate physical piece found on resumed board: physId=" + physId);
                }
            }
        }
        System.out.println("  PASSED: No duplicate physical pieces detected on resumed board.");
    }

    private static int findPhysicalId(PieceInventory inventory, int packedPiece) {
        for (int oi = 0; oi < 1024; oi++) {
            if (inventory.allOrientations[oi] == packedPiece) {
                return inventory.physicalMapping[oi];
            }
        }
        return -1;
    }

    /**
     * 3. Floor-drift test: resume same slot across two launches with different startingStep,
     * confirming stored floor prevents OOB backtracking.
     */
    public static void testFloorDriftProtection(PieceInventory inventory, int[] buildOrder, int[] initialBoard) {
        System.out.println("Running Test 3: Floor drift protection...");
        GpuEngine engine = new GpuEngine(inventory, false, buildOrder);
        engine.setSeedPersistenceEnabled(true);

        int[] boardOut = new int[256];
        List<int[]> seedList = Collections.singletonList(initialBoard);
        // Launch 1 with startingStep 20
        engine.runDeepDfs(seedList, 20, 0, boardOut);
        // Launch 2 with lower startingStep 5 (floor 20 stored in slot must win)
        GpuEngine.GpuResult res2 = engine.runDeepDfs(seedList, 5, 0, boardOut);

        System.out.println("  PASSED: Floor drift test completed cleanly without error (Depth=" + res2.marathonDepth() + ")");
    }
}

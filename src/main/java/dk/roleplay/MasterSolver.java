package dk.roleplay;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Orchestrates the solving process for the Eternity II puzzle using a macro-tile approach.
 * This solver organizes the board into 4x4 sub-grids (macro-tiles) and attempts to
 * solve them in a spiral order starting from the centerpiece. It integrates with
 * {@link CandidateValidator} for parallel validation and supports persistence
 * via {@link CheckpointManager}.
 */
public class MasterSolver implements Runnable {
    private static volatile boolean solved = false;
    private final PieceInventory inventory;
    private final int[][] mainBoard = new int[16][];
    private final boolean[] usedPhysicalPieces = new boolean[256];
    private final CandidateValidator validator;
    private final AtomicInteger scoreRef;
    private final Random rnd;
    // Instead of 0, 1, 2, 3... we spiral outward from the centerpiece!
    // Macro 5 contains the center. Then we surround it: 6, 9, 10, etc.
    private final int[] solveOrder = {5, 6, 9, 10, 4, 8, 1, 2, 0, 3, 7, 11, 13, 14, 15, 12};

    // --- NEW CHECKPOINT & RECORD VARIABLES ---
    private int[][] resumeBoard;
    private int maxMacroReached = 0;
    private long lastSaveTime = 0;

    /**
     * Initializes a new instance of the MasterSolver with the necessary dependencies.
     *
     * @param inventory The inventory containing all puzzle pieces and their orientations.
     * @param validator The component responsible for validating macro-tile consistency.
     * @param scoreRef  A shared atomic integer used to track the current search progress.
     * @param seed      A seed for randomization to ensure diverse search paths.
     */
    public MasterSolver(PieceInventory inventory, CandidateValidator validator, AtomicInteger scoreRef, long seed) {
        this.inventory = inventory;
        this.validator = validator;
        this.scoreRef = scoreRef;
        this.rnd = new Random(seed);
        this.resumeBoard = CheckpointManager.load(); // Load state on startup

        if (this.resumeBoard != null) {
            int recoveredSteps = 0;
            for (int i = 0; i < 16; i++) {
                if (this.resumeBoard[i] != null) {
                    recoveredSteps++;
                }
            }
            if (recoveredSteps > 0) {
                // If we recovered 5 pieces, we successfully reached step 4!
                this.maxMacroReached = recoveredSteps - 1;
                System.out.println("Restored High Score Memory: " + recoveredSteps + " macro-tiles.");
            }
        }
    }

    /**
     * Execution entry point for the solver thread. It handles the high-level logic
     * of seeding the initial state and managing the main backtracking loop until
     * a solution is discovered or the search space is exhausted.
     */
    @Override
    public void run() {
        System.out.println("Solver thread started.");
        while (!solved) {
            reset();
            // FIX: Print the actual macro we are starting with!
            System.out.println("Seeding Macro " + solveOrder[0] + "...");
            seeding();

            int firstMacro = solveOrder[0];
            if (mainBoard[firstMacro] != null) {
                System.out.println("Macro " + firstMacro + " seeded successfully. Starting outward spiral...");
                // Pass "1" because we are moving to step 1 of the solveOrder array
                if (solve(1)) {
                    solved = true;
                    Main.updateDisplay(256, mainBoard);
                    System.out.println("SOLVED!");
                    break;
                } else {
                    System.out.println("Solve failed for this seed. Resetting...");
                    resumeBoard = null; // Clear resume board if branch fails completely
                }
            } else {
                // FIX: Tell the truth about which macro failed!
                System.out.println("Failed to seed Macro " + firstMacro + ". Resetting...");
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }
    }

    private void reset() {
        for (int i = 0; i < 16; i++) mainBoard[i] = null;
        Arrays.fill(usedPhysicalPieces, false);
    }

    private void seeding() {
        // Grab the very first macro-tile we are supposed to solve (Macro 5)
        int firstMacro = solveOrder[0];

        // --- RESUME LOGIC FOR THE FIRST MACRO ---
        if (resumeBoard != null && resumeBoard[firstMacro] != null) {
            mainBoard[firstMacro] = resumeBoard[firstMacro].clone();
            markUsed(mainBoard[firstMacro]);
            Main.updateDisplay(16, mainBoard);
            System.out.println("Resumed Macro " + firstMacro + " from checkpoint.");
            return;
        }

        int[] constraints = ConstraintBuilder.build(firstMacro, mainBoard);
        PermutationGenerator gen = new PermutationGenerator(inventory, usedPhysicalPieces);
        List<int[]> candidates = gen.generate(firstMacro, constraints, 500);

        if (!candidates.isEmpty()) {
            List<int[]> valid = validator.validate(flatten(candidates), candidates.size());
            if (!valid.isEmpty()) {
                int[] choice = valid.get(rnd.nextInt(valid.size()));
                mainBoard[firstMacro] = choice;
                markUsed(choice);
                Main.updateDisplay(16, mainBoard);
                System.out.println("Seeded Macro " + firstMacro + " with " + valid.size() + " valid candidates found.");
            }
        }
    }

    private boolean solve(int step) {
        if (step == 16) {
            return true; // All 16 macros placed!
        }
        if (solved) {
            return false;
        }

        // Fetch the actual Macro ID we should be working on right now
        int macroIdx = solveOrder[step];

        // --- FIXED RECORD LOGIC ---
        // Use 'step' to track our depth, because macroIdx jumps around!
        if (step > maxMacroReached) {
            maxMacroReached = step;
            // We pass (step + 1) because if we are on step 1, we actually have 2 macros placed on the board!
        }

        // ... THE REST OF YOUR SOLVE LOGIC STAYS EXACTLY THE SAME ...
        int[] constraints = ConstraintBuilder.build(macroIdx, mainBoard);
        PermutationGenerator gen = new PermutationGenerator(inventory, usedPhysicalPieces);

        List<int[]> candidates = gen.generate(macroIdx, constraints, 100);
        if (candidates.isEmpty()) {
            return false;
        }

        List<int[]> valid = validator.validate(flatten(candidates), candidates.size());
        if (valid.isEmpty()) {
            return false;
        }

        HeuristicSorter.sort(valid, inventory.colorFrequency);

        // --- FAST FORWARD RESUME LOGIC ---
        boolean isResuming = (resumeBoard != null && resumeBoard[macroIdx] != null);

        for (int[] tile : valid) {
            // If we are recovering a checkpoint, skip all pieces until we find the exact one we saved
            if (isResuming) {
                if (!Arrays.equals(tile, resumeBoard[macroIdx])) {
                    continue; // Skip!
                } else {
                    isResuming = false; // We caught up!
                    resumeBoard[macroIdx] = null; // Clear it so we don't skip if we backtrack here later
                }
            }

            mainBoard[macroIdx] = tile;
            markUsed(tile);

            // FIX 1: Pass the actual number of pieces placed to the display (using step)
            Main.updateDisplay((step + 1) * 16, mainBoard);

            // --- CHECKPOINT SAVE LOGIC (Every 60 Seconds) ---
            long now = System.currentTimeMillis();
            if (now - lastSaveTime > 60000) {
                CheckpointManager.save(mainBoard);
                lastSaveTime = now;
            }

            // FIX 2: We must move to the next chronological step in the array, not the next physical macro ID!
            if (solve(step + 1)) {
                return true;
            }

            unmarkUsed(tile);
            mainBoard[macroIdx] = null;

            // FIX 3: Update display during backtracking (using step)
            Main.updateDisplay(step * 16, mainBoard);

            if (solved) {
                return false;
            }
        }
        return false;
    }

    private int[] flatten(List<int[]> list) {
        int[] flat = new int[list.size() * 16];
        for (int i = 0; i < list.size(); i++) System.arraycopy(list.get(i), 0, flat, i * 16, 16);
        return flat;
    }

    private void markUsed(int[] tile) {
        for (int packed : tile) {
            boolean found = false;
            for (int orientationIdx = 0; orientationIdx < 1024 && !found; orientationIdx++) {
                if (inventory.allOrientations[orientationIdx] == packed) {
                    int physicalIdx = inventory.physicalMapping[orientationIdx];
                    if (!usedPhysicalPieces[physicalIdx]) {
                        usedPhysicalPieces[physicalIdx] = true;
                        found = true;
                    }
                }
            }
        }
    }

    private void unmarkUsed(int[] tile) {
        for (int packed : tile) {
            boolean found = false;
            for (int orientationIdx = 0; orientationIdx < 1024 && !found; orientationIdx++) {
                if (inventory.allOrientations[orientationIdx] == packed) {
                    int physicalIdx = inventory.physicalMapping[orientationIdx];
                    if (usedPhysicalPieces[physicalIdx]) {
                        usedPhysicalPieces[physicalIdx] = false;
                        found = true;
                    }
                }
            }
        }
    }
}
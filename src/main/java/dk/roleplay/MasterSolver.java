package dk.roleplay;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class MasterSolver implements Runnable {
    private final PieceInventory inventory;
    private final int[][] mainBoard = new int[16][];
    private final boolean[] usedPhysicalPieces = new boolean[256];
    private final CandidateValidator validator;
    private final AtomicInteger scoreRef;
    private final Random rnd;
    private static volatile boolean solved = false;
    // Instead of 0, 1, 2, 3... we spiral outward from the centerpiece!
    // Macro 5 contains the center. Then we surround it: 6, 9, 10, etc.
    private final int[] solveOrder = {5, 6, 9, 10, 4, 8, 1, 2, 0, 3, 7, 11, 13, 14, 15, 12};

    // --- NEW CHECKPOINT & RECORD VARIABLES ---
    private int[][] resumeBoard;
    private int maxMacroReached = 0;
    private long lastSaveTime = 0;

    public MasterSolver(PieceInventory inventory, CandidateValidator validator, AtomicInteger scoreRef, long seed) {
        this.inventory = inventory;
        this.validator = validator;
        this.scoreRef = scoreRef;
        this.rnd = new Random(seed);
        this.resumeBoard = CheckpointManager.load(); // Load state on startup
    }

    @Override
    public void run() {
        System.out.println("Solver thread started.");
        while (!solved) {
            reset();
            System.out.println("Seeding Macro 0...");
            seeding();

            if (mainBoard[0] != null) {
                System.out.println("Macro 0 successful. Starting solve...");
                if (solve(1)) {
                    solved = true;
                    Main.updateDisplay(256, mainBoard);
                    RecordManager.saveRecord(mainBoard, 16); // Final win save!
                    System.out.println("SOLVED!");
                    break;
                } else {
                    System.out.println("Solve failed for this seed. Resetting...");
                    resumeBoard = null; // Clear resume board if branch fails completely
                }
            } else {
                System.out.println("Failed to seed Macro 0. Resetting...");
                try { Thread.sleep(500); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            }
        }
    }

    private void reset() {
        for (int i = 0; i < 16; i++) mainBoard[i] = null;
        Arrays.fill(usedPhysicalPieces, false);
    }

    private void seeding() {
        // --- RESUME LOGIC FOR MACRO 0 ---
        if (resumeBoard != null && resumeBoard[0] != null) {
            mainBoard[0] = resumeBoard[0].clone();
            markUsed(mainBoard[0]);
            Main.updateDisplay(16, mainBoard);
            System.out.println("Resumed Macro 0 from checkpoint.");
            return;
        }

        int[] constraints = ConstraintBuilder.build(0, mainBoard);
        PermutationGenerator gen = new PermutationGenerator(inventory, usedPhysicalPieces);
        List<int[]> candidates = gen.generate(0, constraints, 500);

        if (!candidates.isEmpty()) {
            List<int[]> valid = validator.validate(flatten(candidates), candidates.size());
            if (!valid.isEmpty()) {
                int[] choice = valid.get(rnd.nextInt(valid.size()));
                mainBoard[0] = choice;
                markUsed(choice);
                Main.updateDisplay(16, mainBoard);
                System.out.println("Seeded Macro 0 with " + valid.size() + " valid candidates found.");
            }
        }
    }

    private boolean solve(int macroIdx) {
        if (macroIdx == 16) return true;
        if (solved) return false;

        // --- NEW RECORD LOGIC ---
        if (macroIdx > maxMacroReached) {
            maxMacroReached = macroIdx;
            RecordManager.saveRecord(mainBoard, macroIdx);
        }

        int[] constraints = ConstraintBuilder.build(macroIdx, mainBoard);
        PermutationGenerator gen = new PermutationGenerator(inventory, usedPhysicalPieces);

        List<int[]> candidates = gen.generate(macroIdx, constraints, 100);
        if (candidates.isEmpty()) return false;

        List<int[]> valid = validator.validate(flatten(candidates), candidates.size());
        if (valid.isEmpty()) return false;

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
            Main.updateDisplay((macroIdx + 1) * 16, mainBoard);

            // --- CHECKPOINT SAVE LOGIC (Every 60 Seconds) ---
            long now = System.currentTimeMillis();
            if (now - lastSaveTime > 60000) {
                CheckpointManager.save(mainBoard);
                lastSaveTime = now;
            }

            if (solve(macroIdx + 1)) return true;

            unmarkUsed(tile);
            mainBoard[macroIdx] = null;
            Main.updateDisplay(macroIdx * 16, mainBoard);
            if (solved) return false;
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
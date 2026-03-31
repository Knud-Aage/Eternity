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

    public MasterSolver(PieceInventory inventory, CandidateValidator validator, AtomicInteger scoreRef, long seed) {
        this.inventory = inventory;
        this.validator = validator;
        this.scoreRef = scoreRef;
        this.rnd = new Random(seed);
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
                    System.out.println("SOLVED!");
                    break;
                } else {
                    System.out.println("Solve failed for this seed. Resetting...");
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
            } else {
                System.out.println("Macro 0: candidates generated but none passed internal validation.");
            }
        }
    }

    private boolean solve(int macroIdx) {
        if (macroIdx == 16) return true;
        if (solved) return false;

        int[] constraints = ConstraintBuilder.build(macroIdx, mainBoard);
        PermutationGenerator gen = new PermutationGenerator(inventory, usedPhysicalPieces);

        List<int[]> candidates = gen.generate(macroIdx, constraints, 100);
        if (candidates.isEmpty()) return false;

        List<int[]> valid = validator.validate(flatten(candidates), candidates.size());
        if (valid.isEmpty()) return false;

        HeuristicSorter.sort(valid, inventory.colorFrequency);

        for (int[] tile : valid) {
            mainBoard[macroIdx] = tile;
            markUsed(tile);
            Main.updateDisplay((macroIdx + 1) * 16, mainBoard);

            try { Thread.sleep(20); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }

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

    /**
     * BUG FIX: Use physicalMapping for O(1) lookup instead of scanning all 256 pieces.
     * The packed piece value 'p' matches allOrientations[orientationIdx], and physicalMapping
     * gives us the physical piece index directly.
     */
    private void markUsed(int[] tile) {
        for (int packed : tile) {
            boolean found = false;
            // Search through all 1024 orientations to find this exact packed value
            for (int orientationIdx = 0; orientationIdx < 1024 && !found; orientationIdx++) {
                if (inventory.allOrientations[orientationIdx] == packed) {
                    int physicalIdx = inventory.physicalMapping[orientationIdx];
                    if (!usedPhysicalPieces[physicalIdx]) {
                        usedPhysicalPieces[physicalIdx] = true;
                        found = true;
                    }
                }
            }
            if (!found) {
                System.err.println("WARNING: markUsed could not find physical piece for packed=" + packed);
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
            if (!found) {
                System.err.println("WARNING: unmarkUsed could not find physical piece for packed=" + packed);
            }
        }
    }
}


/*
package dk.roleplay;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class MasterSolver implements Runnable {
    private final PieceInventory inventory;
    private final int[][] mainBoard = new int[16][16];
    private final boolean[] usedPhysicalPieces = new boolean[256];
    private final CandidateValidator validator;
    private final Random rnd;
    private static volatile boolean solved = false;

    public MasterSolver(PieceInventory inventory, CandidateValidator validator, AtomicInteger scoreRef, long seed) {
        this.inventory = inventory;
        this.validator = validator;
        this.rnd = new Random(seed);
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
                    System.out.println("SOLVED! Saving...");
                    SolutionExporter.save(mainBoard, "solution");
                    break;
                } else {
                    System.out.println("Solve failed for this seed. Resetting...");
                }
            } else {
                System.out.println("Failed to seed Macro 0. Resetting...");
                try { Thread.sleep(500); } catch (InterruptedException e) {}
            }
        }
    }

    private void reset() {
        for (int i = 0; i < 16; i++) mainBoard[i] = null;
        Arrays.fill(usedPhysicalPieces, false);
    }

    private void seeding() {
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
            }
        }
    }

    private boolean solve(int macroIdx) {
        if (macroIdx == 16) return true;
        if (solved) return false;

        int[] constraints = ConstraintBuilder.build(macroIdx, mainBoard);
        PermutationGenerator gen = new PermutationGenerator(inventory, usedPhysicalPieces);
        
        List<int[]> candidates = gen.generate(macroIdx, constraints, 100);
        if (candidates.isEmpty()) return false;

        List<int[]> valid = validator.validate(flatten(candidates), candidates.size());
        if (valid.isEmpty()) return false;

        HeuristicSorter.sort(valid, inventory.colorFrequency);

        for (int[] tile : valid) {
            mainBoard[macroIdx] = tile;
            markUsed(tile);
            Main.updateDisplay((macroIdx + 1) * 16, mainBoard);

            try { Thread.sleep(20); } catch (InterruptedException e) {}

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
        for (int p : tile) {
            for (int i = 0; i < 256; i++) {
                if (!usedPhysicalPieces[i]) {
                    boolean found = false;
                    for (int r = 0; r < 4; r++) {
                        if (inventory.allOrientations[i * 4 + r] == p) {
                            usedPhysicalPieces[i] = true;
                            found = true;
                            break;
                        }
                    }
                    if (found) break; 
                }
            }
        }
    }

    private void unmarkUsed(int[] tile) {
        for (int p : tile) {
            for (int i = 0; i < 256; i++) {
                if (usedPhysicalPieces[i]) {
                    boolean found = false;
                    for (int r = 0; r < 4; r++) {
                        if (inventory.allOrientations[i * 4 + r] == p) {
                            usedPhysicalPieces[i] = false;
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }
            }
        }
    }
}
*/
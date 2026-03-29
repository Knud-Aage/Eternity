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
            System.out.println("Attempting to seed Macro 0...");
            seeding();
            if (mainBoard[0] != null) {
                System.out.println("Macro 0 placed. Starting search...");
                if (solve(1)) {
                    solved = true;
                    Main.updateDisplay(256, mainBoard);
                    System.out.println("SOLVED! Saving solution...");
                    SolutionExporter.save(mainBoard, "solution");
                    break;
                } else {
                    System.out.println("Search failed for this seed. Retrying...");
                }
            } else {
                System.out.println("Failed to seed Macro 0. Retrying reset...");
                try { Thread.sleep(1000); } catch (InterruptedException e) {}
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
        
        if (candidates.isEmpty()) {
            System.out.println("No candidates found for Macro 0.");
            return;
        }

        List<int[]> valid = validator.validate(flatten(candidates), candidates.size());
        if (!valid.isEmpty()) {
            int[] choice = valid.get(rnd.nextInt(valid.size()));
            mainBoard[0] = choice;
            markUsed(choice);
            Main.updateDisplay(16, mainBoard);
        } else {
            System.out.println("Found " + candidates.size() + " candidates for Macro 0, but none were internally valid.");
        }
    }

    private boolean solve(int macroIdx) {
        if (macroIdx == 16) return true;
        if (solved) return false;

        int[] constraints = ConstraintBuilder.build(macroIdx, mainBoard);
        PermutationGenerator gen = new PermutationGenerator(inventory, usedPhysicalPieces);
        
        List<int[]> candidates = gen.generate(macroIdx, constraints, 200);
        if (candidates.isEmpty()) return false;

        List<int[]> valid = validator.validate(flatten(candidates), candidates.size());
        if (valid.isEmpty()) return false;

        HeuristicSorter.sort(valid, inventory.colorFrequency);

        for (int[] tile : valid) {
            mainBoard[macroIdx] = tile;
            markUsed(tile);
            Main.updateDisplay((macroIdx + 1) * 16, mainBoard);

            try { Thread.sleep(50); } catch (InterruptedException e) {}

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
                    if (found) break; // Move to next piece in tile
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
                    if (found) break; // Move to next piece in tile
                }
            }
        }
    }
}

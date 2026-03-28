package dk.roleplay;

import javax.swing.*;
import java.util.*;
import java.io.*;

public class EternityII {
    private static int[] inventory;
    private static int[][] mainBoard = new int[16][16];
    private static CandidateValidator validator;
    private static int[] colorFrequencies = new int[23];
    private static boolean[] usedPhysicalPieces = new boolean[256];
    private static boolean solved = false;
    private static long totalTries = 0;

    public static void main(String[] args) {
        System.out.println("Starting Eternity II Solver...");
        
        loadPieces("pieces.csv");
        calculateFrequencies();

        try {
            validator = new GpuValidator();
            System.out.println("Using GPU Validator.");
        } catch (Throwable e) {
            validator = new CpuValidator();
            System.out.println("GPU fail or not available. Using CPU Validator.");
        }

        SwingUtilities.invokeLater(() -> BoardVisualizer.createAndShowGUI(mainBoard));
        startCheckpointTimer();
        startStatusTimer();

        // Initial checkpoint load
        CheckpointManager.CheckpointData cp = CheckpointManager.load();
        if (cp != null) {
            inventory = cp.inventory;
            mainBoard = cp.mainBoard;
            reconstructUsedPieces();
            System.out.println("Resuming from checkpoint...");
        }

        // Loop indefinitely until a solution is found
        while (!solved) {
            if (cp == null) {
                shuffleInventory();
                resetBoard();
            }
            cp = null; 

            if (solve(0)) {
                solved = true;
                System.out.println("*********************************");
                System.out.println("   PUZZLE SOLVED SUCCESSFULLY!   ");
                System.out.println("*********************************");
                saveSolution("solution.txt");
                CheckpointManager.save(inventory, mainBoard);
                break;
            }
        }
    }

    private static void resetBoard() {
        for (int i = 0; i < 16; i++) mainBoard[i] = null;
        Arrays.fill(usedPhysicalPieces, false);
    }

    private static void reconstructUsedPieces() {
        Arrays.fill(usedPhysicalPieces, false);
        for (int[] tile : mainBoard) {
            if (tile == null) continue;
            for (int p : tile) {
                if (p == -1) continue;
                for (int i = 0; i < 256; i++) {
                    for (int r = 0; r < 4; r++) {
                        if (inventory[i*4 + r] == p) {
                            usedPhysicalPieces[i] = true;
                            break;
                        }
                    }
                    if (usedPhysicalPieces[i]) break;
                }
            }
        }
    }

    private static void loadPieces(String filename) {
        inventory = new int[1024];
        File file = new File(filename);
        if (!file.exists()) {
            System.out.println(filename + " not found, generating mock pieces...");
            generateMockPieces();
            return;
        }

        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            int i = 0;
            while ((line = br.readLine()) != null && i < 256) {
                String[] c = line.split(",");
                if (c.length < 4) continue;
                int n = Integer.parseInt(c[0].trim()), e = Integer.parseInt(c[1].trim()), s = Integer.parseInt(c[2].trim()), w = Integer.parseInt(c[3].trim());
                int p = PieceUtils.pack(n, e, s, w);
                for (int r = 0; r < 4; r++) {
                    inventory[i*4 + r] = p;
                    p = PieceUtils.rotate(p);
                }
                i++;
            }
        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
        }
    }

    private static void generateMockPieces() {
        Random rnd = new Random(42);
        for (int i = 0; i < 256; i++) {
            int n = rnd.nextInt(23), e = rnd.nextInt(23), s = rnd.nextInt(23), w = rnd.nextInt(23);
            int p = PieceUtils.pack(n, e, s, w);
            for (int r = 0; r < 4; r++) {
                inventory[i*4 + r] = p;
                p = PieceUtils.rotate(p);
            }
        }
    }

    private static void shuffleInventory() {
        List<int[]> physicalGroups = new ArrayList<>();
        for (int i = 0; i < 256; i++) {
            physicalGroups.add(new int[]{inventory[i*4], inventory[i*4+1], inventory[i*4+2], inventory[i*4+3]});
        }
        Collections.shuffle(physicalGroups);
        for (int i = 0; i < 256; i++) {
            System.arraycopy(physicalGroups.get(i), 0, inventory, i*4, 4);
        }
    }

    private static void calculateFrequencies() {
        Arrays.fill(colorFrequencies, 0);
        for (int p : inventory) {
            colorFrequencies[PieceUtils.getNorth(p)]++;
            colorFrequencies[PieceUtils.getEast(p)]++;
            colorFrequencies[PieceUtils.getSouth(p)]++;
            colorFrequencies[PieceUtils.getWest(p)]++;
        }
    }

    private static boolean solve(int macroIdx) {
        if (macroIdx == 16) return true;

        int[] constraints = buildConstraints(macroIdx);
        List<int[]> candidates = generateCandidates(constraints);
        
        if (candidates.isEmpty()) return false;

        int[] flattened = new int[candidates.size() * 16];
        for (int i = 0; i < candidates.size(); i++) System.arraycopy(candidates.get(i), 0, flattened, i * 16, 16);
        
        List<int[]> validTiles = validator.validate(flattened);
        validTiles.sort(Comparator.comparingDouble(EternityII::scoreTile));

        for (int[] tile : validTiles) {
            // Add a small delay so we can see the progress in the GUI
            try { Thread.sleep(10); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            totalTries++;
            if (tryPlaceMacroTile(tile)) {
                mainBoard[macroIdx] = tile;
                if (solve(macroIdx + 1)) return true;
                removeMacroTile(tile);
                mainBoard[macroIdx] = null;
            }
        }
        return false;
    }

    private static boolean tryPlaceMacroTile(int[] tile) {
        List<Integer> usedIndices = new ArrayList<>();
        for (int p : tile) {
            boolean found = false;
            for (int i = 0; i < 256; i++) {
                if (!usedPhysicalPieces[i]) {
                    for (int r = 0; r < 4; r++) {
                        if (inventory[i*4 + r] == p) {
                            usedPhysicalPieces[i] = true;
                            usedIndices.add(i);
                            found = true;
                            break;
                        }
                    }
                }
                if (found) break;
            }
            if (!found) {
                for (int idx : usedIndices) usedPhysicalPieces[idx] = false;
                return false;
            }
        }
        return true;
    }

    private static void removeMacroTile(int[] tile) {
        for (int p : tile) {
            for (int i = 0; i < 256; i++) {
                if (usedPhysicalPieces[i]) {
                    for (int r = 0; r < 4; r++) {
                        if (inventory[i*4 + r] == p) {
                            usedPhysicalPieces[i] = false;
                            break;
                        }
                    }
                }
            }
        }
    }

    private static double scoreTile(int[] tile) {
        double score = 0;
        for (int p : tile) {
            score += 1.0 / (colorFrequencies[PieceUtils.getNorth(p)] + 1);
            score += 1.0 / (colorFrequencies[PieceUtils.getEast(p)] + 1);
            score += 1.0 / (colorFrequencies[PieceUtils.getSouth(p)] + 1);
            score += 1.0 / (colorFrequencies[PieceUtils.getWest(p)] + 1);
        }
        return score;
    }

    private static int[] buildConstraints(int macroIdx) {
        int[] constraints = new int[16];
        int mRow = macroIdx / 4;
        int mCol = macroIdx % 4;

        for (int i = 0; i < 16; i++) {
            int r = i / 4;
            int c = i % 4;
            int n = PieceUtils.WILDCARD, e = PieceUtils.WILDCARD, s = PieceUtils.WILDCARD, w = PieceUtils.WILDCARD;

            if (mRow == 0 && r == 0) n = 0;
            if (mRow == 3 && r == 3) s = 0;
            if (mCol == 0 && c == 0) w = 0;
            if (mCol == 3 && c == 3) e = 0;

            if (r == 0 && mRow > 0 && mainBoard[macroIdx - 4] != null) {
                n = PieceUtils.getSouth(mainBoard[macroIdx - 4][12 + c]);
            }
            if (c == 0 && mCol > 0 && mainBoard[macroIdx - 1] != null) {
                w = PieceUtils.getEast(mainBoard[macroIdx - 1][r * 4 + 3]);
            }
            constraints[i] = PieceUtils.pack(n, e, s, w);
        }
        return constraints;
    }

    /**
     * Generates a batch of 16-piece macro-tiles that satisfy the outer boundary constraints.
     * This is a simplified recursive generator for demonstration.
     */
    private static List<int[]> generateCandidates(int[] constraints) {
        List<int[]> batch = new ArrayList<>();
        findMacroCombinations(constraints, new int[16], 0, batch);
        return batch;
    }

    private static void findMacroCombinations(int[] constraints, int[] currentTile, int pos, List<int[]> results) {
        if (results.size() >= 100) return; // Limit batch size for performance
        if (pos == 16) {
            results.add(currentTile.clone());
            return;
        }

        int target = constraints[pos];
        // Simple heuristic: search through inventory for pieces matching the boundary constraint
        for (int i = 0; i < 256; i++) {
            if (usedPhysicalPieces[i]) continue;
            for (int r = 0; r < 4; r++) {
                int p = inventory[i * 4 + r];
                if (matchesBoundary(p, target)) {
                    currentTile[pos] = p;
                    // Note: In a real solver, you'd check usedPhysicalPieces here too,
                    // but for candidate generation we often allow duplicates and filter later
                    findMacroCombinations(constraints, currentTile, pos + 1, results);
                }
            }
        }
    }

    private static boolean matchesBoundary(int piece, int constraint) {
        if (constraint == PieceUtils.WILDCARD) return true;
        int nC = PieceUtils.getNorth(constraint), eC = PieceUtils.getEast(constraint);
        int sC = PieceUtils.getSouth(constraint), wC = PieceUtils.getWest(constraint);

        if (nC != PieceUtils.WILDCARD && PieceUtils.getNorth(piece) != nC) return false;
        if (eC != PieceUtils.WILDCARD && PieceUtils.getEast(piece) != eC) return false;
        if (sC != PieceUtils.WILDCARD && PieceUtils.getSouth(piece) != sC) return false;
        if (wC != PieceUtils.WILDCARD && PieceUtils.getWest(piece) != wC) return false;
        return true;
    }

    private static void startCheckpointTimer() {
        new java.util.Timer().scheduleAtFixedRate(new TimerTask() {
            @Override public void run() { 
                if (!solved) CheckpointManager.save(inventory, mainBoard); 
            }
        }, 15*60*1000, 15*60*1000);
    }

    private static void startStatusTimer() {
        new java.util.Timer().scheduleAtFixedRate(new TimerTask() {
            @Override public void run() { 
                if (!solved) System.out.println("[Status] Total macro-tiles evaluated: " + totalTries);
            }
        }, 60000, 60000);
    }

    private static void saveSolution(String filename) {
        try (PrintWriter pw = new PrintWriter(new FileWriter(filename))) {
            pw.println("Eternity II Solution - 16x16 Grid");
            pw.println("Format: [Row, Col] -> North, East, South, West");
            for (int m = 0; m < 16; m++) {
                int[] tile = mainBoard[m];
                int mRow = m / 4;
                int mCol = m % 4;
                for (int p = 0; p < 16; p++) {
                    int pRow = p / 4;
                    int pCol = p % 4;
                    int globalRow = mRow * 4 + pRow;
                    int globalCol = mCol * 4 + pCol;
                    int piece = tile[p];
                    pw.printf("[%d, %d] -> %d, %d, %d, %d\n", 
                        globalRow, globalCol, 
                        PieceUtils.getNorth(piece), PieceUtils.getEast(piece), 
                        PieceUtils.getSouth(piece), PieceUtils.getWest(piece));
                }
            }
            System.out.println("Solution saved to " + filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

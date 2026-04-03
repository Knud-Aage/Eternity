package dk.roleplay;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class MasterSolverPBP implements Runnable {
    private final PieceInventory inventory;

    // We replace the 16x16 macro system with a flat 256-piece array!
    private final int[] flatBoard = new int[256];
    private final boolean[] usedPhysicalPieces = new boolean[256];
    private final Random rnd = new Random();
    private long lastSaveTime = System.currentTimeMillis();
    private int centerPhysicalIdx = -1;
    private int deepestPos = 0;

    public MasterSolverPBP(PieceInventory inventory) {
        this.inventory = inventory;

        // Initialize the board with -1 (empty)
        Arrays.fill(flatBoard, -1);

        // Find the physical ID of the true Centerpiece (N=18, E=12, S=18, W=3)
        int targetPiece = PieceUtils.pack(18, 12, 18, 3);
        for (int i = 0; i < 1024; i++) {
            if (inventory.allOrientations[i] == targetPiece) {
                this.centerPhysicalIdx = inventory.physicalMapping[i];
                break;
            }
        }
    }

    @Override
    public void run() {
        System.out.println("Starting Piece-By-Piece (PBP) Solver...");

        // MAGIC TRICK: Pre-place the centerpiece at Global Row 7, Col 7 (Index 119)
        // By locking it in now, the solver treats it as an unmovable obstacle and builds right into it!
        flatBoard[119] = PieceUtils.pack(18, 12, 18, 3);
        usedPhysicalPieces[centerPhysicalIdx] = true;

        // Start the solver at the top-left corner (position 0)
        if (solve(0)) {
            System.out.println("SOLVED!");
        } else {
            System.out.println("Exhausted search space. No solution found.");
        }
    }

    private boolean solve(int pos) {
        // If we reach the pre-placed centerpiece, just skip over it!
        if (pos == 119) {
            return solve(pos + 1);
        }

        // If we reach 256, the entire board is filled perfectly.
        if (pos == 256) {
            return true;
        }

        // Track high scores and update the visualizer
        if (pos > deepestPos) {
            deepestPos = pos;
            int[][] legacyBoard = buildLegacyBoard();

            updateDisplay(legacyBoard);
            RecordManager.saveRecord(legacyBoard, deepestPos); // Save the image!
        }

        // --- CHECKPOINT SAVE LOGIC (Every 60 Seconds) ---
        long now = System.currentTimeMillis();
        if (now - lastSaveTime > 60000) {
            int[][] legacyBoard = buildLegacyBoard();
            CheckpointManager.save(legacyBoard);
            lastSaveTime = now;
            System.out.println(">>> PBP Checkpoint Saved at piece " + pos);
        }
        int row = pos / 16;
        int col = pos % 16;

        // --- DYNAMIC OMNIDIRECTIONAL CONSTRAINTS ---
        // Look at the flat array. If there is a piece next to us, read its color.
        // If it's empty (-1), it's a wildcard! If it's a board edge, it's Gray (0).
        int n_req = (row == 0) ? 0 : (flatBoard[pos - 16] != -1 ? PieceUtils.getSouth(flatBoard[pos - 16]) :
                                      PieceUtils.WILDCARD);
        int s_req = (row == 15) ? 0 : (flatBoard[pos + 16] != -1 ? PieceUtils.getNorth(flatBoard[pos + 16]) :
                                       PieceUtils.WILDCARD);
        int w_req = (col == 0) ? 0 : (flatBoard[pos - 1] != -1 ? PieceUtils.getEast(flatBoard[pos - 1]) :
                                      PieceUtils.WILDCARD);
        int e_req = (col == 15) ? 0 : (flatBoard[pos + 1] != -1 ? PieceUtils.getWest(flatBoard[pos + 1]) :
                                       PieceUtils.WILDCARD);

        // Determine which pool to pull from based on board edges
        int b_req = 0;
        if (row == 0 || row == 15) {
            b_req++;
        }
        if (col == 0 || col == 15) {
            b_req++;
        }

        List<Integer> pool;
        if (b_req == 2) {
            pool = inventory.corners;
        } else if (b_req == 1) {
            pool = inventory.edges;
        } else {
            pool = inventory.interior;
        }

        if (pool.isEmpty()) {
            return false;
        }

        // Add a random offset so the solver doesn't always start with the exact same piece sequence
        int size = pool.size();
        int offset = rnd.nextInt(size);

        for (int i = 0; i < size; i++) {
            int orientationIdx = pool.get((i + offset) % size);
            int p = inventory.allOrientations[orientationIdx];
            int physicalIdx = inventory.physicalMapping[orientationIdx];

            // Is the piece already on the board?
            if (usedPhysicalPieces[physicalIdx]) {
                continue;
            }

            // Does it match the dynamic constraints?
            if (matches(p, n_req, e_req, s_req, w_req)) {

                // IT FITS! Lock it in.
                flatBoard[pos] = p;
                usedPhysicalPieces[physicalIdx] = true;

                // Move to the next spot on the board (pos + 1)
                if (solve(pos + 1)) {
                    return true;
                }

                // BACKTRACK! It was a dead end. Remove it and try the next piece.
                flatBoard[pos] = -1;
                usedPhysicalPieces[physicalIdx] = false;
            }
        }
        return false; // No pieces fit. Dead end.
    }

    private boolean matches(int p, int n, int e, int s, int w) {
        if (n != PieceUtils.WILDCARD && PieceUtils.getNorth(p) != n) {
            return false;
        }
        if (e != PieceUtils.WILDCARD && PieceUtils.getEast(p) != e) {
            return false;
        }
        if (s != PieceUtils.WILDCARD && PieceUtils.getSouth(p) != s) {
            return false;
        }
        return w == PieceUtils.WILDCARD || PieceUtils.getWest(p) == w;
    }

    // --- 1. DYNAMIC ARRAY TRANSLATOR ---
    // Converts the 1D flat array back into the 2D Macro-Tile format
    private int[][] buildLegacyBoard() {
        int[][] legacyBoard = new int[16][];

        for (int i = 0; i < 256; i++) {
            if (flatBoard[i] == -1) {
                continue;
            }

            int gRow = i / 16;
            int gCol = i % 16;

            int mIdx = (gRow / 4) * 4 + (gCol / 4);
            int pIdx = (gRow % 4) * 4 + (gCol % 4);

            if (legacyBoard[mIdx] == null) {
                legacyBoard[mIdx] = new int[16];
                // Fill empty slots with safe Wildcards so the managers don't crash
                Arrays.fill(legacyBoard[mIdx], PieceUtils.pack(PieceUtils.WILDCARD, PieceUtils.WILDCARD,
                        PieceUtils.WILDCARD, PieceUtils.WILDCARD));
            }
            legacyBoard[mIdx][pIdx] = flatBoard[i];
        }
        return legacyBoard;
    }

    // --- 2. UPDATE DISPLAY UTILITY ---
    private void updateDisplay(int[][] legacyBoard) {
        Main.updateDisplay(deepestPos, legacyBoard);
    }
}
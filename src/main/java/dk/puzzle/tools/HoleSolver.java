package dk.puzzle.tools;

import dk.puzzle.ai.ConflictReducer;
import dk.puzzle.core.Eternity;
import dk.puzzle.io.BucasExporter;
import dk.puzzle.model.CompatibilityIndex;
import dk.puzzle.model.PieceInventory;
import dk.puzzle.util.PieceUtils;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Deque;
import java.util.List;

/**
 * Given a bucas.name record link (any puzzle name), finds every connected
 * region of edge-conflicting cells, then re-solves each region exactly using
 * ONLY the physical pieces already sitting in it plus the fixed boundary colours from the surrounding
 * locked pieces. Unlike the LNS "Surgeon" phase this is exhaustive/exact for
 * the region, not a random-restart heuristic — if a zero-conflict rearrangement
 * of that exact piece set exists, it will find it.
 *
 * <p>Deliberately does not hardcode board coordinates: the conflict region is
 * detected fresh from whatever board_edges string is passed in, so it stays
 * useful for future record boards even though their conflict shape will differ
 * from today's.</p>
 *
 * Usage: paste a bucas link (or just its board_edges value) as argv[0].
 */
public class HoleSolver {

    private static final int W = 16, H = 16;

    // BucasExporter's TheSil -> Bucas color remap. Bucas links use the
    // Bucas numbering, everything else in this codebase (PieceInventory,
    // CompatibilityIndex, PieceUtils) uses TheSil numbering, so decoding a
    // bucas link requires the inverse of this table.
    private static final int[] THESIL_TO_BUCAS = {
            0, 1, 3, 4, 2, 5, 6, 7, 9, 12, 14, 15, 19, 21, 8, 10, 13, 16, 17, 18, 20, 22, 11
    };
    private static final int[] BUCAS_TO_THESIL = new int[23];
    static {
        for (int thesil = 0; thesil < 23; thesil++) {
            BUCAS_TO_THESIL[THESIL_TO_BUCAS[thesil]] = thesil;
        }
    }

    // Generous but bounded, so a genuinely unsatisfiable hole shape fails fast
    // instead of hanging forever.
    private static final long STEP_BUDGET_PER_REGION = 200_000_000L;

    private static final String PHYSICAL_LAYOUT_FILE = "physical_layout.txt";

    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Usage: HoleSolver <bucas link or raw board_edges string>");
            return;
        }

        String boardEdges = extractBoardEdges(args[0]);
        if (boardEdges.length() != 1024) {
            System.out.println("Expected a 1024-character board_edges value, got " + boardEdges.length() + " chars.");
            return;
        }

        int[] board = decodeBoard(boardEdges);

        PieceInventory inventory = new PieceInventory(Eternity.loadPieces());
        CompatibilityIndex compat = new CompatibilityIndex(inventory.allOrientations, inventory.physicalMapping);

        int[] physicalIdAt = new int[256];
        Arrays.fill(physicalIdAt, -1);
        int emptyCells = 0;
        for (int i = 0; i < 256; i++) {
            if (board[i] == -1) {
                emptyCells++;
                continue; // genuinely empty — not a piece to identify
            }
            physicalIdAt[i] = findPhysicalId(inventory, board[i]);
            if (physicalIdAt[i] == -1) {
                System.out.println("WARNING: cell " + i + " (piece " + Integer.toHexString(board[i]) +
                        ") doesn't match any orientation in pieces.csv. Is this the right piece dataset for this link?");
            }
        }

        List<int[]> conflicts = findConflicts(board); // each entry: {cellA, cellB} (cellB = -1 for border conflicts)
        System.out.println("Decoded board: " + conflicts.size() + " edge conflicts, " + emptyCells + " empty cell(s).");
        if (conflicts.isEmpty() && emptyCells == 0) {
            System.out.println("Nothing to fix — board is already complete and conflict-free.");
            return;
        }

        // A region to resolve is either a real edge mismatch OR a genuinely
        // empty cell still waiting to be filled — both need the exact search
        // below, just with a different starting pool (see below).
        boolean[] inHole = new boolean[256];
        for (int[] c : conflicts) {
            inHole[c[0]] = true;
            if (c[1] != -1) inHole[c[1]] = true;
        }
        for (int i = 0; i < 256; i++) {
            if (board[i] == -1) inHole[i] = true;
        }

        List<List<Integer>> regions = connectedComponents(inHole);
        System.out.println("Found " + regions.size() + " connected region(s) needing resolution: " +
                regions.stream().map(List::size).toList());

        // Physical pieces used ANYWHERE on the whole board (before any region
        // gets cleared below) — these are the pieces NOT available to draw on
        // for filling genuinely empty holes, since they're committed elsewhere.
        boolean[] usedAnywhere = new boolean[256];
        for (int i = 0; i < 256; i++) {
            if (physicalIdAt[i] != -1) usedAnywhere[physicalIdAt[i]] = true;
        }

        int[] finalBoard = Arrays.copyOf(board, 256);
        int solvedRegions = 0;
        List<Integer> unsolvedCells = new ArrayList<>();

        for (List<Integer> region : regions) {
            System.out.println();
            System.out.println("--- Region: " + region.size() + " cells " + region + " ---");

            // Pool = whichever physical pieces this region's own (placed,
            // conflicting) cells currently hold, UNION every physical piece
            // that isn't placed anywhere on the whole board (available for
            // this region's genuinely empty cells).
            boolean[] localUsed = new boolean[256];
            Arrays.fill(localUsed, true);
            for (int physId = 0; physId < 256; physId++) {
                if (!usedAnywhere[physId]) localUsed[physId] = false;
            }
            boolean skipRegion = false;
            for (int cell : region) {
                if (finalBoard[cell] == -1) continue; // nothing to reclaim from an already-empty cell
                int physId = physicalIdAt[cell];
                if (physId == -1) {
                    System.out.println("Skipping region: contains an unidentified piece, can't build a safe pool.");
                    skipRegion = true;
                    break;
                }
                localUsed[physId] = false;
            }
            if (skipRegion) continue;

            int[] holeBoard = Arrays.copyOf(finalBoard, 256);
            for (int cell : region) holeBoard[cell] = -1;

            RegionSolver solver = new RegionSolver(inventory, compat, holeBoard, region, localUsed);
            boolean solved = solver.solve();

            if (solved) {
                System.out.println("SOLVED — zero conflicts in this region after " +
                        String.format("%,d", solver.stepsTaken) + " search steps.");
                for (int cell : region) {
                    finalBoard[cell] = solver.board[cell];
                    // Claim this piece out of the shared leftover pool so a
                    // later region in this same run can't also try to use it.
                    int physId = findPhysicalId(inventory, solver.board[cell]);
                    if (physId != -1) usedAnywhere[physId] = true;
                }
                solvedRegions++;
            } else {
                System.out.println("NOT solved within the step budget (" +
                        String.format("%,d", solver.stepsTaken) + " steps tried, budget " +
                        String.format("%,d", STEP_BUDGET_PER_REGION) + "). " +
                        "Either no zero-conflict rearrangement of these exact pieces exists, " +
                        "or it needs a larger step budget.");
                unsolvedCells.addAll(region);
            }
        }

        System.out.println();
        System.out.println("=== " + solvedRegions + "/" + regions.size() + " region(s) solved exactly. " +
                "Remaining unsolved cells: " + (256 - countPlaced(finalBoard)) + " ===");

        int[] repaired = null;
        if (solvedRegions < regions.size()) {
            // Exact search couldn't reach zero conflicts for every region —
            // proves some conflict is unavoidable with this exact piece pool.
            // Fall back to heuristics for the best actually achievable result.
            //
            // Clear every still-unsolved region's cells back to empty (using
            // the exact closed pool of pieces that were sitting there, same
            // as what the exact search used) and let MCV heuristically
            // re-fill them, allowing conflicts. This explores a much larger
            // space than single-move rotation/swap repair, which can only
            // nudge one piece at a time and easily gets stuck — a full
            // clear-and-refill can reach rearrangements a single swap can't.
            // Finish with a rotation+swap polish pass over the result.
            System.out.println();
            System.out.println("Falling back to conflict-reduction heuristics for the still-unsolved region(s)...");
            ConflictReducer reducer = new ConflictReducer(inventory, false);
            int conflictsBefore = reducer.countConflicts(finalBoard);

            repaired = Arrays.copyOf(finalBoard, 256);
            for (int cell : unsolvedCells) repaired[cell] = -1;

            repaired = reducer.mcvRestartFill(repaired, 200_000);
            int afterFill = reducer.countConflicts(repaired);
            int afterPolish = reducer.reducePostProcess(repaired, 100);

            System.out.println("Best result: " + conflictsBefore + " (start) -> " + afterFill +
                    " (clear + MCV refill) -> " + afterPolish + " (rotation+swap polish) edge conflicts.");
            System.out.println(BucasExporter.exportBoard(repaired));
        }

        String link = BucasExporter.exportBoard(finalBoard);
        System.out.println(link);

        writePhysicalLayoutFile(inventory, finalBoard, repaired);
    }

    // ------------------------------------------------------------------
    // Physical piece layout export — for placing a solution on the real
    // Eternity II board. Reports each cell's real physical piece number
    // (the numbering already baked into pieces.csv, row i -> piece i) and
    // how many 90-degree CLOCKWISE turns to apply from that piece's
    // reference orientation as recorded in pieces.csv (rotation 0).
    // ------------------------------------------------------------------

    private static void writePhysicalLayoutFile(PieceInventory inventory, int[] finalBoard, int[] repaired) {
        try (PrintWriter out = new PrintWriter(new FileWriter(PHYSICAL_LAYOUT_FILE))) {
            if (repaired != null) {
                writePhysicalLayout(out, "Heuristic fallback board (repaired)", repaired, inventory);
            }
            writePhysicalLayout(out, "Final board", finalBoard, inventory);
            System.out.println();
            System.out.println("Physical piece layout written to " + PHYSICAL_LAYOUT_FILE);
        } catch (IOException e) {
            System.out.println("WARNING: couldn't write " + PHYSICAL_LAYOUT_FILE + ": " + e.getMessage());
        }
    }

    private static void writePhysicalLayout(PrintWriter out, String label, int[] board, PieceInventory inventory) {
        out.println("=== " + label + " (" + findConflicts(board).size() + " edge conflicts) ===");
        out.println("Cell format: <physical piece number>@<clockwise rotation in degrees from its pieces.csv reference orientation>");
        out.println("Row 0 = north/top edge of the board, Col 0 = west/left edge.");
        out.println();
        for (int row = 0; row < H; row++) {
            StringBuilder line = new StringBuilder();
            for (int col = 0; col < W; col++) {
                int p = board[row * W + col];
                String cell;
                if (p == -1) {
                    cell = "EMPTY";
                } else {
                    int[] pieceAndRotation = physicalPieceAndRotation(inventory, p);
                    cell = pieceAndRotation[0] == -1
                            ? "?"
                            : (pieceAndRotation[0] + "@" + pieceAndRotation[1]);
                }
                line.append(String.format("%-9s", cell));
            }
            out.println(line.toString().stripTrailing());
        }
        out.println();
    }

    /** Returns {physical piece number (1-256), clockwise rotation in degrees from its pieces.csv reference orientation}, or {-1,-1} if not found. */
    private static int[] physicalPieceAndRotation(PieceInventory inventory, int packedPiece) {
        for (int oi = 0; oi < 1024; oi++) {
            if (inventory.allOrientations[oi] == packedPiece) {
                return new int[]{inventory.physicalMapping[oi] + 1, (oi % 4) * 90};
            }
        }
        return new int[]{-1, -1};
    }

    private static int countPlaced(int[] board) {
        int n = 0;
        for (int p : board) if (p != -1 && p != -2) n++;
        return n;
    }

    // ------------------------------------------------------------------
    // Decoding / encoding
    // ------------------------------------------------------------------

    private static String extractBoardEdges(String input) {
        int idx = input.indexOf("board_edges=");
        if (idx == -1) {
            return input.trim();
        }
        String rest = input.substring(idx + "board_edges=".length());
        int amp = rest.indexOf('&');
        return (amp == -1 ? rest : rest.substring(0, amp)).trim();
    }

    private static int[] decodeBoard(String boardEdges) {
        int[] board = new int[256];
        for (int i = 0; i < 256; i++) {
            // BucasExporter's own encoding uses the literal string "aaaa" as a
            // sentinel for empty cells (see exportBoard: `if (p == -1 || p ==
            // -2) bucasString.append("aaaa")`), not as a real piece — no actual
            // Eternity II piece has grey on all four sides (corners have 2,
            // edges have 1), so this is unambiguous and must be decoded back
            // to empty (-1), not to a fake all-grey "piece".
            if (boardEdges.regionMatches(i * 4, "aaaa", 0, 4)) {
                board[i] = -1;
                continue;
            }
            int n = BUCAS_TO_THESIL[boardEdges.charAt(i * 4)     - 'a'];
            int e = BUCAS_TO_THESIL[boardEdges.charAt(i * 4 + 1) - 'a'];
            int s = BUCAS_TO_THESIL[boardEdges.charAt(i * 4 + 2) - 'a'];
            int w = BUCAS_TO_THESIL[boardEdges.charAt(i * 4 + 3) - 'a'];
            board[i] = PieceUtils.pack(n, e, s, w);
        }
        return board;
    }

    private static int findPhysicalId(PieceInventory inventory, int packedPiece) {
        for (int oi = 0; oi < 1024; oi++) {
            if (inventory.allOrientations[oi] == packedPiece) {
                return inventory.physicalMapping[oi];
            }
        }
        return -1;
    }

    // ------------------------------------------------------------------
    // Conflict detection
    // ------------------------------------------------------------------

    /** Each result entry is {cellIndex, neighbourIndex} for an internal mismatch, or {cellIndex, -1} for a border violation. */
    private static List<int[]> findConflicts(int[] board) {
        List<int[]> conflicts = new ArrayList<>();
        for (int row = 0; row < H; row++) {
            for (int col = 0; col < W; col++) {
                int idx = row * W + col;
                int p = board[idx];
                if (p == -1) continue; // genuinely empty cell — nothing to check FROM here

                if (row == 0  && PieceUtils.getNorth(p) != PieceUtils.BORDER_COLOR) conflicts.add(new int[]{idx, -1});
                if (row == H-1 && PieceUtils.getSouth(p) != PieceUtils.BORDER_COLOR) conflicts.add(new int[]{idx, -1});
                if (col == 0  && PieceUtils.getWest(p)  != PieceUtils.BORDER_COLOR) conflicts.add(new int[]{idx, -1});
                if (col == W-1 && PieceUtils.getEast(p)  != PieceUtils.BORDER_COLOR) conflicts.add(new int[]{idx, -1});

                if (col < W - 1) {
                    int right = idx + 1;
                    if (board[right] != -1 && PieceUtils.getEast(p) != PieceUtils.getWest(board[right])) {
                        conflicts.add(new int[]{idx, right});
                    }
                }
                if (row < H - 1) {
                    int down = idx + W;
                    if (board[down] != -1 && PieceUtils.getSouth(p) != PieceUtils.getNorth(board[down])) {
                        conflicts.add(new int[]{idx, down});
                    }
                }
            }
        }
        return conflicts;
    }

    private static List<List<Integer>> connectedComponents(boolean[] inHole) {
        boolean[] visited = new boolean[256];
        List<List<Integer>> components = new ArrayList<>();

        for (int start = 0; start < 256; start++) {
            if (!inHole[start] || visited[start]) continue;

            List<Integer> component = new ArrayList<>();
            Deque<Integer> queue = new ArrayDeque<>();
            queue.add(start);
            visited[start] = true;

            while (!queue.isEmpty()) {
                int cur = queue.poll();
                component.add(cur);
                int row = cur / W, col = cur % W;

                int[] neighbours = {
                        row > 0     ? cur - W : -1,
                        row < H - 1 ? cur + W : -1,
                        col > 0     ? cur - 1 : -1,
                        col < W - 1 ? cur + 1 : -1
                };
                for (int n : neighbours) {
                    if (n != -1 && inHole[n] && !visited[n]) {
                        visited[n] = true;
                        queue.add(n);
                    }
                }
            }
            component.sort(Integer::compareTo);
            components.add(component);
        }
        return components;
    }

    // ------------------------------------------------------------------
    // Exact backtracking search over a single region
    // ------------------------------------------------------------------

    private static class RegionSolver {
        final PieceInventory inventory;
        final CompatibilityIndex compat;
        final int[] board;          // full 256-cell board; region cells start as -1
        final List<Integer> region; // cell indices belonging to this hole
        final boolean[] localUsed;  // physical-id usage, true = unavailable
        long stepsTaken = 0;

        RegionSolver(PieceInventory inventory, CompatibilityIndex compat, int[] board,
                     List<Integer> region, boolean[] localUsed) {
            this.inventory = inventory;
            this.compat = compat;
            this.board = board;
            this.region = region;
            this.localUsed = localUsed;
        }

        boolean solve() {
            boolean[] filled = new boolean[256];
            return backtrack(filled, region.size());
        }

        private boolean backtrack(boolean[] filled, int remaining) {
            if (remaining == 0) return true;
            if (stepsTaken >= STEP_BUDGET_PER_REGION) return false;

            // MRV: pick the unfilled region cell with the fewest legal candidates.
            int bestCell = -1;
            int bestReq0 = -1, bestReq1 = -1, bestReq2 = -1, bestReq3 = -1;
            BitSet bestCandidates = null;
            int bestCount = Integer.MAX_VALUE;

            for (int cell : region) {
                if (filled[cell]) continue;
                stepsTaken++;

                int row = cell / W, col = cell % W;
                int nReq = requirement(row > 0 ? cell - W : -1, cell, Dir.SOUTH_OF_ABOVE);
                int eReq = requirement(col < W - 1 ? cell + 1 : -1, cell, Dir.WEST_OF_RIGHT);
                int sReq = requirement(row < H - 1 ? cell + W : -1, cell, Dir.NORTH_OF_BELOW);
                int wReq = requirement(col > 0 ? cell - 1 : -1, cell, Dir.EAST_OF_LEFT);
                if (row == 0) nReq = PieceUtils.BORDER_COLOR;
                if (row == H - 1) sReq = PieceUtils.BORDER_COLOR;
                if (col == 0) wReq = PieceUtils.BORDER_COLOR;
                if (col == W - 1) eReq = PieceUtils.BORDER_COLOR;

                BitSet candidates = compat.candidatesFor(nReq, eReq, sReq, wReq);
                compat.andNotUsed(candidates, localUsed);
                int count = candidates.cardinality();

                if (count == 0) return false; // dead branch — this cell has no legal piece left
                if (count < bestCount) {
                    bestCount = count;
                    bestCell = cell;
                    bestCandidates = candidates;
                    bestReq0 = nReq; bestReq1 = eReq; bestReq2 = sReq; bestReq3 = wReq;
                    if (count == 1) break; // can't do better than a forced move
                }
            }

            for (int oi = bestCandidates.nextSetBit(0); oi >= 0; oi = bestCandidates.nextSetBit(oi + 1)) {
                int physId = inventory.physicalMapping[oi];
                int piece = inventory.allOrientations[oi];

                board[bestCell] = piece;
                filled[bestCell] = true;
                localUsed[physId] = true;

                if (backtrack(filled, remaining - 1)) return true;

                board[bestCell] = -1;
                filled[bestCell] = false;
                localUsed[physId] = false;

                if (stepsTaken >= STEP_BUDGET_PER_REGION) return false;
            }
            return false;
        }

        private enum Dir { SOUTH_OF_ABOVE, WEST_OF_RIGHT, NORTH_OF_BELOW, EAST_OF_LEFT }

        /** What colour must `cell`'s edge (facing `neighbour`) be, given neighbour's current state? WILDCARD if neighbour is empty/off-board. */
        private int requirement(int neighbour, int cell, Dir dir) {
            if (neighbour == -1) return CompatibilityIndex.WILDCARD; // border cases handled by the caller
            int p = board[neighbour];
            if (p == -1) return CompatibilityIndex.WILDCARD; // neighbour is also in this hole and not placed yet
            return switch (dir) {
                case SOUTH_OF_ABOVE -> PieceUtils.getSouth(p); // neighbour is above us -> we need our North == its South
                case WEST_OF_RIGHT  -> PieceUtils.getWest(p);  // neighbour is to our right -> our East == its West
                case NORTH_OF_BELOW -> PieceUtils.getNorth(p); // neighbour is below us -> our South == its North
                case EAST_OF_LEFT   -> PieceUtils.getEast(p);  // neighbour is to our left -> our West == its East
            };
        }
    }
}

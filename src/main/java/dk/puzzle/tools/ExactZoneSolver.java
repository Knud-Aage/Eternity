package dk.puzzle.tools;

import dk.puzzle.ai.ConflictAnnealer;
import dk.puzzle.io.BucasExporter;
import dk.puzzle.util.PieceUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Whole-zone EXACT minimum-conflict solver ("prove-or-improve").
 *
 * <p>Takes a complete board, extracts the residual zone (all cells near a
 * conflict, or an explicit row band), and answers exactly: what is the
 * minimum conflict count achievable by rearranging ONLY those pieces in ONLY
 * those cells? It searches with an iteratively raised conflict budget
 * B = 0, 1, 2, …:</p>
 *
 * <ul>
 *   <li>level B <b>UNSAT</b> → certificate: the zone cannot be arranged with
 *       ≤ B conflicts (a hard lower bound, even if you stop afterwards);</li>
 *   <li>level B <b>SAT</b> → an arrangement with exactly B conflicts, which is
 *       PROVEN optimal when every lower level already returned UNSAT.</li>
 * </ul>
 *
 * <p>Reaching UNSAT at (incumbent − 1) proves the input board cannot be
 * improved by any rearrangement of the zone — the conflicts are caused by the
 * piece pool itself, and only a different base (different leftover pieces)
 * can do better. This turns days of speculative polishing into a definite
 * answer per board.</p>
 *
 * <p>The search is a depth-first branch-and-bound: most-constrained-cell
 * ordering, cheapest-candidate-first, pruned by budget slack and by an
 * admissible lower bound (sum over open frontier cells of each cell's
 * minimum placement cost — valid because those edge sets are disjoint and
 * placement costs only grow as more neighbours become known). The frame is a
 * hard constraint and pieces stay in their geometric class, matching how
 * every legal Eternity II board is structured. Parallelised by splitting the
 * root cell's candidates across worker threads.</p>
 *
 * Usage:
 *   ExactZoneSolver &lt;link | link-file | board_edges&gt; [options]
 * Options:
 *   --radius N        zone = conflict cells dilated by N (Chebyshev, default 2)
 *   --rows A-B        zone = all cells in 0-based rows A..B inclusive (overrides --radius)
 *   --center R,C      zone = square of the given radius around 0-based cell (R,C) — isolate one cluster
 *   --minutes M       wall-clock budget (default 30)
 *   --threads T       worker threads (default: cores − 2)
 *   --budget-start B  skip levels below B (resume after earlier UNSAT certificates)
 *   --locked          protect the official hint cells (constrained runs)
 */
public final class ExactZoneSolver {

    private static final int W = 16, H = 16, CELLS = 256;
    private static final Set<Integer> HINT_CELLS = Set.of(135, 221, 45, 210, 34);

    // Side kinds, precomputed per zone cell (see buildSideTables)
    private static final int SIDE_HARD_GREY = -2;  // off-board: piece side must be grey
    private static final int SIDE_DYNAMIC = -1;    // neighbour is a zone cell: read the board
    // >= 0: fixed neighbour's constant required colour

    // ------------------------------------------------------------------
    // Immutable problem description (shared by all workers)
    // ------------------------------------------------------------------
    private final int[] fixedBoard;      // board with zone cells = -1
    private final int[] zoneCells;       // zone cell indices, ascending
    private final int[] zoneIndexOf;     // 256 -> index in zoneCells, or -1
    private final int poolSize;
    private final int[] poolRot;         // [poolSize*4] all rotations of each pool piece
    private final int[] poolGreys;       // [poolSize] border count (class)
    private final int[] cellGreys;       // [zoneSize] required class per zone cell
    private final int[][] sideKind;      // [zoneSize][4] N,E,S,W
    private final int[][] sideColor;     // [zoneSize][4] required colour when kind >= 0
    private final int[][] neighborZoneIdx; // [zoneSize][4] zone index of neighbour when DYNAMIC

    // Bitmask machinery over pool-orientation space (poolSize*4 bits):
    // candidate counting per cell becomes a handful of 64-bit ANDs + popcounts
    // instead of a full pool scan (same idea as CompatibilityIndex).
    private static final int NUM_COLORS = 23;
    private final int maskWords;
    private final long[][][] sideColorMask; // [side][color][word]: orientations with that colour on that side
    private final long[][] cellStaticMask;  // [zoneIdx][word]: class filter + hard frame greys

    private final AtomicLong nodes = new AtomicLong();
    private final AtomicBoolean found = new AtomicBoolean();
    private volatile int[] solution;     // zone pieces in zoneCells order when found

    private ExactZoneSolver(int[] board, boolean[] inZone) {
        List<Integer> cells = new ArrayList<>();
        for (int pos = 0; pos < CELLS; pos++) {
            if (inZone[pos]) cells.add(pos);
        }
        int n = cells.size();
        zoneCells = cells.stream().mapToInt(Integer::intValue).toArray();
        zoneIndexOf = new int[CELLS];
        Arrays.fill(zoneIndexOf, -1);
        for (int i = 0; i < n; i++) zoneIndexOf[zoneCells[i]] = i;

        poolSize = n;
        poolRot = new int[n * 4];
        poolGreys = new int[n];
        cellGreys = new int[n];
        fixedBoard = Arrays.copyOf(board, CELLS);
        for (int i = 0; i < n; i++) {
            int piece = board[zoneCells[i]];
            fixedBoard[zoneCells[i]] = -1;
            poolGreys[i] = PieceUtils.getBorderCount(piece);
            int p = piece;
            for (int r = 0; r < 4; r++, p = PieceUtils.rotate(p)) {
                poolRot[i * 4 + r] = p;
            }
            cellGreys[i] = cellClassGreys(zoneCells[i]);
        }

        sideKind = new int[n][4];
        sideColor = new int[n][4];
        neighborZoneIdx = new int[n][4];
        buildSideTables();

        maskWords = (n * 4 + 63) / 64;
        sideColorMask = new long[4][NUM_COLORS][maskWords];
        for (int o = 0; o < n * 4; o++) {
            int piece = poolRot[o];
            sideColorMask[0][PieceUtils.getNorth(piece)][o >> 6] |= 1L << (o & 63);
            sideColorMask[1][PieceUtils.getEast(piece)][o >> 6] |= 1L << (o & 63);
            sideColorMask[2][PieceUtils.getSouth(piece)][o >> 6] |= 1L << (o & 63);
            sideColorMask[3][PieceUtils.getWest(piece)][o >> 6] |= 1L << (o & 63);
        }
        cellStaticMask = new long[n][maskWords];
        for (int i = 0; i < n; i++) {
            long[] mask = cellStaticMask[i];
            for (int o = 0; o < n * 4; o++) {
                if (poolGreys[o / 4] == cellGreys[i]) mask[o >> 6] |= 1L << (o & 63);
            }
            for (int s = 0; s < 4; s++) {
                if (sideKind[i][s] == SIDE_HARD_GREY) {
                    for (int w = 0; w < maskWords; w++) {
                        mask[w] &= sideColorMask[s][PieceUtils.BORDER_COLOR][w];
                    }
                }
            }
        }
    }

    private void buildSideTables() {
        for (int i = 0; i < zoneCells.length; i++) {
            int pos = zoneCells[i];
            int row = pos / W, col = pos % W;
            // side order: 0=N, 1=E, 2=S, 3=W
            int[] nb = {row > 0 ? pos - W : -1, col < W - 1 ? pos + 1 : -1,
                        row < H - 1 ? pos + W : -1, col > 0 ? pos - 1 : -1};
            for (int s = 0; s < 4; s++) {
                if (nb[s] == -1) {
                    sideKind[i][s] = SIDE_HARD_GREY;
                } else if (zoneIndexOf[nb[s]] >= 0) {
                    sideKind[i][s] = SIDE_DYNAMIC;
                    neighborZoneIdx[i][s] = zoneIndexOf[nb[s]];
                } else {
                    int p = fixedBoard[nb[s]];
                    sideKind[i][s] = switch (s) {
                        case 0 -> PieceUtils.getSouth(p);
                        case 1 -> PieceUtils.getWest(p);
                        case 2 -> PieceUtils.getNorth(p);
                        default -> PieceUtils.getEast(p);
                    };
                    sideColor[i][s] = sideKind[i][s];
                }
            }
        }
    }

    private static int cellClassGreys(int pos) {
        int row = pos / W, col = pos % W;
        boolean r = row == 0 || row == H - 1, c = col == 0 || col == W - 1;
        return r && c ? 2 : (r || c ? 1 : 0);
    }

    // ------------------------------------------------------------------
    // One budget level
    // ------------------------------------------------------------------

    private enum LevelOutcome { SAT, UNSAT, TIMEOUT }

    private LevelOutcome solveLevel(int budget, int threads, long deadlineNanos) {
        found.set(false);
        solution = null;

        // Root: most-constrained zone cell on the empty zone, its candidates split across threads.
        Worker root = new Worker(budget, deadlineNanos);
        int rootCell = root.pickCell();
        int rootCount = root.scanCandidates(rootCell, budget, 0, 0);
        if (rootCount == 0) return LevelOutcome.UNSAT;

        int[] rootPieces = Arrays.copyOf(root.candPiece[0], rootCount);
        int[] rootIdx = Arrays.copyOf(root.candPoolIdx[0], rootCount);
        int[] rootCost = Arrays.copyOf(root.candCost[0], rootCount);

        AtomicInteger nextRoot = new AtomicInteger();
        AtomicBoolean timedOut = new AtomicBoolean();
        Thread[] pool = new Thread[Math.max(1, threads)];
        for (int t = 0; t < pool.length; t++) {
            pool[t] = new Thread(() -> {
                Worker w = new Worker(budget, deadlineNanos);
                int i;
                while ((i = nextRoot.getAndIncrement()) < rootPieces.length) {
                    if (found.get()) return;
                    if (System.nanoTime() >= deadlineNanos) {
                        timedOut.set(true);
                        return;
                    }
                    w.place(rootCell, rootPieces[i], rootIdx[i]);
                    if (w.dfs(rootCost[i], 1)) {
                        captureSolution(w);
                        return;
                    }
                    w.unplace(rootCell, rootIdx[i]);
                    if (w.timedOut) {
                        timedOut.set(true);
                        return;
                    }
                }
            }, "zone-solver-" + t);
            pool[t].setDaemon(true);
            pool[t].start();
        }
        for (Thread t : pool) {
            try {
                t.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        if (found.get()) return LevelOutcome.SAT;
        return timedOut.get() ? LevelOutcome.TIMEOUT : LevelOutcome.UNSAT;
    }

    private synchronized void captureSolution(Worker w) {
        if (!found.compareAndSet(false, true)) return;
        int[] sol = new int[zoneCells.length];
        for (int i = 0; i < zoneCells.length; i++) sol[i] = w.board[zoneCells[i]];
        solution = sol;
    }

    // ------------------------------------------------------------------
    // Per-thread DFS worker
    // ------------------------------------------------------------------

    private final class Worker {
        final int budget;
        final long deadline;
        final int[] board = Arrays.copyOf(fixedBoard, CELLS); // zone cells -1
        final boolean[] used = new boolean[poolSize];
        // Per-depth candidate buffers (bucket-sorted by cost 0..4)
        final int[][] candPiece = new int[poolSize + 1][poolSize * 4];
        final int[][] candPoolIdx = new int[poolSize + 1][poolSize * 4];
        final int[][] candCost = new int[poolSize + 1][poolSize * 4];
        // Reusable bucket scratch (allocation-free hot path)
        final int[][] bPiece = new int[5][poolSize * 4];
        final int[][] bIdx = new int[5][poolSize * 4];
        final int[] bN = new int[5];
        final long[] usedMask = new long[maskWords]; // bit o set = orientation o's piece is used
        final long[] tmp = new long[maskWords];
        final long[] base = new long[maskWords];
        final int[] reqSide = new int[4];
        final int[] reqColor = new int[4];
        long localNodes = 0;
        boolean timedOut = false;

        Worker(int budget, long deadline) {
            this.budget = budget;
            this.deadline = deadline;
        }

        /** Root cell for the level: the zone cell with the most fixed/border sides. */
        int pickCell() {
            int best = -1, bestKnown = -1;
            for (int i = 0; i < zoneCells.length; i++) {
                int known = 0;
                for (int s = 0; s < 4; s++) {
                    int kind = sideKind[i][s];
                    if (kind == SIDE_HARD_GREY || kind >= 0) known++;
                }
                if (known > bestKnown) {
                    bestKnown = known;
                    best = zoneCells[i];
                }
            }
            return best;
        }

        void place(int cell, int piece, int poolIdx) {
            board[cell] = piece;
            used[poolIdx] = true;
            usedMask[poolIdx >> 4] |= 0xFL << ((poolIdx & 15) << 2);
        }

        void unplace(int cell, int poolIdx) {
            board[cell] = -1;
            used[poolIdx] = false;
            usedMask[poolIdx >> 4] &= ~(0xFL << ((poolIdx & 15) << 2));
        }

        /**
         * Depth-first search. costSoFar includes every edge whose second
         * endpoint is already placed. Returns true when a complete fill with
         * total cost ≤ budget was found (board left in the solved state).
         */
        boolean dfs(int costSoFar, int depth) {
            if (depth == zoneCells.length) return true;
            if ((++localNodes & 0x3FFF) == 0) {
                nodes.addAndGet(0x4000);
                if (found.get()) return false;
                if (System.nanoTime() >= deadline) {
                    timedOut = true;
                    return false;
                }
            }

            // Frontier scan via bitmasks: for each open cell count its EXACT
            // (zero-mismatch) candidates with a few ANDs + popcounts. Cells
            // with none contribute >= 1 conflict each — an admissible bound,
            // since each open cell's known edges are disjoint edge sets.
            int bestCell = -1, bestCount = Integer.MAX_VALUE;
            int zeroExactCell = -1, zeroExactCells = 0;
            for (int i = 0; i < zoneCells.length; i++) {
                int cell = zoneCells[i];
                if (board[cell] != -1) continue;

                boolean frontier = false;
                long any = 0;
                long[] stat = cellStaticMask[i];
                for (int w = 0; w < maskWords; w++) {
                    base[w] = stat[w] & ~usedMask[w];
                    tmp[w] = base[w];
                    any |= base[w];
                }
                if (any == 0) return false; // no legal piece at ANY cost — dead branch

                int nReq = 0;
                for (int s = 0; s < 4; s++) {
                    int kind = sideKind[i][s];
                    int req;
                    if (kind >= 0) {
                        req = kind; // fixed neighbour colour (soft, but exact-fit requires match)
                    } else if (kind == SIDE_DYNAMIC) {
                        int nb = board[zoneCells[neighborZoneIdx[i][s]]];
                        if (nb == -1) continue;
                        req = switch (s) {
                            case 0 -> PieceUtils.getSouth(nb);
                            case 1 -> PieceUtils.getWest(nb);
                            case 2 -> PieceUtils.getNorth(nb);
                            default -> PieceUtils.getEast(nb);
                        };
                    } else {
                        frontier = true; // hard grey already folded into the static mask
                        continue;
                    }
                    frontier = true;
                    reqSide[nReq] = s;
                    reqColor[nReq++] = req;
                    long[] m = sideColorMask[s][req];
                    for (int w = 0; w < maskWords; w++) tmp[w] &= m[w];
                }
                if (!frontier) continue; // nothing known yet — no information here

                int exact = 0;
                for (int w = 0; w < maskWords; w++) exact += Long.bitCount(tmp[w]);

                if (exact == 0) {
                    // This cell will cost at least 1. Check if even a single
                    // mismatch is impossible (no piece matches all-but-one
                    // constraint): then it costs at least 2.
                    int minHere = 2;
                    for (int excl = 0; excl < nReq && minHere == 2; excl++) {
                        long anyRelaxed = 0;
                        for (int w = 0; w < maskWords; w++) {
                            long v = base[w];
                            for (int j = 0; j < nReq; j++) {
                                if (j != excl) v &= sideColorMask[reqSide[j]][reqColor[j]][w];
                            }
                            anyRelaxed |= v;
                        }
                        if (anyRelaxed != 0) minHere = 1;
                    }
                    zeroExactCells += minHere;
                    if (costSoFar + zeroExactCells > budget) return false; // admissible bound
                    if (zeroExactCell == -1) zeroExactCell = cell;
                } else if (exact < bestCount) {
                    bestCount = exact;
                    bestCell = cell;
                }
            }

            // Fail-first: expand a forced-conflict cell when one exists (it
            // consumes budget immediately, tightening every child), otherwise
            // the cell with the fewest exact fits, otherwise any open cell.
            if (zeroExactCell != -1) bestCell = zeroExactCell;
            if (bestCell == -1) {
                for (int cell : zoneCells) {
                    if (board[cell] == -1) {
                        bestCell = cell;
                        break;
                    }
                }
            }

            int count = scanCandidates(bestCell, budget, costSoFar, depth);
            int[] pieces = candPiece[depth];
            int[] idxs = candPoolIdx[depth];
            int[] costs = candCost[depth];
            for (int i = 0; i < count; i++) {
                int c = costs[i];
                if (costSoFar + c > budget) break; // bucket-sorted: rest are worse
                place(bestCell, pieces[i], idxs[i]);
                if (dfs(costSoFar + c, depth + 1)) return true;
                unplace(bestCell, idxs[i]);
                if (timedOut || found.get()) return false;
            }
            return false;
        }

        /** Soft-mismatch count for placing this oriented piece; MAX_VALUE if a hard (frame) constraint fails. */
        int placementCost(int i, int piece) {
            int cost = 0;
            for (int s = 0; s < 4; s++) {
                int side = switch (s) {
                    case 0 -> PieceUtils.getNorth(piece);
                    case 1 -> PieceUtils.getEast(piece);
                    case 2 -> PieceUtils.getSouth(piece);
                    default -> PieceUtils.getWest(piece);
                };
                int kind = sideKind[i][s];
                if (kind == SIDE_HARD_GREY) {
                    if (side != PieceUtils.BORDER_COLOR) return Integer.MAX_VALUE;
                } else if (kind >= 0) {
                    if (side != sideColor[i][s]) cost++;
                } else {
                    int nb = board[zoneCells[neighborZoneIdx[i][s]]];
                    if (nb != -1) {
                        int facing = switch (s) {
                            case 0 -> PieceUtils.getSouth(nb);
                            case 1 -> PieceUtils.getWest(nb);
                            case 2 -> PieceUtils.getNorth(nb);
                            default -> PieceUtils.getEast(nb);
                        };
                        if (side != facing) cost++;
                    }
                }
            }
            return cost;
        }

        /**
         * Fills the depth-level candidate buffers for {@code cell},
         * bucket-sorted by placement cost. Returns the candidate count.
         */
        int scanCandidates(int cell, int budget, int costSoFar, int depth) {
            int i = zoneIndexOf[cell];
            int slack = budget - costSoFar;
            int wanted = cellGreys[i];

            Arrays.fill(bN, 0);
            for (int k = 0; k < poolSize; k++) {
                if (used[k] || poolGreys[k] != wanted) continue;
                for (int r = 0; r < 4; r++) {
                    int piece = poolRot[k * 4 + r];
                    int c = placementCost(i, piece);
                    if (c > slack || c > 4) continue;
                    bPiece[c][bN[c]] = piece;
                    bIdx[c][bN[c]++] = k;
                }
            }
            int n = 0;
            for (int c = 0; c <= 4; c++) {
                for (int j = 0; j < bN[c]; j++) {
                    candPiece[depth][n] = bPiece[c][j];
                    candPoolIdx[depth][n] = bIdx[c][j];
                    candCost[depth][n++] = c;
                }
            }
            return n;
        }
    }

    // ------------------------------------------------------------------
    // Zone construction + incumbent accounting
    // ------------------------------------------------------------------

    private static boolean[] zoneFromConflicts(int[] board, int radius, Set<Integer> locked) {
        boolean[] zone = new boolean[CELLS];
        for (int pos = 0; pos < CELLS; pos++) {
            if (!hasConflict(board, pos)) continue;
            int cr = pos / W, cc = pos % W;
            for (int r = Math.max(0, cr - radius); r <= Math.min(H - 1, cr + radius); r++) {
                for (int c = Math.max(0, cc - radius); c <= Math.min(W - 1, cc + radius); c++) {
                    if (!locked.contains(r * W + c)) zone[r * W + c] = true;
                }
            }
        }
        return zone;
    }

    private static boolean[] zoneFromRows(int rowFrom, int rowTo, Set<Integer> locked) {
        boolean[] zone = new boolean[CELLS];
        for (int r = rowFrom; r <= rowTo; r++) {
            for (int c = 0; c < W; c++) {
                if (!locked.contains(r * W + c)) zone[r * W + c] = true;
            }
        }
        return zone;
    }

    private static boolean hasConflict(int[] board, int pos) {
        int p = board[pos];
        int row = pos / W, col = pos % W;
        if (row > 0 && PieceUtils.getNorth(p) != PieceUtils.getSouth(board[pos - W])) return true;
        if (row < H - 1 && PieceUtils.getSouth(p) != PieceUtils.getNorth(board[pos + W])) return true;
        if (col > 0 && PieceUtils.getWest(p) != PieceUtils.getEast(board[pos - 1])) return true;
        if (col < W - 1 && PieceUtils.getEast(p) != PieceUtils.getWest(board[pos + 1])) return true;
        return false;
    }

    /** Cost of the incumbent zone arrangement: every internal edge incident to the zone, counted once. */
    private int incumbentZoneCost(int[] board) {
        int cost = 0;
        for (int pos = 0; pos < CELLS; pos++) {
            int row = pos / W, col = pos % W;
            if (col < W - 1) {
                boolean incident = zoneIndexOf[pos] >= 0 || zoneIndexOf[pos + 1] >= 0;
                if (incident && PieceUtils.getEast(board[pos]) != PieceUtils.getWest(board[pos + 1])) cost++;
            }
            if (row < H - 1) {
                boolean incident = zoneIndexOf[pos] >= 0 || zoneIndexOf[pos + W] >= 0;
                if (incident && PieceUtils.getSouth(board[pos]) != PieceUtils.getNorth(board[pos + W])) cost++;
            }
        }
        return cost;
    }

    private int[] rebuildFullBoard() {
        int[] full = Arrays.copyOf(fixedBoard, CELLS);
        for (int i = 0; i < zoneCells.length; i++) full[zoneCells[i]] = solution[i];
        return full;
    }

    // ------------------------------------------------------------------
    // CLI
    // ------------------------------------------------------------------

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.out.println("Usage: ExactZoneSolver <link | link-file | board_edges> " +
                    "[--radius N] [--rows A-B] [--minutes M] [--threads T] [--budget-start B] [--locked]");
            return;
        }

        String input = args[0];
        Path asPath = Path.of(input);
        if (Files.isRegularFile(asPath)) input = Files.readString(asPath);

        int radius = 2, rowFrom = -1, rowTo = -1, threads = Math.max(1, Runtime.getRuntime().availableProcessors() - 2);
        int centerRow = -1, centerCol = -1;
        long minutes = 30;
        int budgetStart = 0;
        boolean lockHints = false;
        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--radius" -> radius = Integer.parseInt(args[++i]);
                case "--center" -> {
                    String[] parts = args[++i].split(",");
                    centerRow = Integer.parseInt(parts[0]);
                    centerCol = Integer.parseInt(parts[1]);
                }
                case "--rows" -> {
                    String[] parts = args[++i].split("-");
                    rowFrom = Integer.parseInt(parts[0]);
                    rowTo = Integer.parseInt(parts[1]);
                }
                case "--minutes" -> minutes = Long.parseLong(args[++i]);
                case "--threads" -> threads = Integer.parseInt(args[++i]);
                case "--budget-start" -> budgetStart = Integer.parseInt(args[++i]);
                case "--locked" -> lockHints = true;
                default -> {
                    System.out.println("Unknown option: " + args[i]);
                    return;
                }
            }
        }

        String boardEdges = HoleSolver.extractBoardEdges(input);
        if (boardEdges.length() != 1024) {
            System.out.println("Expected a 1024-character board_edges value, got " + boardEdges.length() + " chars.");
            return;
        }
        int[] board = HoleSolver.decodeBoard(boardEdges);
        for (int p : board) {
            if (p == -1) {
                System.out.println("Board has empty cells — fill it first (HoleSolver).");
                return;
            }
        }

        Set<Integer> locked = lockHints ? HINT_CELLS : Set.of();

        int rawConflicts = ConflictAnnealer.countInternalConflicts(board);
        int rawFrame = ConflictAnnealer.countBorderViolations(board);
        ConflictAnnealer.normalize(board, locked);
        int conflicts = ConflictAnnealer.countInternalConflicts(board);
        System.out.printf("Input: %d internal conflicts", rawConflicts);
        if (rawFrame > 0) {
            System.out.printf(" + %d frame violations -> normalized to %d clean conflicts", rawFrame, conflicts);
        }
        System.out.println();

        boolean[] zone;
        if (centerRow >= 0) {
            zone = new boolean[CELLS];
            for (int r = Math.max(0, centerRow - radius); r <= Math.min(H - 1, centerRow + radius); r++) {
                for (int c = Math.max(0, centerCol - radius); c <= Math.min(W - 1, centerCol + radius); c++) {
                    if (!locked.contains(r * W + c)) zone[r * W + c] = true;
                }
            }
        } else if (rowFrom >= 0) {
            zone = zoneFromRows(rowFrom, rowTo, locked);
        } else {
            zone = zoneFromConflicts(board, radius, locked);
        }

        ExactZoneSolver solver = new ExactZoneSolver(board, zone);
        int zoneSize = solver.zoneCells.length;
        int incumbent = solver.incumbentZoneCost(board);

        int minRow = 15, maxRow = 0;
        for (int pos : solver.zoneCells) {
            minRow = Math.min(minRow, pos / W);
            maxRow = Math.max(maxRow, pos / W);
        }
        System.out.printf("Zone: %d cells (rows %d-%d), incumbent zone cost %d, budget levels %d..%d%n",
                zoneSize, minRow, maxRow, incumbent, budgetStart, incumbent - 1);
        System.out.printf("Running %d threads, %d minute(s) wall clock.%n%n", threads, minutes);

        long deadline = System.nanoTime() + minutes * 60_000_000_000L;

        // Progress reporter
        Thread reporter = getReporterThread(solver);

        int provenFloor = budgetStart; // zone cost proven >= this
        for (int budget = budgetStart; budget < incumbent; budget++) {
            long levelStart = System.nanoTime();
            long nodesBefore = solver.nodes.get();
            LevelOutcome outcome = solver.solveLevel(budget, threads, deadline);
            double secs = (System.nanoTime() - levelStart) / 1e9;
            long levelNodes = solver.nodes.get() - nodesBefore;

            switch (outcome) {
                case UNSAT -> {
                    provenFloor = budget + 1;
                    System.out.printf("[B=%d] UNSAT in %.1fs (%,d nodes) -> zone cost PROVEN >= %d%n",
                            budget, secs, levelNodes, provenFloor);
                }
                case SAT -> {
                    int[] full = solver.rebuildFullBoard();
                    int total = ConflictAnnealer.countInternalConflicts(full);
                    System.out.printf("[B=%d] SAT in %.1fs (%,d nodes)!%n", budget, secs, levelNodes);
                    System.out.printf("IMPROVED BOARD: %d total conflicts (was %d)%s%n",
                            total, conflicts,
                            budget == provenFloor ? " — PROVEN OPTIMAL for this zone" : "");
                    System.out.println(BucasExporter.exportBoard(full));
                    reporter.interrupt();
                    return;
                }
                case TIMEOUT -> {
                    System.out.printf("[B=%d] TIMEOUT after %.1fs (%,d nodes this level).%n", budget, secs, levelNodes);
                    System.out.printf("Certified so far: zone cost >= %d. Resume with --budget-start %d and more time.%n",
                            provenFloor, budget);
                    reporter.interrupt();
                    return;
                }
            }
        }

        reporter.interrupt();
        System.out.println();
        System.out.printf("PROVEN: no rearrangement of this %d-cell zone beats the incumbent %d conflicts.%n",
                zoneSize, conflicts);
        System.out.println("The conflicts are caused by the piece pool (the base), not the arrangement.");
        System.out.println("This base/lineage cannot improve without changing which pieces are left over.");
    }

    private static Thread getReporterThread(ExactZoneSolver solver) {
        Thread reporter = new Thread(() -> {
            long start = System.nanoTime();
            while (!Thread.currentThread().isInterrupted()) {
                try {
                    Thread.sleep(60_000);
                } catch (InterruptedException e) {
                    return;
                }
                long n = solver.nodes.get();
                double elapsed = (System.nanoTime() - start) / 1e9;
                System.out.printf("    ... %,d nodes, %.1fM nodes/s, %.0fs elapsed%n",
                        n, n / elapsed / 1e6, elapsed);
            }
        }, "zone-progress");
        reporter.setDaemon(true);
        reporter.start();
        return reporter;
    }
}
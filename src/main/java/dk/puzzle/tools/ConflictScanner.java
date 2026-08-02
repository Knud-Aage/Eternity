package dk.puzzle.tools;

import dk.puzzle.ai.ConflictReducer;
import dk.puzzle.core.Eternity;
import dk.puzzle.model.PieceInventory;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Scans a whole directory of Blackwood-solver save files (the
 * "{pieces}_{guid}_{rand}.txt" format both Save_Board and BwUtil.saveBoard
 * produce) and reports the RAW edge-conflict count for every one of them --
 * not just the single deepest (all-time piece-count record) board.
 *
 * <p>A save that never set a new depth record can still have fewer genuine
 * conflicts than the current deepest board: the break-count budget only ever
 * accumulates within a single attempt, so depth and conflict count are not
 * the same axis across DIFFERENT attempts. A tracker that only ever looks at
 * "the current maximum piece count" silently ignores every one of those.</p>
 *
 * <p><b>Conflicts are only counted among the pieces actually placed</b> --
 * empty cells are skipped entirely, never counted as a conflict (matches
 * {@link ConflictReducer#countConflicts}'s own semantics). A shallow board
 * with "0 conflicts" is NOT a solved puzzle; it just means whatever pieces
 * are placed happen to fit together, with everything else left empty.</p>
 *
 * <p><b>Very shallow boards are trivially, meaninglessly conflict-free</b>:
 * Blackwood's algorithm allows zero breaks at all for steps 1-200 (the break
 * budget only unlocks starting at step 201), so any saved board with roughly
 * 190-200 pieces is GUARANTEED 0 conflicts by construction -- no skill or
 * luck involved. Comparing "lowest conflict count" across the WHOLE pool of
 * saved boards is therefore meaningless: it will almost always immediately
 * latch onto 0 from one of these trivial shallow boards and never move
 * again. Pass {@code minPieces} to restrict the comparison to boards deep
 * enough to be genuinely competitive (e.g. within ~15 pieces of the current
 * depth record) -- that's the only way "lowest conflicts" says anything
 * about whether deep attempts are actually getting cleaner over time.</p>
 *
 * <p>Deliberately fast: only decodes each board and counts conflicts on the
 * pieces actually placed ({@link ConflictReducer#countConflicts}) -- it does
 * NOT run {@link HoleSolver}'s exact-search + heuristic hole-filling, which
 * is orders of magnitude slower and meant for deep-diving one specific
 * board, not scanning thousands of them in one pass.</p>
 */
public class ConflictScanner {

    private static final int BOARD_SIZE = 256;

    public record ScanRow(String filename, int pieces, int conflicts) {
        public int holes() {
            return BOARD_SIZE - pieces;
        }
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.out.println("Usage: ConflictScanner <directory> [outputCsv] [minPieces]");
            System.out.println("  minPieces: skip boards shallower than this (see class Javadoc for why --");
            System.out.println("  boards below roughly 200 pieces are trivially, meaninglessly 0-conflict).");
            return;
        }
        Path dir = Path.of(args[0]);
        int minPieces = args.length >= 3 ? Integer.parseInt(args[2]) : 0;
        PieceInventory inventory = new PieceInventory(Eternity.loadPieces());

        List<ScanRow> rows = scan(dir, inventory, minPieces);

        List<ScanRow> byConflicts = new ArrayList<>(rows);
        byConflicts.sort(Comparator.comparingInt(ScanRow::conflicts)
                .thenComparing(Comparator.comparingInt(ScanRow::pieces).reversed()));

        List<ScanRow> byDepth = new ArrayList<>(rows);
        byDepth.sort(Comparator.comparingInt(ScanRow::pieces).reversed());

        System.out.println("Scanned " + rows.size() + " save file(s) in " + dir
                + (minPieces > 0 ? " (pieces >= " + minPieces + ")" : ""));
        System.out.println();
        System.out.println("Top 10 lowest-conflict boards (all PARTIAL boards -- holes = 256 - pieces, never counted as conflicts):");
        byConflicts.stream().limit(10).forEach(r ->
                System.out.printf("  %4d conflicts  %3d pieces  %3d holes  %s%n", r.conflicts(), r.pieces(), r.holes(), r.filename()));

        System.out.println();
        System.out.println("Top 10 deepest boards:");
        byDepth.stream().limit(10).forEach(r ->
                System.out.printf("  %3d pieces  %3d holes  %4d conflicts  %s%n", r.pieces(), r.holes(), r.conflicts(), r.filename()));

        if (args.length >= 2) {
            try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(Path.of(args[1])))) {
                out.println("filename,pieces,holes,conflicts");
                for (ScanRow r : byConflicts) {
                    out.printf("%s,%d,%d,%d%n", r.filename(), r.pieces(), r.holes(), r.conflicts());
                }
            }
            System.out.println();
            System.out.println("Full results written to " + args[1]);
        }
    }

    /** Same as {@link #scan(Path, PieceInventory, int)} with no depth filter (minPieces=0). */
    public static List<ScanRow> scan(Path dir, PieceInventory inventory) throws IOException {
        return scan(dir, inventory, 0);
    }

    /**
     * Scans every *.txt file directly inside {@code dir}; files that can't be parsed, or whose
     * piece count is below {@code minPieces}, are silently skipped (see class Javadoc for why a
     * depth filter matters here).
     */
    public static List<ScanRow> scan(Path dir, PieceInventory inventory, int minPieces) throws IOException {
        List<ScanRow> rows = new ArrayList<>();
        ConflictReducer reducer = new ConflictReducer(inventory, false);

        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir, "*.txt")) {
            for (Path file : stream) {
                ScanRow row = scanFile(file, inventory, reducer, minPieces);
                if (row != null) {
                    rows.add(row);
                }
            }
        }
        return rows;
    }

    private static ScanRow scanFile(Path file, PieceInventory inventory, ConflictReducer reducer, int minPieces) throws IOException {
        String filename = file.getFileName().toString();
        int pieces = parsePieceCountFromFilename(filename);
        if (pieces == -1 || pieces < minPieces) {
            return null; // depth check happens before ever reading/decoding the file -- cheap early skip
        }

        String link = null;
        for (String line : Files.readAllLines(file)) {
            if (line.contains("https://")) {
                link = line.trim();
                break;
            }
        }
        if (link == null) {
            return null;
        }

        int[] board = HoleSolver.decodeBoardAuto(link, inventory, false);
        int conflicts = reducer.countConflicts(board);
        return new ScanRow(filename, pieces, conflicts);
    }

    /** Filenames are "{pieces}_{guid-or-hash}_{rand}.txt" -- pieces is the leading token before the first underscore. */
    static int parsePieceCountFromFilename(String filename) {
        int underscore = filename.indexOf('_');
        if (underscore == -1) {
            return -1;
        }
        try {
            return Integer.parseInt(filename.substring(0, underscore));
        } catch (NumberFormatException e) {
            return -1;
        }
    }
}

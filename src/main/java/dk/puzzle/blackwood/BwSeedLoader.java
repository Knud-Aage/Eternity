package dk.puzzle.blackwood;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Loads previously saved deep boards and converts them into the step-ordered form the GPU kernel
 * replays from, so a run can start at the frontier instead of re-deriving 250 pieces of
 * already-known progress from scratch every time.
 *
 * <p>Reads the grid written by {@link BwUtil#buildBoardString} -- 16 lines, <b>board row 15
 * first</b>, cells of the form {@code "NNN/R"} (piece number / rotation) or {@code "---/-"} for
 * empty. The C# solver's own {@code Save_Board} emits a byte-identical grid layout, so boards from
 * either solver load through this same parser; only the trailing bucas link differs, and that is
 * ignored here.</p>
 *
 * <p>The kernel needs pieces in SEARCH-STEP order, not board order, because it replays step 0, 1,
 * 2... along Blackwood's inward spiral. Board index for each step comes from
 * {@code GpuTableSet.stepBoardIdx}. A seed is truncated at the first step whose board cell is
 * empty: the kernel can resume from a prefix, but a hole mid-sequence would leave later pieces
 * with no valid neighbour context.</p>
 */
public final class BwSeedLoader {

    private static final Logger logger = LogManager.getLogger(BwSeedLoader.class);

    private BwSeedLoader() {
    }

    /** One saved board, already converted to the kernel's step-ordered encoding. */
    public record Seed(Path source, int depth, int[] stepEncoded) {
    }

    /**
     * @param dirs         directories to scan (missing ones are skipped, not an error)
     * @param minDepth     ignore boards shallower than this
     * @param maxSeeds     keep at most this many, deepest first
     * @param stepBoardIdx step -> board index, from {@code GpuTableSet.stepBoardIdx}
     */
    public static List<Seed> load(List<Path> dirs, int minDepth, int maxSeeds, int[] stepBoardIdx) {
        List<Seed> seeds = new ArrayList<>();
        for (Path dir : dirs) {
            if (!Files.isDirectory(dir)) {
                logger.debug("Seed directory {} does not exist, skipping", dir);
                continue;
            }
            int loadedHere = 0;
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir, "*.txt")) {
                for (Path file : stream) {
                    // Cheap pre-filter on the filename's leading depth ("<pieces>_<hash>_<n>.txt")
                    // so a directory of 14,000 shallow boards isn't fully parsed to reject them.
                    if (depthFromFilename(file) < minDepth) continue;
                    Seed seed = parse(file, stepBoardIdx);
                    if (seed != null && seed.depth() >= minDepth) {
                        seeds.add(seed);
                        loadedHere++;
                    }
                }
            } catch (IOException e) {
                logger.warn("Could not scan seed directory {}", dir, e);
            }
            if (loadedHere > 0) logger.info("Loaded {} candidate seed board(s) from {}", loadedHere, dir);
        }

        seeds.sort(Comparator.comparingInt(Seed::depth).reversed());
        return seeds.size() > maxSeeds ? new ArrayList<>(seeds.subList(0, maxSeeds)) : seeds;
    }

    /** -1 when the name does not carry a parseable leading depth. */
    private static int depthFromFilename(Path file) {
        String name = file.getFileName().toString();
        int underscore = name.indexOf('_');
        if (underscore <= 0) return -1;
        try {
            return Integer.parseInt(name.substring(0, underscore));
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    /** Returns null if the file is not a readable board grid. */
    static Seed parse(Path file, int[] stepBoardIdx) {
        List<String> lines;
        try {
            lines = Files.readAllLines(file);
        } catch (IOException e) {
            logger.warn("Could not read seed board {}", file, e);
            return null;
        }
        if (lines.size() < 16) return null;

        // board[boardIdx] = (pieceNumber << 2) | rotation, or -1 where empty.
        int[] byBoardIdx = new int[256];
        java.util.Arrays.fill(byBoardIdx, -1);

        for (int line = 0; line < 16; line++) {
            String[] cells = lines.get(line).trim().split("\\s+");
            if (cells.length < 16) return null;
            int boardRow = 15 - line; // the grid is written with row 15 first
            for (int col = 0; col < 16; col++) {
                String cell = cells[col];
                int slash = cell.indexOf('/');
                if (slash <= 0) return null;
                String pieceText = cell.substring(0, slash);
                String rotText = cell.substring(slash + 1);
                if (pieceText.startsWith("-") || rotText.startsWith("-")) continue; // empty cell
                try {
                    int pieceNumber = Integer.parseInt(pieceText.trim());
                    int rotation = Integer.parseInt(rotText.trim());
                    if (pieceNumber < 1 || pieceNumber > 256 || rotation < 0 || rotation > 3) return null;
                    byBoardIdx[boardRow * 16 + col] = (pieceNumber << 2) | rotation;
                } catch (NumberFormatException e) {
                    return null;
                }
            }
        }

        int[] stepEncoded = new int[256];
        java.util.Arrays.fill(stepEncoded, -1);
        int depth = 0;
        for (int step = 0; step < 256; step++) {
            int v = byBoardIdx[stepBoardIdx[step]];
            if (v < 0) break; // truncate at the first gap -- a prefix is resumable, a hole is not
            stepEncoded[step] = v;
            depth++;
        }
        if (depth == 0) return null;
        return new Seed(file, depth, stepEncoded);
    }
}

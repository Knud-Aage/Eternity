package dk.puzzle.tools;

import dk.puzzle.ai.ConflictAnnealer;
import dk.puzzle.io.BucasExporter;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;

/**
 * CLI front-end for {@link ConflictAnnealer}: takes a complete board (bucas
 * link, link file, or raw board_edges string), runs the simulated-annealing
 * conflict minimiser for a wall-clock budget, and prints the improved board
 * as a bucas link.
 *
 * <p>Complements {@link HoleSolver}: HoleSolver answers "can these exact
 * regions reach zero conflicts?" exactly; BoardOptimizer instead spends its
 * whole budget pushing the global conflict count as low as it can get,
 * regardless of provability.</p>
 *
 * Usage:
 *   BoardOptimizer &lt;link | link-file | board_edges&gt; [seconds=60] [threads=half the cores] [--locked]
 *
 * --locked protects the official hint cells (center 135 + 221/45/210/34) from
 * being moved, for optimising constrained-run boards.
 */
public final class BoardOptimizer {

    private static final Set<Integer> HINT_CELLS = Set.of(135, 221, 45, 210, 34);

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.out.println("Usage: BoardOptimizer <link | link-file | board_edges> [seconds=60] [threads] [--locked]");
            return;
        }

        String input = args[0];
        Path asPath = Path.of(input);
        if (Files.isRegularFile(asPath)) {
            input = Files.readString(asPath);
        }

        long seconds = 60;
        int threads = Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
        boolean lockHints = false;
        int bareNumbers = 0;
        for (int i = 1; i < args.length; i++) {
            if ("--locked".equalsIgnoreCase(args[i])) {
                lockHints = true;
            } else if (args[i].matches("\\d+")) {
                if (bareNumbers++ == 0) seconds = Long.parseLong(args[i]);
                else threads = Integer.parseInt(args[i]);
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
                System.out.println("Board has empty cells — run HoleSolver first to fill them, then optimise.");
                return;
            }
        }

        int startConflicts = ConflictAnnealer.countInternalConflicts(board);
        int startBorder = ConflictAnnealer.countBorderViolations(board);
        System.out.printf("Start: %d / 480 internal edge conflicts%s%n", startConflicts,
                startBorder > 0 ? " (WARNING: " + startBorder + " frame violations!)" : "");
        System.out.printf("Annealing for %ds on %d threads%s...%n", seconds, threads,
                lockHints ? " (hint cells locked)" : "");

        ConflictAnnealer.Result result = ConflictAnnealer.optimize(
                board,
                lockHints ? HINT_CELLS : Set.of(),
                seconds * 1000L,
                threads,
                (b, conflicts, elapsedMs) ->
                        System.out.printf("  %6.1fs  ->  %d conflicts%n", elapsedMs / 1000.0, conflicts));

        System.out.println();
        System.out.printf("Stats: %,d SA moves, %,d LNS kicks (%,d found improvements)%n",
                result.moves(), result.lnsKicks(), result.lnsWins());
        System.out.printf("Final: %d / 480 internal edge conflicts (started at %d)%n",
                result.internalConflicts(), startConflicts);
        if (result.borderConflicts() > 0) {
            System.out.println("WARNING: " + result.borderConflicts() + " frame violations remain.");
        }
        System.out.println(BucasExporter.exportBoard(result.board()));
    }
}
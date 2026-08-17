package dk.puzzle.blackwood;

import dk.puzzle.core.Eternity;
import dk.puzzle.gpu.BlackwoodGpuEngine;
import dk.puzzle.model.PieceInventory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Verifies {@code BlackwoodGpuRunner.trySave}'s new conflict-based scoring against REAL kernel
 * output, not a hand-built fixture -- reconstructing a synthetic board in the kernel's own packed
 * format host-side would risk exactly the kind of format-guessing mistake this project has already
 * been burned by once (three attempts to reconstruct a link before getting the packing right).
 *
 * <p>Uses a scratch output directory and an artificially low {@code currentHighScore} so the very
 * first real launch trivially counts as a new record, forcing {@code trySave} to run on genuine
 * GPU-produced board data. 'Harness' suffix keeps Surefire from collecting it; needs real CUDA
 * hardware.</p>
 */
public class BlackwoodGpuScoringHarness {

    private static final String PIECES_PATH = "src/main/resources/JBlackwood_Pieces.txt";
    private static final Pattern LABELLED_NAME = Pattern.compile("^Errors(\\d+)_Base(\\d+)_.*_RawBoard\\.txt$");

    public static void main(String[] args) throws Exception {
        Path scratchDir = Files.createTempDirectory("bw_gpu_scoring_harness_");
        System.out.println("=== Scoring harness, scratch dir: " + scratchDir + " ===");

        BlackwoodSolver solver = new BlackwoodSolver(190, scratchDir, 1, PIECES_PATH);
        List<BwPiece> pieces = BwUtil.getPieces(PIECES_PATH);
        BwPiece[] pieceByNumber = new BwPiece[257];
        for (BwPiece p : pieces) pieceByNumber[p.pieceNumber()] = p;
        PieceInventory inventory = new PieceInventory(Eternity.loadPieces());

        solver.prepare();
        BlackwoodGpuEngine engine = new BlackwoodGpuEngine();
        engine.uploadTables(BwGpuTables.build(solver));
        engine.resetEpoch();

        System.out.println("Running until a depth record fires trySave (currentHighScore starts at 0)...");
        int currentHighScore = 0;
        int[] bestBoardOut = new int[256];
        for (int launch = 0; launch < 200 && currentHighScore == 0; launch++) {
            BlackwoodGpuEngine.GpuResult r =
                    engine.runBlackwoodDfs(System.nanoTime(), 50_000L, 512, currentHighScore, bestBoardOut);
            if (r.newHighScore() > currentHighScore) {
                currentHighScore = r.newHighScore();
                System.out.printf("Launch %d: newHighScore=%d, invoking trySave...%n", launch, currentHighScore);
                BlackwoodGpuRunner.trySave(bestBoardOut, currentHighScore, pieceByNumber, inventory, scratchDir);
            }
        }

        System.out.println();
        System.out.println("=== Result ===");
        try (var stream = Files.newDirectoryStream(scratchDir)) {
            boolean any = false;
            for (Path p : stream) {
                any = true;
                String name = p.getFileName().toString();
                System.out.println("  " + name + "  (" + Files.size(p) + " bytes)");
                Matcher m = LABELLED_NAME.matcher(name);
                if (m.matches()) {
                    System.out.printf("    -> parsed: conflicts=%s depth=%s%n", m.group(1), m.group(2));
                }
            }
            if (!any) {
                System.out.println("NOTHING SAVED. Either no depth record fired in 200 launches, or trySave");
                System.out.println("rejected every candidate (bestConflictsOnDisk starts at MAX_VALUE for an");
                System.out.println("empty dir, so keepThreshold is MAX_VALUE and the first save should never");
                System.out.println("be rejected -- investigate if this happens).");
                System.exit(1);
            }
        }

        System.out.println();
        System.out.println("Cross-check the printed conflict count above against a manual");
        System.out.println("HoleSolver CLI run on the same file's link if you want independent confirmation.");
    }
}

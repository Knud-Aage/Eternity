package dk.puzzle.blackwood;

import dk.puzzle.gpu.BlackwoodGpuEngine;

/**
 * Finds the real WDDM TDR ceiling on this machine/driver by deliberately escalating a single
 * launch's stepBudget until either a CUDA failure occurs or a generous ceiling survives cleanly.
 *
 * <p>Motivated by the two extreme "long launch" outliers found in logs/eternity_solver-*.log
 * (2026-08-16 10:25:38, 751813 ms; 2026-08-18 05:16:15, 9857717 ms) -- both timestamp-correlated
 * with a confirmed Windows sleep/resume event (Event 42/107 in the System log, "Button or Lid"),
 * so neither is real evidence about the TDR ceiling: a launch already in flight when the machine
 * sleeps just resumes afterward, inflating its logged wall-clock time by however long the machine
 * was actually asleep. This harness instead measures deliberately, with the machine required to
 * stay awake throughout -- a sleep mid-run would silently reproduce the same confound.</p>
 *
 * <p>Uses {@code NUM_THREADS = 1024} to match current production ({@code BlackwoodGpuRunner}), no
 * seeding (irrelevant to raw single-launch wall-clock survival, and seed-replay adds its own
 * overhead that would muddy the timing). If a launch fails, the exception is logged and the process
 * exits -- a real TDR reset invalidates the CUDA context, so there is no point trying to continue in
 * the same process.</p>
 *
 * <p>'Harness' suffix keeps Surefire from collecting it; needs real CUDA hardware, and must not run
 * concurrently with another GPU process (contends for the same device and would skew both).</p>
 */
public class BlackwoodGpuTdrCeilingHarness {

    private static final String PIECES_PATH = "src/main/resources/JBlackwood_Pieces.txt";
    private static final int NUM_THREADS = 1024;
    private static final long START_STEP_BUDGET = 200_000L;
    /** Matches the existing production auto-tune's own doubling convention (BlackwoodGpuRunner). */
    private static final long GROWTH_FACTOR = 2L;
    /** A clean two minutes of active compute is >=3600x the "~2000ms default" folklore -- plenty. */
    private static final long STOP_AFTER_MILLIS = 120_000L;

    public static void main(String[] args) throws Exception {
        BlackwoodSolver solver = new BlackwoodSolver(999, null, 1, PIECES_PATH); // never saves
        solver.prepare();
        BwGpuTables.GpuTableSet tables = BwGpuTables.build(solver);

        BlackwoodGpuEngine engine = new BlackwoodGpuEngine();
        engine.uploadTables(tables);

        System.out.println("=== TDR ceiling probe ===");
        System.out.printf("%d threads, starting stepBudget=%d, doubling each launch, stopping after a clean %ds launch.%n",
                NUM_THREADS, START_STEP_BUDGET, STOP_AFTER_MILLIS / 1000);
        System.out.println("Assumes the machine stayed awake for the whole run -- a sleep/resume mid-launch");
        System.out.println("would inflate elapsed time the same way it did in the incidental log outliers.");
        System.out.println();
        System.out.printf("%12s %12s %10s  %s%n", "stepBudget", "elapsedMs", "result", "nodesTaken");

        long stepBudget = START_STEP_BUDGET;
        long launchCounter = 0;
        while (true) {
            int[] bestBoardOut = new int[256];
            long start = System.nanoTime();
            try {
                BlackwoodGpuEngine.GpuResult r = engine.runBlackwoodDfs(
                        System.nanoTime() ^ (launchCounter * 0x9E3779B97F4A7C15L),
                        stepBudget, NUM_THREADS, 0, bestBoardOut);
                long elapsedMs = (System.nanoTime() - start) / 1_000_000L;
                System.out.printf("%12d %12d %10s  %d%n", stepBudget, elapsedMs, "OK", r.nodesTaken());

                if (elapsedMs >= STOP_AFTER_MILLIS) {
                    System.out.println();
                    System.out.printf("Reached a clean %d ms launch with no failure -- stopping here.%n", elapsedMs);
                    System.out.println("No TDR kill observed up to this duration on this machine/driver.");
                    break;
                }
            } catch (Throwable t) {
                long elapsedMs = (System.nanoTime() - start) / 1_000_000L;
                System.out.printf("%12d %12d %10s%n", stepBudget, elapsedMs, "FAILED");
                System.out.println();
                System.out.println("First failure at stepBudget=" + stepBudget + ", elapsed=" + elapsedMs + "ms:");
                t.printStackTrace(System.out);
                System.out.println();
                System.out.println("That is the real ceiling on this machine/driver -- keep stepBudget comfortably below it.");
                break;
            }
            stepBudget *= GROWTH_FACTOR;
            launchCounter++;
        }
    }
}

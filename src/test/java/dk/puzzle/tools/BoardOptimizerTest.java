package dk.puzzle.tools;

import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Unit tests for BoardOptimizer.
 *
 * <p>BoardOptimizer is pure CLI glue: besides the {@code HINT_CELLS}
 * constant, every other line of the class lives inside {@code main(String[])}
 * — argv parsing (seconds/threads/--locked), reading an optional link file
 * from disk, and delegating straight to {@link dk.puzzle.ai.ConflictAnnealer#optimize}
 * (itself a real, time-budgeted simulated-annealing search, not something to
 * unit-test on a schedule). None of that is factored into a separately
 * callable, pure method, and per this project's rules main() is never
 * invoked from a test (it also does real file I/O via {@code Files.readString}
 * and would run a genuine multi-thread annealing search). The one piece of
 * real, deterministic, standalone logic in this file — the official hint
 * cell set used by {@code --locked} — is covered below.</p>
 */
class BoardOptimizerTest {

    @SuppressWarnings("unchecked")
    private Set<Integer> getHintCells() throws Exception {
        Field field = BoardOptimizer.class.getDeclaredField("HINT_CELLS");
        field.setAccessible(true);
        return (Set<Integer>) field.get(null);
    }

    @Test
    void testHintCellsMatchesTheDocumentedOfficialHintPositions() throws Exception {
        assertEquals(Set.of(135, 221, 45, 210, 34), getHintCells(),
                "HINT_CELLS must be exactly the center (135) + official hint positions (221/45/210/34)");
    }

}

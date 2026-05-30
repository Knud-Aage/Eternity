package dk.puzzle.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Holder for de bedste brætter fundet under søgningen.
 * Bruges af Phase 3 til at køre LNS på top-N brætter i stedet for kun ét.
 */
public class TopBoardRegistry {

    public static final int MAX_SIZE = 20;

    private final List<Entry> entries = new ArrayList<>();

    public synchronized void offer(int[] board, int score) {
        // Tjek om vi allerede har dette bræt (undgå dubletter med samme score)
        for (Entry e : entries) {
            if (e.score == score && Arrays.equals(e.board, board)) return;
        }

        entries.add(new Entry(Arrays.copyOf(board, 256), score));
        entries.sort(Comparator.comparingInt((Entry e) -> e.score).reversed());

        if (entries.size() > MAX_SIZE) {
            entries.remove(entries.size() - 1);
        }
    }

    /** Returnerer det bedste bræt, eller null hvis tom. */
    public synchronized int[] getBest() {
        if (entries.isEmpty()) return null;
        return Arrays.copyOf(entries.get(0).board, 256);
    }

    /** Returnerer alle top-brætter som en liste (kopi). */
    public synchronized List<int[]> getAll() {
        List<int[]> result = new ArrayList<>(entries.size());
        for (Entry e : entries) result.add(Arrays.copyOf(e.board, 256));
        return result;
    }

    /** Returnerer det næste bræt i round-robin rækkefølge til LNS-kørsel. */
    private int roundRobinIdx = 0;
    public synchronized int[] nextForRepair() {
        if (entries.isEmpty()) return null;
        int idx = roundRobinIdx % entries.size();
        roundRobinIdx++;
        return Arrays.copyOf(entries.get(idx).board, 256);
    }

    public synchronized int size() { return entries.size(); }

    public synchronized int bestScore() {
        if (entries.isEmpty()) return 0;
        return entries.getFirst().score;
    }

    /**
     * Exposes the registry for the CheckpointManager so it can be saved to disk.
     */
    public List<int[]> getRawRegistry() {
        return new ArrayList<>(this.entries.stream().map(e -> e.board).toList());
    }

    public static class Entry {
        public final int[] board;
        public final int score;
        Entry(int[] board, int score) { this.board = board; this.score = score; }
    }
}

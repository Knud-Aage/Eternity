package dk.puzzle.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class TopBoardRegistry {

    public static final int MAX_SIZE = 20;

    private final List<Entry> entries = new ArrayList<>();

    public synchronized void offer(int[] board, int score) {
        for (Entry e : entries) {
            if (e.score == score && Arrays.equals(e.board, board)) return;
        }

        entries.add(new Entry(Arrays.copyOf(board, 256), score));
        entries.sort(Comparator.comparingInt((Entry e) -> e.score).reversed());

        if (entries.size() > MAX_SIZE) {
            entries.remove(entries.size() - 1);
        }
    }

    public synchronized int[] getBest() {
        if (entries.isEmpty()) return null;
        return Arrays.copyOf(entries.get(0).board, 256);
    }

    public synchronized List<int[]> getAll() {
        List<int[]> result = new ArrayList<>(entries.size());
        for (Entry e : entries) result.add(Arrays.copyOf(e.board, 256));
        return result;
    }

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
    public synchronized List<int[]> getRawRegistry() {
        List<int[]> result = new ArrayList<>(entries.size());
        for (Entry e : entries) result.add(Arrays.copyOf(e.board, 256));
        return result;
    }

    public static class Entry {
        public final int[] board;
        public final int score;
        Entry(int[] board, int score) { this.board = board; this.score = score; }
    }
}

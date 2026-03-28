package dk.roleplay;

import java.io.*;
import java.util.Arrays;


public class CheckpointManager {
    private CheckpointManager() {
        /* This utility class should not be instantiated */
    }

    private static final String FILE = "checkpoint.txt";
    

    public static void save(int[] inventory, int[][] mainBoard) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(FILE))) {
            writer.println(Arrays.toString(inventory));
            for (int[] tile : mainBoard) {
                writer.println(Arrays.toString(tile));
            }
            System.out.println("Checkpoint saved.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static CheckpointData load() {
        File f = new File(FILE);
        if (!f.exists()) return null;
        try (BufferedReader reader = new BufferedReader(new FileReader(f))) {
            int[] inv = parseArray(reader.readLine());
            int[][] board = new int[16][16];
            for (int i = 0; i < 16; i++) {
                board[i] = parseArray(reader.readLine());
            }
            return new CheckpointData(inv, board);
        } catch (Exception e) {
            return null;
        }
    }

    private static int[] parseArray(String line) {
        if (line == null || line.equals("null")) return null;
        String[] parts = line.replace("[", "").replace("]", "").split(", ");
        int[] arr = new int[parts.length];
        for (int i = 0; i < parts.length; i++) arr[i] = Integer.parseInt(parts[i]);
        return arr;
    }

    public static class CheckpointData {
        public int[] inventory;
        public int[][] mainBoard;
        public CheckpointData(int[] inv, int[][] board) { this.inventory = inv; this.mainBoard = board; }
    }
}

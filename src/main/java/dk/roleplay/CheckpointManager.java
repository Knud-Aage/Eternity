package dk.roleplay;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;

public class CheckpointManager {
    private static final String FILE = "checkpoint.txt";

    public static void save(int[] inventory, int[][] mainBoard, int bestScore) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(FILE))) {
            writer.println(bestScore);
            writer.println(Arrays.toString(inventory));
            for (int[] tile : mainBoard) {
                writer.println(Arrays.toString(tile));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static CheckpointData load() {
        File f = new File(FILE);
        if (!f.exists()) {
            return null;
        }
        try (BufferedReader reader = new BufferedReader(new FileReader(f))) {
            int score = Integer.parseInt(reader.readLine());
            int[] inv = parseArray(reader.readLine());
            int[][] board = new int[16][16];
            for (
                    int i = 0;
                    i < 16;
                    i++
            ) {
                board[i] = parseArray(reader.readLine());
            }
            return new CheckpointData(inv, board, score);
        } catch (Exception e) {
            return null;
        }
    }

    private static int[] parseArray(String line) {
        if (line == null || line.equals("null")) {
            return null;
        }
        String[] pts = line.replace("[", "").replace("]", "").split(", ");
        int[] res = new int[pts.length];
        for (
                int i = 0;
                i < pts.length;
                i++
        ) {
            res[i] = Integer.parseInt(pts[i].trim());
        }
        return res;
    }

    public static class CheckpointData {
        public int[] inventory;
        public int[][] mainBoard;
        public int bestScore;

        public CheckpointData(int[] i, int[][] b, int s) {
            inventory = i;
            mainBoard = b;
            bestScore = s;
        }
    }
}

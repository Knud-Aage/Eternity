package dk.roleplay;

import java.io.*;
import java.util.Arrays;

public class CheckpointManager {
    private static final String FILE_NAME = "checkpoint.dat";

    public static synchronized void save(int[][] mainBoard) {
        try (PrintWriter out = new PrintWriter(new FileWriter(FILE_NAME))) {
            for (int i = 0; i < 16; i++) {
                if (mainBoard[i] == null) continue;
                out.print(i + ":");
                for (int p = 0; p < 16; p++) {
                    out.print(mainBoard[i][p] + (p < 15 ? "," : ""));
                }
                out.println();
            }
        } catch (IOException e) {
            System.err.println("Failed to save checkpoint: " + e.getMessage());
        }
    }

    public static int[][] load() {
        File file = new File(FILE_NAME);
        if (!file.exists() || file.length() == 0) return null;

        int[][] board = new int[16][];
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String[] parts = line.split(":");
                if (parts.length < 2) continue;

                int mIdx = Integer.parseInt(parts[0]);
                String[] piecesStr = parts[1].split(",");
                int[] macroTile = new int[16];
                for (int i = 0; i < 16; i++) {
                    macroTile[i] = Integer.parseInt(piecesStr[i]);
                }
                board[mIdx] = macroTile;
            }
            System.out.println(">>> Checkpoint found! Preparing to resume...");
            return board;
        } catch (Exception e) {
            System.err.println("Failed to load checkpoint. Starting fresh. Error: " + e.getMessage());
            return null;
        }
    }

    public static void clear() {
        File file = new File(FILE_NAME);
        if (file.exists()) {
            file.delete();
        }
    }
}

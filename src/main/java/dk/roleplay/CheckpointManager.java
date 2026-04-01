package dk.roleplay;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;

public class CheckpointManager {
    private static final String FILE_NAME = "checkpoint.dat";

    public static synchronized void save(int[][] mainBoard) {
        try (PrintWriter out = new PrintWriter(FILE_NAME)) {
            for (int i = 0; i < 16; i++) {
                if (mainBoard[i] == null) continue;
                out.print(i + ":");
                for (int p = 0; p < 16; p++) {
                    out.print(mainBoard[i][p] + (p < 15 ? "," : ""));
                }
                out.println();
            }
        } catch (Exception e) { System.err.println("Failed to save checkpoint."); }
    }

    public static int[][] load() {
        File file = new File(FILE_NAME);
        if (!file.exists()) return null;

        int[][] board = new int[16][];
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(":");
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
            System.err.println("Failed to load checkpoint. Starting fresh.");
            return null;
        }
    }
}
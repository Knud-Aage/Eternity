package dk.roleplay;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Handles saving and loading solver progress to a persistent file.
 * This allows the search to resume after a program restart.
 */
public class CheckpointManager {
    private static final String FILE_NAME = "checkpoint.dat";

    /**
     * Saves the current board state to a checkpoint file.
     *
     * @param mainBoard The 16x16 board state represented as 4x4 macro-tiles
     */
    public static synchronized void save(int[][] mainBoard) {
        try (PrintWriter out = new PrintWriter(new FileWriter(FILE_NAME))) {
            for (int i = 0; i < 16; i++) {
                if (mainBoard[i] == null) {
                    continue;
                }
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

    /**
     * Loads the last saved board state from the checkpoint file.
     *
     * @return The reconstructed board state, or null if no valid checkpoint exists
     */
    public static int[][] load() {
        File file = new File(FILE_NAME);
        if (!file.exists() || file.length() == 0) {
            return null;
        }

        int[][] board = new int[16][];
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    continue;
                }
                String[] parts = line.split(":");
                if (parts.length < 2) {
                    continue;
                }

                int mIdx = Integer.parseInt(parts[0]);
                String[] piecesStr = parts[1].split(",");
                int[] macroTile = new int[16];
                for (int i = 0; i < 16; i++) {
                    macroTile[i] = Integer.parseInt(piecesStr[i]);
                }
                board[mIdx] = macroTile;
            }
            String now = LocalDateTime.now()
                    .format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
            System.out.println(now + " >>> Checkpoint found! Preparing to resume...");
            return board;
        } catch (Exception e) {
            String now = LocalDateTime.now()
                    .format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
            System.err.println(now + " Failed to load checkpoint. Starting fresh. Error: " + e.getMessage());
            return null;
        }
    }

    /**
     * Deletes the checkpoint file from disk.
     */
    public static void clear() {
        File file = new File(FILE_NAME);
        if (file.exists()) {
            file.delete();
        }
    }
}

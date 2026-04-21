package dk.puzzle;

import java.io.*;

public class CheckpointManager {

    // 1. Save a High Score Record (Goes into the Strategy Folder)
    public static void saveRecordCheckpoint(int[][] board, int score, String profile) {
        File dir = new File(profile);
        if (!dir.exists()) {
            dir.mkdirs(); // Create the folder (e.g., "TYPEWRITER_LOCKED") if it doesn't exist
        }
        File file = new File(dir, "checkpoint_" + score + ".dat");
        saveToFile(board, file);
    }

    // 2. Save the ongoing Working State (Goes into the Main root folder)
    public static void saveWorkingState(int[][] board) {
        File file = new File("checkpoint.dat");
        saveToFile(board, file);
    }

    // 3. The Smart Loader (Checks Strategy folder first, then Main folder)
    public static int[][] loadSmartCheckpoint(String profile) {
        File dir = new File(profile);
        int highestScore = -1;
        File bestFile = null;

        // Step A: Scan the profile folder for any checkpoint > 209
        if (dir.exists() && dir.isDirectory()) {
            File[] files = dir.listFiles((d, name) -> name.startsWith("checkpoint_") && name.endsWith(".dat"));
            if (files != null) {
                for (File f : files) {
                    try {
                        String numStr = f.getName().replace("checkpoint_", "").replace(".dat", "");
                        int score = Integer.parseInt(numStr);
                        if (score > highestScore) {
                            highestScore = score;
                            bestFile = f;
                        }
                    } catch (NumberFormatException e) {
                        // Ignore files that don't match the exact naming convention
                    }
                }
            }
        }

        // Step B: Decide which file to load based on the > 209 rule
        if (highestScore > 209 && bestFile != null) {
            System.out.println(">>> [SMART LOAD] Found High-Score Checkpoint in '" + profile + "/' with " + highestScore + " pieces.");
            return loadFromFile(bestFile);
        }

        System.out.println(">>> [SMART LOAD] No checkpoint > 209 found in '" + profile + "/'. Falling back to main working state...");
        File mainFile = new File("checkpoint.dat");
        if (mainFile.exists()) {
            return loadFromFile(mainFile);
        }

        System.out.println(">>> [SMART LOAD] No checkpoints found at all. Starting from a blank slate.");
        return null;
    }

    // Helper: Standard Java Serialization writer
    private static void saveToFile(int[][] board, File file) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
            oos.writeObject(board);
        } catch (IOException e) {
            System.err.println("Failed to save checkpoint: " + e.getMessage());
        }
    }

    // Helper: Standard Java Serialization reader
    private static int[][] loadFromFile(File file) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
            return (int[][]) ois.readObject();
        } catch (Exception e) {
            System.err.println("Failed to load checkpoint " + file.getName() + ": " + e.getMessage());
            return null;
        }
    }
}
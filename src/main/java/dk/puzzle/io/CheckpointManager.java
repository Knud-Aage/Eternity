package dk.puzzle.io;

import java.io.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;


/**
 * Manages the persistence of puzzle board states to disk.
 *
 * <p>This class handles the serialization and deserialization of board configurations,
 * allowing the solver to resume from the most advanced state (highest score) found
 * in a specific profile directory.</p>
 */
public class CheckpointManager {

    private static final Logger logger = LogManager.getLogger(CheckpointManager.class);

    /**
     * Automatically identifies and loads the checkpoint with the highest piece count 
     * from a specific profile directory.
     *
     * <p>The method scans the specified folder for files ending in {@code .dat},
     * extracts the numeric score from the filename, and loads the file associated 
     * with the maximum score.</p>
     *
     * @param profileFolder The directory path containing the checkpoint files.
     * @return A 16x16 integer array representing the saved board state, or 
     *         {@code null} if no valid checkpoints are found.
     */
    public static int[][] loadSmartCheckpoint(String profileFolder) {
        java.io.File folder = new java.io.File(profileFolder);

        if (!folder.exists() || !folder.isDirectory()) {
            logger.info(">>> [SMART LOAD] Folder '" + profileFolder + "' doesn't exist yet. Start with an empty board.");
            return null;
        }

        java.io.File[] files = folder.listFiles((dir, name) -> name.toLowerCase().endsWith(".dat"));
        if (files == null || files.length == 0) {
            logger.info(">>> [SMART LOAD] No .dat checkpoints found in the folder '" + profileFolder + "'.");
            return null;
        }

        java.io.File bestFile = null;
        int maxScore = -1;

        for (java.io.File f : files) {
            String name = f.getName();

            String numbersOnly = name.replaceAll("[^0-9]", "");

            if (!numbersOnly.isEmpty()) {
                try {
                    int score = Integer.parseInt(numbersOnly);
                    if (score > maxScore) {
                        maxScore = score;
                        bestFile = f;
                    }
                } catch (NumberFormatException ignored) {}
            }
        }

        if (bestFile != null) {
            logger.info(">>> [SMART LOAD] Loads the highest score checkpoint: " + bestFile.getName() + " from folder: " + profileFolder);
            return loadBoardFromFile(bestFile);
        }

        return null;
    }

    /**
     * Saves the current board state to a serialized checkpoint file.
     *
     * <p>Files are stored in the specified directory using the naming convention
     * {@code checkpoint_[score].dat}.</p>
     *
     * @param board The 16x16 integer array representing the current puzzle board state.
     * @param score The number of pieces successfully placed on the board.
     * @param profileFolder The directory path where the checkpoint file will be saved.
     */
    public static void saveRecordCheckpoint(int[][] board, int score, String profileFolder) {
        java.io.File folder = new java.io.File(profileFolder);
        if (!folder.exists()) {
            folder.mkdirs();
        }

        java.io.File file = new java.io.File(folder, "checkpoint_" + score + ".dat");

        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
            oos.writeObject(board);
        } catch (IOException e) {
            System.err.println(">>> [FEJL] Cou: " + e.getMessage());
        }
    }

    private static int[][] loadBoardFromFile(java.io.File file) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
            return (int[][]) ois.readObject();
        } catch (Exception e) {
            System.err.println(">>> Error: Couldn't read the file " + file.getName() + ": " + e.getMessage());
            return null;
        }
    }
}
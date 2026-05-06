package dk.puzzle;

import java.io.*;

public class CheckpointManager {

    public static int[][] loadSmartCheckpoint(String profileFolder) {
        File folder = new File(profileFolder);

        if (!folder.exists() || !folder.isDirectory()) {
            System.out.println(">>> [SMART LOAD] Folder '" + profileFolder + "' doesn't exist yet. Start with an empty board.");
            return null;
        }

        File[] files = folder.listFiles((dir, name) -> name.toLowerCase().endsWith(".dat"));
        if (files == null || files.length == 0) {
            System.out.println(">>> [SMART LOAD] No .dat checkpoints found in the folder '" + profileFolder + "'.");
            return null;
        }

        File bestFile = null;
        int maxScore = -1;

        for (File f : files) {
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
            System.out.println(">>> [SMART LOAD] Loads the highest score checkpoint: " + bestFile.getName() + " from folder: " + profileFolder);
            return loadBoardFromFile(bestFile);
        }

        return null;
    }

    public static void saveRecordCheckpoint(int[][] board, int score, String profileFolder) {
        File folder = new File(profileFolder);
        if (!folder.exists()) {
            folder.mkdirs();
        }

        File file = new File(folder, "checkpoint_" + score + ".dat");

        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
            oos.writeObject(board);
        } catch (IOException e) {
            System.err.println(">>> [FEJL] Couldn't save the checkpoint: " + e.getMessage());
        }
    }

    private static int[][] loadBoardFromFile(File file) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
            return (int[][]) ois.readObject();
        } catch (Exception e) {
            System.err.println(">>> [FEJL] Couldn't read the file " + file.getName() + ": " + e.getMessage());
            return null;
        }
    }
}
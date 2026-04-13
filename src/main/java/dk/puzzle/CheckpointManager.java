package dk.puzzle;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

public class CheckpointManager {

    public static void save(int[][] board, String strategyName) {
        String filename = "checkpoint_" + strategyName + ".dat";
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename))) {
            out.writeObject(board);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static int[][] load(String strategyName) {
        String filename = "checkpoint_" + strategyName + ".dat";
        File file = new File(filename);
        if (!file.exists()) {
            return null;
        }
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(file))) {
            return (int[][]) in.readObject();
        } catch (Exception e) {
            System.out.println("Could not load " + filename + " - starting fresh.");
            return null;
        }
    }
}
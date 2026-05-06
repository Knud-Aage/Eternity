package dk.puzzle;

import java.io.*;

public class CheckpointManager {

    // ==========================================================
    // 1. SMART LOAD: Finder automatisk filen med det højeste tal i den valgte mappe
    // ==========================================================
    public static int[][] loadSmartCheckpoint(String profileFolder) {
        File folder = new File(profileFolder);

        // Tjek om mappen overhovedet findes (f.eks. "TYPEWRITER_LOCKED")
        if (!folder.exists() || !folder.isDirectory()) {
            System.out.println(">>> [SMART LOAD] Mappen '" + profileFolder + "' findes ikke endnu. Starter med et tomt bræt.");
            return null;
        }

        // Find alle .dat filer i mappen
        File[] files = folder.listFiles((dir, name) -> name.toLowerCase().endsWith(".dat"));
        if (files == null || files.length == 0) {
            System.out.println(">>> [SMART LOAD] Ingen .dat checkpoints fundet i '" + profileFolder + "'.");
            return null;
        }

        File bestFile = null;
        int maxScore = -1;

        // Scan filnavnene for at finde det højeste tal
        for (File f : files) {
            String name = f.getName();

            // MAGIC TRICK: Fjerner alle bogstaver og tegn, så kun tallene er tilbage
            // F.eks. "checkpoint_214.dat" -> "214"
            String numbersOnly = name.replaceAll("[^0-9]", "");

            if (!numbersOnly.isEmpty()) {
                try {
                    int score = Integer.parseInt(numbersOnly);
                    if (score > maxScore) {
                        maxScore = score;
                        bestFile = f;
                    }
                } catch (NumberFormatException ignored) {
                    // Ignorer filer, hvor tallet er for stort/mærkeligt
                }
            }
        }

        // Hvis vi fandt en fil med et tal, så indlæs den!
        if (bestFile != null) {
            System.out.println(">>> [SMART LOAD] Indlæser det største checkpoint: " + bestFile.getName() + " fra mappen " + profileFolder);
            return loadBoardFromFile(bestFile);
        }

        return null;
    }

    // ==========================================================
    // 2. GEM-FUNKTION: Opretter mappen automatisk og gemmer med scoren i navnet
    // ==========================================================
    public static void saveRecordCheckpoint(int[][] board, int score, String profileFolder) {
        File folder = new File(profileFolder);
        if (!folder.exists()) {
            folder.mkdirs(); // Opret mappen (f.eks. TYPEWRITER_LOCKED), hvis den mangler
        }

        File file = new File(folder, "checkpoint_" + score + ".dat");

        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
            oos.writeObject(board);
        } catch (IOException e) {
            System.err.println(">>> [FEJL] Kunne ikke gemme checkpoint: " + e.getMessage());
        }
    }

    // ==========================================================
    // 3. HJÆLPEFUNKTION: Læser den fysiske fil ind i et 2D array
    // ==========================================================
    private static int[][] loadBoardFromFile(File file) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
            return (int[][]) ois.readObject();
        } catch (Exception e) {
            System.err.println(">>> [FEJL] Kunne ikke læse filen " + file.getName() + ": " + e.getMessage());
            return null;
        }
    }
}
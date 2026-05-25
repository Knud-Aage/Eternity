package dk.puzzle.io;

import dk.puzzle.model.PieceInventory;

import java.io.PrintWriter;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;

import static dk.puzzle.core.Eternity.loadPieces;

public class BoardImporter {

    public static void main(String[] args) {
        // 1. Define the input and output filenames
        String inputFileName = "src\\main\\resources\\Raw_468_Board.txt";
        String outputFileName = "src\\main\\resources\\Test_Board_468.csv";

        try {
            // 2. Read the entire content of the raw text file into a String
            String rawBoardText = new String(Files.readAllBytes(Paths.get(inputFileName)));


            // 3. Initialize your piece inventory
            // Make sure your PieceInventory constructor or initialization loads your pieces!
            int[] basePieces = loadPieces();
            PieceInventory inventory = new PieceInventory(basePieces);


            // 4. Create the importer and run the conversion
            BoardImporter importer = new BoardImporter();
            importer.generateCsvFromRawText(rawBoardText, inventory, outputFileName);

            System.out.println("Conversion finished successfully!");

        } catch (IOException e) {
            System.err.println("Error reading the file: " + inputFileName);
            System.err.println("Make sure the file exists in your project's root folder.");
            e.printStackTrace();
        }
    }

    /**
     * Converts a raw Eternity II partial solution string into a standard CSV file.
     *
     * @param rawBoardText The 16x16 grid text (e.g., "4/1 56/2 ...")
     * @param inventory    Your PieceInventory to look up piece colors by ID
     * @param outputPath   The path to save the generated CSV file
     */
    public void generateCsvFromRawText(String rawBoardText, PieceInventory inventory, String outputPath) {
        try (PrintWriter writer = new PrintWriter(new File(outputPath))) {
            // Write standard CSV header
            writer.println("Row,Col,North,East,South,West");

            String[] lines = rawBoardText.trim().split("\\r?\\n");

            for (int row = 0; row < 16; row++) {
                if (row >= lines.length) break;

                // Split by one or more spaces
                String[] tokens = lines[row].trim().split("\\s+");

                for (int col = 0; col < 16; col++) {
                    if (col >= tokens.length) break;

                    String token = tokens[col];

                    // Handle empty/unsolved slots
                    if (token.equals("---/-")) {
                        continue; // Skip writing this slot to CSV
                    }

                    // Parse piece ID and rotation
                    String[] parts = token.split("/");
                    int physicalId = Integer.parseInt(parts[0]);
                    int rotation = Integer.parseInt(parts[1]);

                    // Look up the base colors for this piece
                    // (Using the method we discussed earlier)
                    int[] baseColors = inventory.getBaseColors(physicalId);

                    int n = baseColors[0];
                    int e = baseColors[1];
                    int s = baseColors[2];
                    int w = baseColors[3];

                    // Apply rotation clockwise
                    int rN = n, rE = e, rS = s, rW = w;
                    if (rotation == 1) { // 90 degrees clockwise
                        rN = w; rE = n; rS = e; rW = s;
                    } else if (rotation == 2) { // 180 degrees
                        rN = s; rE = w; rS = n; rW = e;
                    } else if (rotation == 3) { // 270 degrees clockwise
                        rN = e; rE = s; rS = w; rW = n;
                    }

                    // Write the translated piece to the CSV file
                    writer.printf("%d,%d,%d,%d,%d,%d%n", row, col, rN, rE, rS, rW);
                }
            }
            System.out.println("Successfully generated CSV file: " + outputPath);
        } catch (IOException e) {
            System.err.println("Error writing to the file: " + outputPath);
            e.printStackTrace();
        }
    }
}
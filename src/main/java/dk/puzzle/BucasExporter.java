package dk.puzzle;

import java.io.BufferedReader;
import java.io.FileReader;

public class BucasExporter {

    public static void main(String[] args) {
//        String filename = "records/SPIRAL/Record_209Pieces.csv";
        String filename = "records/TYPEWRITER_LOCKED/Record_210Pieces.csv";

        String[] board = new String[256];
        for (int i = 0; i < 256; i++) {
            board[i] = "0000";
        }

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line = br.readLine();
            if (line == null) {
                return;
            }

            boolean isMacroFormat = line.toLowerCase().contains("macro");

            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length >= 6) {
                    int pos1 = Integer.parseInt(parts[0].trim());
                    int pos2 = Integer.parseInt(parts[1].trim());

                    int absoluteIndex = 0;
                    if (isMacroFormat) {
                        int row = (pos1 / 4) * 4 + (pos2 / 4);
                        int col = (pos1 % 4) * 4 + (pos2 % 4);
                        absoluteIndex = row * 16 + col;
                    } else {
                        absoluteIndex = pos1 * 16 + pos2;
                    }

                    int n = Integer.parseInt(parts[2].trim());

                    if (n >= 0 && n <= 22) {
                        int e = Integer.parseInt(parts[3].trim());
                        int s = Integer.parseInt(parts[4].trim());
                        int w = Integer.parseInt(parts[5].trim());

                        String pieceStr = "" +
                                (char) ('a' + n) +
                                (char) ('a' + e) +
                                (char) ('a' + s) +
                                (char) ('a' + w);

                        board[absoluteIndex] = pieceStr;
                    }
                }
            }

            StringBuilder bucasString = new StringBuilder();
            for (int i = 0; i < 256; i++) {
                bucasString.append(board[i]);
            }

            String finalUrl = "https://e2.bucas.name/#puzzle=Joshua_Blackwood_470&board_w=16&board_h=16&board_edges="
                    + bucasString
                    + "&motifs_order=jblackwood";

            System.out.println("Finished. Press on this link to see the puzzle:");
            System.out.println("------------------------------------------------------");
            System.out.println(finalUrl);
            System.out.println("------------------------------------------------------");

        } catch (Exception e) {
            System.out.println("Couldn't find the file. Be sure that it is placed in the records folder.");
            e.printStackTrace();
        }
    }
}
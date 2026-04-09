package dk.roleplay;

import java.io.BufferedReader;
import java.io.FileReader;

public class BucasExporter {

    public static void main(String[] args) {
        // Skift navnet her, hvis din fil hedder noget andet!
        String filename = "records/Record_206Pieces.csv";
        
        // Vi bygger et "virtuelt bræt" med 256 faste pladser.
        // Vi fylder det med "0000" (tomme brikker) fra start.
        String[] board = new String[256];
        for (int i = 0; i < 256; i++) {
            board[i] = "0000";
        }
        
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line = br.readLine(); // Læs første linje (headeren)
            if (line == null) return;

            // Vi tjekker automatisk om din CSV bruger "MacroIndex" eller "Row/Col"
            boolean isMacroFormat = line.toLowerCase().contains("macro");
            
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length >= 6) {
                    int pos1 = Integer.parseInt(parts[0].trim());
                    int pos2 = Integer.parseInt(parts[1].trim());

                    // Udregn den præcise plads (0-255) ud fra koordinaterne
                    int absoluteIndex = 0;
                    if (isMacroFormat) {
                        int row = (pos1 / 4) * 4 + (pos2 / 4);
                        int col = (pos1 % 4) * 4 + (pos2 % 4);
                        absoluteIndex = row * 16 + col;
                    } else {
                        absoluteIndex = pos1 * 16 + pos2;
                    }

                    int n = Integer.parseInt(parts[2].trim());

                    // Vi sikrer, at brikken er gyldig. Tomme brikker (ofte -1 eller 255 i data) springes over.
                    if (n >= 0 && n <= 22) {
                        int e = Integer.parseInt(parts[3].trim());
                        int s = Integer.parseInt(parts[4].trim());
                        int w = Integer.parseInt(parts[5].trim());

                        // Oversæt farvekoderne til bogstaver og lås dem fast på den rigtige plads
                        String pieceStr = "" +
                                (char)('a' + n) +
                                (char)('a' + e) +
                                (char)('a' + s) +
                                (char)('a' + w);

                        board[absoluteIndex] = pieceStr;
                    }
                }
            }
            
            // Saml det færdige puslespil fra venstre mod højre
            StringBuilder bucasString = new StringBuilder();
            for (int i = 0; i < 256; i++) {
                bucasString.append(board[i]);
            }
            
            // Byg den fulde URL
            String finalUrl = "https://e2.bucas.name/#puzzle=Joshua_Blackwood_470&board_w=16&board_h=16&board_edges=" 
                              + bucasString.toString() 
                              + "&motifs_order=jblackwood";
            
            System.out.println("Færdig! Kopiér dette link og send det til din kollega:");
            System.out.println("------------------------------------------------------");
            System.out.println(finalUrl);
            System.out.println("------------------------------------------------------");
            
        } catch (Exception e) {
            System.out.println("Kunne ikke finde filen! Sørg for at den ligger i records-mappen.");
            e.printStackTrace();
        }
    }
}
package dk.roleplay;

import java.io.BufferedReader;
import java.io.FileReader;

public class BucasExporter {

    public static void main(String[] args) {
        // Skift navnet her, hvis din fil hedder noget andet!
        String filename = "records/Record_213Pieces.csv";
        
        StringBuilder bucasString = new StringBuilder();
        
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            boolean isHeader = true;
            int count = 0;
            
            while ((line = br.readLine()) != null) {
                // Spring den første linje (kolonne-navnene) over
                if (isHeader) { 
                    isHeader = false; 
                    continue; 
                }
                
                String[] parts = line.split(",");
                if (parts.length >= 6) {
                    int n = Integer.parseInt(parts[2].trim());
                    int e = Integer.parseInt(parts[3].trim());
                    int s = Integer.parseInt(parts[4].trim());
                    int w = Integer.parseInt(parts[5].trim());
                    
                    // Oversæt farvekoderne (0-22) til bogstaver (a-w)
                    bucasString.append((char)('a' + n));
                    bucasString.append((char)('a' + e));
                    bucasString.append((char)('a' + s));
                    bucasString.append((char)('a' + w));
                    count++;
                }
            }
            
            // Udfyld resten af de 256 felter med "0000" (tomme brikker)
            while (count < 256) {
                bucasString.append("0000");
                count++;
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
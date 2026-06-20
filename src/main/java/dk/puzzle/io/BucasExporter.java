package dk.puzzle.io;

import java.net.URL;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import dk.puzzle.model.PieceInventory;
import dk.puzzle.util.PieceUtils;

public class BucasExporter {

    /**
     * Helper class representing a placed piece with its 4 edge colors
     */
    static class Piece {
        String north, east, south, west;

        public Piece(String n, String e, String s, String w) {
            this.north = n;
            this.east = e;
            this.south = s;
            this.west = w;
        }
    }

    /**
     * Generates a Bucas Visualizer link for the KnudHansen puzzle configuration.
     * Based on the official Javascript parser:
     * - board_edges requires exactly 1024 contiguous lowercase letters (no commas).
     * - board_pieces requires exactly 768 contiguous digits (3 digits per piece, no commas).
     * - show_piecenumber=1 forces the piece overlay to render on page load.
     */
    public static String exportBoard(int[] board, PieceInventory inventory) {
        StringBuilder sb = new StringBuilder();

        // Use the KnudHansen puzzle ID and enable the piece numbers overlay
        sb.append("https://e2.bucas.name/#puzzle=KnudHansen&board_w=16&board_h=16&show_piecenumber=1");

        StringBuilder edgesSb = new StringBuilder();
        StringBuilder piecesSb = new StringBuilder();

        for (int i = 0; i < 256; i++) {
            int p = board[i];

            if (p == -1 || p == -2) {
                // Empty slots use color 0 ('a') and piece ID '000'
                edgesSb.append("aaaa");
                piecesSb.append("000");
            } else {
                int n = PieceUtils.getNorth(p);
                int e = PieceUtils.getEast(p);
                int s = PieceUtils.getSouth(p);
                int w = PieceUtils.getWest(p);

                // Map numeric colors (0-22) to alphabetical characters ('a'-'w')
                edgesSb.append((char) ('a' + n));
                edgesSb.append((char) ('a' + e));
                edgesSb.append((char) ('a' + s));
                edgesSb.append((char) ('a' + w));

                // Find the physical piece ID (1-based index)
                int physId = -1;
                for (int oi = 0; oi < 1024; oi++) {
                    if (inventory.allOrientations[oi] == p) {
                        physId = inventory.physicalMapping[oi] + 1; // Convert to 1-based index
                        break;
                    }
                }

                // Format the ID as exactly 3 digits (e.g., 45 becomes "045")
                piecesSb.append(String.format("%03d", physId));
            }
        }

        // Append both parameters without any commas
        sb.append("&board_edges=").append(edgesSb.toString());
        sb.append("&board_pieces=").append(piecesSb.toString());

        return sb.toString();
    }

    /**
     * Verifies a Bucas link by parsing the contiguous character strings
     * in exact accordance with the Bucas JS constraints.
     */
    public static void verifyBucasLink(String urlString) throws Exception {
        // Handle URL fragment strings identifier (#) safely converted to queries
        String standardUrl = urlString.replace("#", "?");
        URL url = new URL(standardUrl);
        Map<String, String> params = getQueryParams(url.getQuery());

        // Step 1: Parse Grid Matrix Metadata
        int width = Integer.parseInt(params.getOrDefault("board_w", "16"));
        int height = Integer.parseInt(params.getOrDefault("board_h", "16"));
        String edgesStr = params.get("board_edges");

        if (edgesStr == null || edgesStr.trim().isEmpty()) {
            throw new IllegalArgumentException("Missing 'board_edges' parameter in the link.");
        }

        int totalExpectedChars = width * height * 4;

        if (edgesStr.length() < totalExpectedChars) {
            throw new IllegalArgumentException("Insufficient data! Found " + edgesStr.length()
                    + " edge characters, but expected " + totalExpectedChars + " for a " + width + "x" + height + " board.");
        }

        // Step 2: Build Grid Matrix [Row][Column]
        Piece[][] board = new Piece[height][width];
        int charIdx = 0;
        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++) {
                String n = String.valueOf(edgesStr.charAt(charIdx++));
                String e = String.valueOf(edgesStr.charAt(charIdx++));
                String s = String.valueOf(edgesStr.charAt(charIdx++));
                String w = String.valueOf(edgesStr.charAt(charIdx++));
                board[r][c] = new Piece(n, e, s, w);
            }
        }

        System.out.println("--- Eternity II Board Setup parsed ---");
        System.out.printf("Dimensions: %d columns x %d rows\n", width, height);
        System.out.println("Processing integrity checks...\n");

        // Step 3: Evaluate All Constraints (Inner & Outer Frame Boundaries)
        int totalInternalEdges = 0;
        int matchedInternalEdges = 0;
        int outerBorderMismatches = 0;

        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++) {
                Piece current = board[r][c];

                // Check North Boundary
                if (r == 0) {
                    if (!current.north.equals("a")) outerBorderMismatches++; // 'a' corresponds to color 0 (grey border)
                } else {
                    totalInternalEdges++;
                    if (current.north.equals(board[r - 1][c].south)) {
                        matchedInternalEdges++;
                    }
                }

                // Check East Boundary
                if (c == width - 1) {
                    if (!current.east.equals("a")) outerBorderMismatches++;
                } else {
                    totalInternalEdges++;
                    if (current.east.equals(board[r][c + 1].west)) {
                        matchedInternalEdges++;
                    }
                }

                // Check South Boundary
                if (r == height - 1) {
                    if (!current.south.equals("a")) outerBorderMismatches++;
                } else {
                    totalInternalEdges++;
                    if (current.south.equals(board[r + 1][c].north)) {
                        matchedInternalEdges++;
                    }
                }

                // Check West Boundary
                if (c == 0) {
                    if (!current.west.equals("a")) outerBorderMismatches++;
                } else {
                    totalInternalEdges++;
                    if (current.west.equals(board[r][c - 1].east)) {
                        matchedInternalEdges++;
                    }
                }
            }
        }

        // Step 4: Display Output Summary
        int brokenEdges = totalInternalEdges - matchedInternalEdges;
        System.out.println("--- Validation Results ---");
        System.out.println("Frame Border Violations (Should be 0): " + outerBorderMismatches);
        System.out.println("Total Internal Edge Interfaces Check: " + totalInternalEdges);
        System.out.println("Successfully Matched Edges: " + matchedInternalEdges);
        System.out.println("Broken / Mismatched Edges: " + brokenEdges);

        if (outerBorderMismatches == 0 && brokenEdges == 0) {
            System.out.println("\nSUCCESS: The provided link is a VALID full solution!");
        } else {
            System.out.println("\nFAILURE: Invalid placement matrix detected.");
        }
    }

    // Parses raw URL query parameters into a manageable Key-Value map
    private static Map<String, String> getQueryParams(String query) throws Exception {
        Map<String, String> paramsMap = new HashMap<>();
        if (query == null) return paramsMap;
        String[] pairs = query.split("&");
        for (String pair : pairs) {
            int idx = pair.indexOf("=");
            String key = idx > 0 ? URLDecoder.decode(pair.substring(0, idx), StandardCharsets.UTF_8.name()) : pair;
            String value = idx > 0 && pair.length() > idx + 1 ? URLDecoder.decode(pair.substring(idx + 1), StandardCharsets.UTF_8.name()) : "";
            paramsMap.put(key, value);
        }
        return paramsMap;
    }
}
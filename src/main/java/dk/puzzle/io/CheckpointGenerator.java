package dk.puzzle.io;

import dk.puzzle.model.PieceInventory;
import dk.puzzle.util.PieceUtils; // Husk at rette pakkenavnet, hvis det er anderledes

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static dk.puzzle.core.Eternity.loadPieces;

public class CheckpointGenerator {

    static int[] jbToUserColor = new int[23];
    static boolean[] userColorUsed = new boolean[23];

    public static void main(String[] args) {
//        String csvFile = "records\\TYPEWRITER_LOCKED\\Record_254Pieces.csv";
//        String profileFolder = "profiles";
        int currentScore = 254; // The imported board has 254 pieces placed
        String jbPiecesFile = "src\\main\\resources\\JBlackwood_Pieces.txt";
        String rawBoardFile = "src\\main\\resources\\Raw_468_Board.txt";
        String profileFolder = "profiles";

        try {
            System.out.println(">>> Starting Auto-Color-Mapper...");

            // 1. Load JBlackwood's piece definitions
            int[][] jbPieces = loadJBlackwoodPieces(jbPiecesFile);

            // 2. Load your own inventory pieces
            int[] basePieces = loadPieces();
            PieceInventory inventory = new PieceInventory(basePieces);
            int[][] userPieces = extractUserBasePieces(inventory.allOrientations);

            // 3. Crack the color mapping
            Arrays.fill(jbToUserColor, -1);
            jbToUserColor[0] = 0; // Gray is always 0
            userColorUsed[0] = true;

            if (crackColorCode(1, jbPieces, userPieces)) {
                System.out.println(">>> [SUCCESS] Color map cracked!");
            } else {
                System.err.println(">>> [ERROR] Could not match colors. Check your piece definitions.");
                return;
            }

            // 4. Build the board using your bit-packed pieces
            int[][] board = createTranslatedBoard(rawBoardFile, jbPieces, inventory.allOrientations);

            // 5. Save using your existing manager logic
            saveRecordCheckpoint(board, currentScore, profileFolder);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static boolean crackColorCode(int currentColorId, int[][] jbPieces, int[][] userPieces) {
        if (currentColorId > 22) return true;

        for (int userColorGuess = 1; userColorGuess <= 22; userColorGuess++) {
            if (userColorUsed[userColorGuess]) continue;

            jbToUserColor[currentColorId] = userColorGuess;
            userColorUsed[userColorGuess] = true;

            if (isValidPartialMap(jbPieces, userPieces) && crackColorCode(currentColorId + 1, jbPieces, userPieces)) {
                return true;
            }

            jbToUserColor[currentColorId] = -1;
            userColorUsed[userColorGuess] = false;
        }
        return false;
    }

    private static boolean isValidPartialMap(int[][] jbPieces, int[][] userPieces) {
        for (int[] jb : jbPieces) {
            int cN = jbToUserColor[jb[0]];
            int cE = jbToUserColor[jb[1]];
            int cS = jbToUserColor[jb[2]];
            int cW = jbToUserColor[jb[3]];

            if (cN != -1 && cE != -1 && cS != -1 && cW != -1) {
                if (!pieceExistsInUserSet(cN, cE, cS, cW, userPieces)) return false;
            }
        }
        return true;
    }

    private static boolean pieceExistsInUserSet(int n, int e, int s, int w, int[][] userPieces) {
        for (int[] up : userPieces) {
            if (up[0] == n && up[1] == e && up[2] == s && up[3] == w) return true;
            if (up[3] == n && up[0] == e && up[1] == s && up[2] == w) return true;
            if (up[2] == n && up[3] == e && up[0] == s && up[1] == w) return true;
            if (up[1] == n && up[2] == e && up[3] == s && up[0] == w) return true;
        }
        return false;
    }

    private static int[][] createTranslatedBoard(String rawBoardFile, int[][] jbPieces, int[] allOrientations) throws IOException {
        int[][] board = new int[16][16];
        for (int r = 0; r < 16; r++) Arrays.fill(board[r], -1);

        String rawBoard = new String(Files.readAllBytes(Paths.get(rawBoardFile)));
        String[] lines = rawBoard.trim().split("\\r?\\n");

        for (int row = 0; row < 16; row++) {
            if (row >= lines.length) break;
            String[] tokens = lines[row].trim().split("\\s+");

            for (int col = 0; col < 16; col++) {
                if (col >= tokens.length) break;
                String token = tokens[col];
                if (token.equals("---/-")) continue;

                String[] parts = token.split("/");
                int physicalId = Integer.parseInt(parts[0]);
                int rotation = Integer.parseInt(parts[1]);

                int[] jbColors = jbPieces[physicalId - 1];
                int n = jbToUserColor[jbColors[0]];
                int e = jbToUserColor[jbColors[1]];
                int s = jbToUserColor[jbColors[2]];
                int w = jbToUserColor[jbColors[3]];

                int rN = n, rE = e, rS = s, rW = w;
                if (rotation == 1) { rN = w; rE = n; rS = e; rW = s; }
                else if (rotation == 2) { rN = s; rE = w; rS = n; rW = e; }
                else if (rotation == 3) { rN = e; rE = s; rS = w; rW = n; }

                int matchedPiece = findPieceByColors(allOrientations, rN, rE, rS, rW);
                if (matchedPiece != -1) board[row][col] = matchedPiece;
            }
        }
        return board;
    }

    private static int[][] extractUserBasePieces(int[] allOrientations) {
        int[][] bases = new int[256][4];
        for (int i = 0; i < 256; i++) {
            int p = allOrientations[i * 4];
            bases[i][0] = PieceUtils.getNorth(p);
            bases[i][1] = PieceUtils.getEast(p);
            bases[i][2] = PieceUtils.getSouth(p);
            bases[i][3] = PieceUtils.getWest(p);
        }
        return bases;
    }

    private static int[][] loadJBlackwoodPieces(String filepath) throws IOException {
        int[][] pieces = new int[256][4];
        List<String> lines = Files.readAllLines(Paths.get(filepath));
        Pattern p = Pattern.compile("PieceNumber\\s*=\\s*(\\d+).*?TopSide\\s*=\\s*(\\d+).*?RightSide\\s*=\\s*(\\d+).*?BottomSide\\s*=\\s*(\\d+).*?LeftSide\\s*=\\s*(\\d+)");

        for (String line : lines) {
            Matcher m = p.matcher(line);
            if (m.find()) {
                int id = Integer.parseInt(m.group(1));
                pieces[id - 1][0] = Integer.parseInt(m.group(2));
                pieces[id - 1][1] = Integer.parseInt(m.group(3));
                pieces[id - 1][2] = Integer.parseInt(m.group(4));
                pieces[id - 1][3] = Integer.parseInt(m.group(5));
            }
        }
        return pieces;
    }

    private static int findPieceByColors(int[] allOrientations, int n, int e, int s, int w) {
        for (int piece : allOrientations) {
            if (PieceUtils.getNorth(piece) == n && PieceUtils.getEast(piece) == e &&
                PieceUtils.getSouth(piece) == s && PieceUtils.getWest(piece) == w) return piece;
        }
        return -1;
    }

    public static void saveRecordCheckpoint(int[][] board, int score, String profileFolder) {
        java.io.File folder = new java.io.File(profileFolder);
        if (!folder.exists()) folder.mkdirs();
        java.io.File file = new java.io.File(folder, "checkpoint_" + score + ".dat");
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
            oos.writeObject(board);
            System.out.println(">>> [SUCCESS] Saved 2D checkpoint to: " + file.getAbsolutePath());
        } catch (IOException e) {
            System.err.println(">>> [ERROR] Could not save checkpoint: " + e.getMessage());
        }
    }
}
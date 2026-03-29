package dk.roleplay;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class TabuSearchLayer {
    private static final int TABU_LIMIT = 200;
    private final List<String> tabuList = new LinkedList<>();

    public void run(int[][] mainBoard, boolean[] used, int[] inventory, int iterations) {
        Random rnd = new Random();
        for (
                int i = 0;
                i < iterations;
                i++
        ) {
            int m1 = rnd.nextInt(16), m2 = rnd.nextInt(16);
            int p1 = rnd.nextInt(16), p2 = rnd.nextInt(16);

            if (mainBoard[m1] == null || mainBoard[m2] == null) {
                continue;
            }

            String move = m1 + "-" + p1 + ":" + m2 + "-" + p2;
            if (tabuList.contains(move)) {
                continue;
            }

            int scoreBefore = evaluate(mainBoard);

            int temp = mainBoard[m1][p1];
            mainBoard[m1][p1] = mainBoard[m2][p2];
            mainBoard[m2][p2] = temp;

            int scoreAfter = evaluate(mainBoard);

            if (scoreAfter < scoreBefore) {
                // Revert
                mainBoard[m2][p2] = mainBoard[m1][p1];
                mainBoard[m1][p1] = temp;
            } else {
                tabuList.add(move);
                if (tabuList.size() > TABU_LIMIT) {
                    tabuList.remove(0);
                }
            }
        }
    }

    private int evaluate(int[][] board) {
        int score = 0;
        // Internal horizontal and vertical edge match evaluation
        for (
                int m = 0;
                m < 16;
                m++
        ) {
            if (board[m] == null) {
                continue;
            }
            for (
                    int p = 0;
                    p < 16;
                    p++
            ) {
                int r = p / 4, c = p % 4;
                int globalRow = (m / 4) * 4 + r;
                int globalCol = (m % 4) * 4 + c;
                // Neighbor checks (similar to canPlace logic)
                if (globalCol < 15) {
                    int nm = (globalRow / 4) * 4 + ((globalCol + 1) / 4);
                    int np = (globalRow % 4) * 4 + ((globalCol + 1) % 4);
                    if (board[nm] != null && PieceUtils.getEast(board[m][p]) == PieceUtils.getWest(board[nm][np])) {
                        score++;
                    }
                }
                if (globalRow < 15) {
                    int nm = (((globalRow + 1) / 4) * 4) + (globalCol / 4);
                    int np = (((globalRow + 1) % 4) * 4) + (globalCol % 4);
                    if (board[nm] != null && PieceUtils.getSouth(board[m][p]) == PieceUtils.getNorth(board[nm][np])) {
                        score++;
                    }
                }
            }
        }
        return score;
    }
}

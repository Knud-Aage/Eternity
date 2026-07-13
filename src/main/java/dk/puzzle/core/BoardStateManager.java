package dk.puzzle.core;

import java.util.Arrays;

public class BoardStateManager {
    private final int[] flatResumeBoard = new int[256];
    private final int[] globalBestBoard = new int[256];
    private int absoluteHighScore = 0;

    public BoardStateManager() {
        Arrays.fill(globalBestBoard, -1);
        Arrays.fill(flatResumeBoard, -1);
    }

    public synchronized void updateGlobalBestIfHigher(int[] newBoard, int score) {
        if (score > absoluteHighScore) {
            this.absoluteHighScore = score;
            System.arraycopy(newBoard, 0, this.globalBestBoard, 0, 256);
        }
    }

    public int[] getFlatResumeBoardCopy() {
        int[] copy = new int[256];
        System.arraycopy(flatResumeBoard, 0, copy, 0, 256);
        return copy;
    }

    public int[] getGlobalBestBoardCopy() {
        int[] copy = new int[256];
        System.arraycopy(globalBestBoard, 0, copy, 0, 256);
        return copy;
    }

    public int getAbsoluteHighScore() {
        return absoluteHighScore;
    }

    public void setAbsoluteHighScore(int score) {
        this.absoluteHighScore = score;
    }

}

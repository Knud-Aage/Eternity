package dk.puzzle;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class SurgeonHeuristics {
    private final boolean lockCenter;
    private double targetedHolesPercentage;

    public SurgeonHeuristics(boolean lockCenter, double targetedHolesPercentage) {
        this.lockCenter = lockCenter;
        this.targetedHolesPercentage = targetedHolesPercentage;
    }

    public void setTargetedHolesPercentage(double percentage) {
        this.targetedHolesPercentage = percentage;
    }

    public List<int[]> punchHoles(int[] bestBoard, int numClones, int numHoles, int[] tabuTenure, int currentIteration, int currentHighScore, int[] buildOrder) {
        List<int[]> swissCheeseBoards = new ArrayList<>(numClones);
        Random rnd = new Random();
        int[] placedIndices = new int[256];
        int placedCount = 0;

        for (int i = 0; i < 256; i++) {
            if (bestBoard[i] != -1 && bestBoard[i] != -2) {
                if (lockCenter && i == 135) continue;
                if (tabuTenure[i] > currentIteration) continue;
                placedIndices[placedCount++] = i;
            }
        }

        int actualHoles = Math.min(numHoles, placedCount);
        int[] conflicts = scoreConflicts(bestBoard);

        Integer[] sortedByConflict = new Integer[placedCount];
        for (int i = 0; i < placedCount; i++) sortedByConflict[i] = placedIndices[i];
        Arrays.sort(sortedByConflict, (a, b) -> Integer.compare(conflicts[b], conflicts[a]));

        int hotZoneSize = Math.max(actualHoles, placedCount / 4);
        int[] hotZone = new int[hotZoneSize];
        for (int i = 0; i < hotZoneSize; i++) hotZone[i] = sortedByConflict[i];

        int targetedHoles = (int) Math.round(actualHoles * targetedHolesPercentage);
        int randomHoles = actualHoles - targetedHoles;

        for (int clone = 0; clone < numClones; clone++) {
            int[] clonedBoard = new int[256];
            System.arraycopy(bestBoard, 0, clonedBoard, 0, 256);
            boolean[] punched = new boolean[256];

            int hotPicked = 0;
            int[] hotShuffled = hotZone.clone();
            for (int i = 0; i < hotShuffled.length && hotPicked < targetedHoles; i++) {
                int swapIdx = i + rnd.nextInt(hotShuffled.length - i);
                int tmp = hotShuffled[i];
                hotShuffled[i] = hotShuffled[swapIdx];
                hotShuffled[swapIdx] = tmp;
                int holeIdx = hotShuffled[i];
                if (!punched[holeIdx]) {
                    clonedBoard[holeIdx] = -2;
                    punched[holeIdx] = true;
                    hotPicked++;
                }
            }

            int randPicked = 0;
            int[] allShuffled = Arrays.copyOf(placedIndices, placedCount);
            for (int i = 0; i < allShuffled.length && randPicked < randomHoles; i++) {
                int swapIdx = i + rnd.nextInt(allShuffled.length - i);
                int tmp = allShuffled[i];
                allShuffled[i] = allShuffled[swapIdx];
                allShuffled[swapIdx] = tmp;
                int holeIdx = allShuffled[i];
                if (!punched[holeIdx]) {
                    clonedBoard[holeIdx] = -2;
                    punched[holeIdx] = true;
                    randPicked++;
                }
            }

            if (currentHighScore < 256) clonedBoard[buildOrder[currentHighScore]] = -2;
            swissCheeseBoards.add(clonedBoard);
        }
        return swissCheeseBoards;
    }

    private int[] scoreConflicts(int[] sourceBoard) {
        int[] conflicts = new int[256];
        for (int idx = 0; idx < 256; idx++) {
            int p = sourceBoard[idx];
            if (p == -1 || p == -2 || (lockCenter && idx == 135)) continue;

            int row = idx / 16;
            int col = idx % 16;
            int score = 0;

            if (row > 0 && sourceBoard[idx - 16] > 0 && PieceUtils.getSouth(sourceBoard[idx - 16]) != PieceUtils.getNorth(p)) score++;
            if (row < 15 && sourceBoard[idx + 16] > 0 && PieceUtils.getNorth(sourceBoard[idx + 16]) != PieceUtils.getSouth(p)) score++;
            if (col > 0 && sourceBoard[idx - 1] > 0 && PieceUtils.getEast(sourceBoard[idx - 1]) != PieceUtils.getWest(p)) score++;
            if (col < 15 && sourceBoard[idx + 1] > 0 && PieceUtils.getWest(sourceBoard[idx + 1]) != PieceUtils.getEast(p)) score++;

            conflicts[idx] = score;
        }
        return conflicts;
    }
}
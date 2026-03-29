package dk.roleplay;

public class PieceUtils {
    public static final int BORDER_COLOR = 0;
    public static final int WILDCARD = 0xFF;

    public static int pack(int n, int e, int s, int w) {
        return ((n & 0xFF) << 24) | ((e & 0xFF) << 16) | ((s & 0xFF) << 8) | (w & 0xFF);
    }

    public static int getNorth(int p) {
        return (p >>> 24) & 0xFF;
    }

    public static int getEast(int p) {
        return (p >>> 16) & 0xFF;
    }

    public static int getSouth(int p) {
        return (p >>> 8) & 0xFF;
    }

    public static int getWest(int p) {
        return p & 0xFF;
    }

    public static int rotate(int p) {
        // N->E, E->S, S->W, W->N (Clockwise)
        int n = (p >>> 24) & 0xFF;
        int e = (p >>> 16) & 0xFF;
        int s = (p >>> 8) & 0xFF;
        int w = p & 0xFF;
        return pack(w, n, e, s);
    }

    public static int getBorderCount(int p) {
        int count = 0;
        if (getNorth(p) == BORDER_COLOR) {
            count++;
        }
        if (getEast(p) == BORDER_COLOR) {
            count++;
        }
        if (getSouth(p) == BORDER_COLOR) {
            count++;
        }
        if (getWest(p) == BORDER_COLOR) {
            count++;
        }
        return count;
    }
}

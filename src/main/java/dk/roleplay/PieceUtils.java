package dk.roleplay;

public class PieceUtils {
    public static final int WILDCARD = 0xFF;

    public static int pack(int n, int e, int s, int w) {
        return ((n & 0xFF) << 24) | ((e & 0xFF) << 16) | ((s & 0xFF) << 8) | (w & 0xFF);
    }

    public static int getNorth(int p) { return (p >>> 24) & 0xFF; }
    public static int getEast(int p)  { return (p >>> 16) & 0xFF; }
    public static int getSouth(int p) { return (p >>> 8) & 0xFF; }
    public static int getWest(int p)  { return p & 0xFF; }

    public static int rotate(int p) {
        // North -> East, East -> South, South -> West, West -> North
        return pack(getWest(p), getNorth(p), getEast(p), getSouth(p));
    }
}

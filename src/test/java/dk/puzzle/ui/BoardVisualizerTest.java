package dk.puzzle.ui;

import org.junit.jupiter.api.Test;
import sun.misc.Unsafe;

import java.awt.Color;
import java.awt.geom.Area;
import java.lang.reflect.Field;
import java.lang.reflect.Method;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for BoardVisualizer's hardware/display-independent logic.
 *
 * <p>BoardVisualizer is a {@code JPanel} whose job is almost entirely rendering: the
 * constructor loads and pre-rotates pattern images from disk ({@code loadImages},
 * {@code rotateImage}), and {@code paintComponent}/{@code drawPiece}/
 * {@code drawPatternTriangle} paint the live board into a real {@code Graphics2D}
 * context supplied by Swing, reading {@code getWidth()}/{@code getHeight()} from the
 * component's on-screen bounds. None of that is meaningfully unit-testable: it either
 * depends on files under an "Assets" directory relative to the process's working
 * directory, or on component geometry that only exists once a real (non-headless)
 * container has laid the panel out, or produces pixel output whose exact appearance is
 * an implementation/rendering detail rather than a business rule. Per this task's
 * guidance, off-screen rendering captures were deliberately not forced here for the
 * same reason {@code GpuEngineTest} leaves hardware-only GPU paths untested (see that
 * class's javadoc) -- the cost/flakiness is not justified by the value of pinning down
 * exact pixels.</p>
 *
 * <p>Two private methods, however, are pure and deterministic with no Swing/AWT display
 * coupling and no file I/O:</p>
 * <ul>
 *   <li>{@code getFallbackColor(int)} is a plain switch-based color-mapping function --
 *   it takes a pattern id and returns a {@link Color} constant. Constructing a
 *   {@code Color} does not touch any display device, so this is safe and valuable to
 *   test directly.</li>
 *   <li>{@code rebuildClipCache(int)} computes four {@link java.awt.Polygon}/{@link Area}
 *   triangles (North/East/South/West) purely from the integer tile size. Building
 *   {@code Polygon}/{@code Area} geometry and querying {@code Area.contains(x, y)} is
 *   pure math -- it never allocates a graphics device or peer -- so the resulting
 *   triangle shapes can be verified deterministically.</li>
 * </ul>
 *
 * <p>Both methods are private and are only ever invoked from rendering code, so -- exactly
 * as {@code GpuEngineTest} does for {@code GpuEngine} -- instances are allocated via
 * {@link Unsafe#allocateInstance}, which skips the constructor (and therefore its file
 * I/O and Swing setup) entirely. Neither method under test reads the {@code board} or
 * {@code rotatedImages} fields, so an uninitialized instance is safe to exercise them
 * on.</p>
 */
class BoardVisualizerTest {

    private BoardVisualizer newUninitializedVisualizer() throws Exception {
        Field theUnsafe = Unsafe.class.getDeclaredField("theUnsafe");
        theUnsafe.setAccessible(true);
        Unsafe unsafe = (Unsafe) theUnsafe.get(null);
        return (BoardVisualizer) unsafe.allocateInstance(BoardVisualizer.class);
    }

    private Color invokeGetFallbackColor(BoardVisualizer visualizer, int val) throws Exception {
        Method method = BoardVisualizer.class.getDeclaredMethod("getFallbackColor", int.class);
        method.setAccessible(true);
        return (Color) method.invoke(visualizer, val);
    }

    private void invokeRebuildClipCache(BoardVisualizer visualizer, int size) throws Exception {
        Method method = BoardVisualizer.class.getDeclaredMethod("rebuildClipCache", int.class);
        method.setAccessible(true);
        method.invoke(visualizer, size);
    }

    private int getCachedTileSize(BoardVisualizer visualizer) throws Exception {
        Field field = BoardVisualizer.class.getDeclaredField("cachedTileSize");
        field.setAccessible(true);
        return field.getInt(visualizer);
    }

    private Area[] getCachedClipAreas(BoardVisualizer visualizer) throws Exception {
        Field field = BoardVisualizer.class.getDeclaredField("cachedClipAreas");
        field.setAccessible(true);
        return (Area[]) field.get(visualizer);
    }

    @Test
    void testGetFallbackColorKnownSingleValueCases() throws Exception {
        BoardVisualizer visualizer = newUninitializedVisualizer();

        assertEquals(new Color(100, 105, 110), invokeGetFallbackColor(visualizer, 0),
                "Pattern id 0 must map to its documented fallback color");
        assertEquals(new Color(80, 100, 200), invokeGetFallbackColor(visualizer, 10),
                "Pattern id 10 must map to its documented fallback color");
        assertEquals(new Color(180, 60, 100), invokeGetFallbackColor(visualizer, 21),
                "Pattern id 21 must map to its documented fallback color");
    }

    @Test
    void testGetFallbackColorAliasedValuesShareSameColor() throws Exception {
        BoardVisualizer visualizer = newUninitializedVisualizer();

        Color expected = new Color(230, 130, 180);
        assertEquals(expected, invokeGetFallbackColor(visualizer, 2), "Pattern id 2 must use the shared color group");
        assertEquals(expected, invokeGetFallbackColor(visualizer, 6),
                "Pattern id 6 must alias to the same color as pattern id 2");
        assertEquals(expected, invokeGetFallbackColor(visualizer, 13),
                "Pattern id 13 must alias to the same color as pattern id 2");

        Color otherGroup = new Color(100, 200, 230);
        assertEquals(otherGroup, invokeGetFallbackColor(visualizer, 1));
        assertEquals(otherGroup, invokeGetFallbackColor(visualizer, 17),
                "Pattern id 17 must alias to the same color as pattern id 1");
    }

    @Test
    void testGetFallbackColorOutOfRangeReturnsDefault() throws Exception {
        BoardVisualizer visualizer = newUninitializedVisualizer();

        Color defaultColor = new Color(40, 40, 40);
        assertEquals(defaultColor, invokeGetFallbackColor(visualizer, 99),
                "Unknown pattern ids above the known range must fall back to the default color");
        assertEquals(defaultColor, invokeGetFallbackColor(visualizer, -5),
                "Negative/invalid pattern ids must fall back to the default color");
        assertEquals(defaultColor, invokeGetFallbackColor(visualizer, 23),
                "Pattern id just past the documented 1-22 range must fall back to the default color");
    }

    @Test
    void testRebuildClipCacheUpdatesCachedTileSize() throws Exception {
        BoardVisualizer visualizer = newUninitializedVisualizer();

        invokeRebuildClipCache(visualizer, 120);
        assertEquals(120, getCachedTileSize(visualizer), "cachedTileSize must be updated to the requested tile size");

        invokeRebuildClipCache(visualizer, 64);
        assertEquals(64, getCachedTileSize(visualizer), "cachedTileSize must reflect the most recent rebuild");
    }

    @Test
    void testRebuildClipCacheProducesFourDistinctDirectionalTriangles() throws Exception {
        BoardVisualizer visualizer = newUninitializedVisualizer();

        // size=120 keeps the four triangle centroids on exact integer coordinates,
        // well clear of any shared edge, so containment checks are unambiguous.
        invokeRebuildClipCache(visualizer, 120);
        Area[] areas = getCachedClipAreas(visualizer);

        assertEquals(4, areas.length, "Clip cache must contain exactly 4 triangles: North, East, South, West");

        double northX = 60, northY = 20;   // centroid of (0,0),(120,0),(60,60)
        double eastX = 100, eastY = 60;    // centroid of (120,0),(120,120),(60,60)
        double southX = 60, southY = 100;  // centroid of (0,120),(120,120),(60,60)
        double westX = 20, westY = 60;     // centroid of (0,0),(0,120),(60,60)

        Area north = areas[0];
        Area east = areas[1];
        Area south = areas[2];
        Area west = areas[3];

        assertTrue(north.contains(northX, northY), "North triangle must contain its own centroid");
        assertFalse(north.contains(eastX, eastY), "North triangle must not contain the East centroid");
        assertFalse(north.contains(southX, southY), "North triangle must not contain the South centroid");
        assertFalse(north.contains(westX, westY), "North triangle must not contain the West centroid");

        assertTrue(east.contains(eastX, eastY), "East triangle must contain its own centroid");
        assertFalse(east.contains(northX, northY), "East triangle must not contain the North centroid");
        assertFalse(east.contains(southX, southY), "East triangle must not contain the South centroid");
        assertFalse(east.contains(westX, westY), "East triangle must not contain the West centroid");

        assertTrue(south.contains(southX, southY), "South triangle must contain its own centroid");
        assertFalse(south.contains(northX, northY), "South triangle must not contain the North centroid");
        assertFalse(south.contains(eastX, eastY), "South triangle must not contain the East centroid");
        assertFalse(south.contains(westX, westY), "South triangle must not contain the West centroid");

        assertTrue(west.contains(westX, westY), "West triangle must contain its own centroid");
        assertFalse(west.contains(northX, northY), "West triangle must not contain the North centroid");
        assertFalse(west.contains(eastX, eastY), "West triangle must not contain the East centroid");
        assertFalse(west.contains(southX, southY), "West triangle must not contain the South centroid");
    }
}

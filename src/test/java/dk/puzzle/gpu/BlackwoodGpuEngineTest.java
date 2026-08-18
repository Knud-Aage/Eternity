package dk.puzzle.gpu;

import org.junit.jupiter.api.Test;
import sun.misc.Unsafe;

import java.lang.reflect.Field;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for BlackwoodGpuEngine's hardware-independent logic.
 *
 * <p>Same rationale as {@link GpuEngineTest}: the constructor performs real CUDA device
 * initialization, so it isn't unit-test territory. {@link Unsafe#allocateInstance} skips the
 * constructor entirely -- safe here only because {@code sharedCacheEnabled}'s getter/setter never
 * touch any of the (therefore still-null) device buffer fields.</p>
 */
class BlackwoodGpuEngineTest {

    private BlackwoodGpuEngine newUninitializedEngine() throws Exception {
        Field theUnsafe = Unsafe.class.getDeclaredField("theUnsafe");
        theUnsafe.setAccessible(true);
        Unsafe unsafe = (Unsafe) theUnsafe.get(null);
        return (BlackwoodGpuEngine) unsafe.allocateInstance(BlackwoodGpuEngine.class);
    }

    @Test
    void testSharedCacheDefaultsFalse() throws Exception {
        BlackwoodGpuEngine engine = newUninitializedEngine();

        assertFalse(engine.isSharedCacheEnabled(), "Must default false -- preserves existing behaviour until deliberately flipped");
    }

    @Test
    void testSharedCacheSetterRoundTrips() throws Exception {
        BlackwoodGpuEngine engine = newUninitializedEngine();

        engine.setSharedCacheEnabled(true);
        assertTrue(engine.isSharedCacheEnabled());

        engine.setSharedCacheEnabled(false);
        assertFalse(engine.isSharedCacheEnabled());
    }
}

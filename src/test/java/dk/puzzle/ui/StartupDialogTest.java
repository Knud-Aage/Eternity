package dk.puzzle.ui;

import org.junit.jupiter.api.Test;
import sun.misc.Unsafe;

import java.lang.reflect.Field;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for StartupDialog's exposed configuration getters.
 *
 * <p>StartupDialog extends JDialog. Its constructor builds JComboBox/JCheckBox/JButton
 * components, lays them out, and wires an ActionListener that copies the user's choices
 * into private fields when "Start Engine" is clicked. {@code JDialog} is a {@code Window}
 * subclass, and {@code Window}'s constructor calls into {@code GraphicsEnvironment} and
 * throws {@code HeadlessException} when no display is available, so the constructor
 * itself -- and everything reachable only through it (layout, the button-click
 * ActionListener, {@code setVisible}) -- cannot be safely exercised in this (potentially
 * headless) test environment. That is component construction/layout/event-wiring with no
 * extractable pure logic, so it is intentionally left untested here, mirroring the
 * precedent set in {@code GpuEngineTest}'s class javadoc for construction that is coupled
 * to environment/hardware state rather than to a testable abstraction seam.</p>
 *
 * <p>The five {@code is*()} getters, however, are plain boolean field reads with no Swing
 * coupling of their own: {@code dk.puzzle.core.Eternity.main()} calls them only after the
 * dialog has already closed, to read back whatever the (untested) button click recorded.
 * To verify those reads without ever running the constructor -- and therefore without
 * touching AWT/Swing at all -- instances are allocated via {@link Unsafe#allocateInstance},
 * exactly as {@code GpuEngineTest} does for {@code GpuEngine}, and the private backing
 * fields are set directly via reflection before asserting on the public getter.</p>
 */
class StartupDialogTest {

    private StartupDialog newUninitializedDialog() throws Exception {
        Field theUnsafe = Unsafe.class.getDeclaredField("theUnsafe");
        theUnsafe.setAccessible(true);
        Unsafe unsafe = (Unsafe) theUnsafe.get(null);
        return (StartupDialog) unsafe.allocateInstance(StartupDialog.class);
    }

    private void setBooleanField(StartupDialog dialog, String fieldName, boolean value) throws Exception {
        Field field = StartupDialog.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        field.setBoolean(dialog, value);
    }

    @Test
    void testIsStartClickedReflectsBackingField() throws Exception {
        StartupDialog dialog = newUninitializedDialog();

        setBooleanField(dialog, "startClicked", false);
        assertFalse(dialog.isStartClicked(), "isStartClicked must return false when startClicked field is false");

        setBooleanField(dialog, "startClicked", true);
        assertTrue(dialog.isStartClicked(), "isStartClicked must return true when startClicked field is true");
    }

    @Test
    void testIsUsePbpReflectsBackingField() throws Exception {
        StartupDialog dialog = newUninitializedDialog();

        setBooleanField(dialog, "usePbp", false);
        assertFalse(dialog.isUsePbp(), "isUsePbp must return false when usePbp field is false");

        setBooleanField(dialog, "usePbp", true);
        assertTrue(dialog.isUsePbp(), "isUsePbp must return true when usePbp field is true");
    }

    @Test
    void testIsUseGpuReflectsBackingField() throws Exception {
        StartupDialog dialog = newUninitializedDialog();

        setBooleanField(dialog, "useGpu", false);
        assertFalse(dialog.isUseGpu(), "isUseGpu must return false when useGpu field is false");

        setBooleanField(dialog, "useGpu", true);
        assertTrue(dialog.isUseGpu(), "isUseGpu must return true when useGpu field is true");
    }

    @Test
    void testIsUseSpiralReflectsBackingField() throws Exception {
        StartupDialog dialog = newUninitializedDialog();

        setBooleanField(dialog, "useSpiral", false);
        assertFalse(dialog.isUseSpiral(), "isUseSpiral must return false when useSpiral field is false");

        setBooleanField(dialog, "useSpiral", true);
        assertTrue(dialog.isUseSpiral(), "isUseSpiral must return true when useSpiral field is true");
    }

    @Test
    void testIsLockCenterReflectsBackingField() throws Exception {
        StartupDialog dialog = newUninitializedDialog();

        setBooleanField(dialog, "lockCenter", false);
        assertFalse(dialog.isLockCenter(), "isLockCenter must return false when lockCenter field is false");

        setBooleanField(dialog, "lockCenter", true);
        assertTrue(dialog.isLockCenter(), "isLockCenter must return true when lockCenter field is true");
    }
}

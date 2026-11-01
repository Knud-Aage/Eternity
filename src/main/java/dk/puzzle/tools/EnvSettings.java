package dk.puzzle.tools;

/** Environment-variable settings shared by the polish stages. */
final class EnvSettings {

    private EnvSettings() {
    }

    /** One malformed setting falls back to its own default and no longer discards the settings read after it. */
    static int envInt(String owner, String name, int fallback) {
        String v = System.getenv(name);
        if (v == null || v.isBlank()) return fallback;
        try {
            return Integer.parseInt(v.trim());
        } catch (NumberFormatException e) {
            System.err.println(owner + ": ignoring malformed " + name + "=" + v);
            return fallback;
        }
    }
}

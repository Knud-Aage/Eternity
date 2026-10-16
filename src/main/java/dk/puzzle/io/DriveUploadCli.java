package dk.puzzle.io;

import java.io.File;
import java.nio.file.Files;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * CLI entry point so the C# solver (Eternity-9-breaks) can trigger the same Drive upload as the
 * GPU/Java engines' own {@code uploadRecordToDrive} without reimplementing it cross-language --
 * shelled out to exactly like Util.cs already does for HoleSolver labelling.
 */
public class DriveUploadCli {
    public static void main(String[] args) throws Exception {
        String prefix = args[0];
        int conflicts = Integer.parseInt(args[1]);
        String source = args[2];
        String completedLink = args[3];

        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        File linkFile = File.createTempFile(prefix + "_", "_link.txt");
        Files.writeString(linkFile.toPath(), String.format(
                "Edge Conflicts: %d%nSource: %s%nTime: %s%n%s%n", conflicts, source, timestamp, completedLink));
        RecordManager.uploadToDrive(linkFile, "text/plain", "Blackwood");
        linkFile.delete();
    }
}

package dk.puzzle.io;

import com.google.api.services.drive.Drive;
import com.google.api.services.drive.model.File;
import com.google.api.services.drive.model.FileList;
import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link GoogleDriveConfig#getOrCreateFolder}.
 *
 * <p>{@code getDriveService()} performs a real OAuth2 authorization flow
 * (reads {@code /credentials.json}, spins up a local HTTP receiver on port
 * 8888, persists tokens to disk) and is never invoked here - there is no
 * safe, side-effect-free way to exercise it. {@code getOrCreateFolder}'s own
 * "find or create" branching, however, only depends on the injected
 * {@link Drive} client, so it is tested here against a fully Mockito-mocked
 * client. No network call is made anywhere in this file.</p>
 */
class GoogleDriveConfigTest {

    @Test
    void testGetOrCreateFolderReturnsExistingFolderIdWithoutCreatingANewOne() throws Exception {
        Drive driveService = mock(Drive.class);
        Drive.Files filesApi = mock(Drive.Files.class);
        Drive.Files.List listRequest = mock(Drive.Files.List.class);

        File existingFolder = new File();
        existingFolder.setId("existing-folder-id");
        FileList existingResult = new FileList();
        existingResult.setFiles(List.of(existingFolder));

        when(driveService.files()).thenReturn(filesApi);
        when(filesApi.list()).thenReturn(listRequest);
        when(listRequest.setQ(anyString())).thenReturn(listRequest);
        when(listRequest.setSpaces(anyString())).thenReturn(listRequest);
        when(listRequest.setFields(anyString())).thenReturn(listRequest);
        when(listRequest.execute()).thenReturn(existingResult);

        String folderId = GoogleDriveConfig.getOrCreateFolder(driveService, "MyProfile");

        assertEquals("existing-folder-id", folderId, "An existing folder's id must be returned as-is");
        verify(filesApi, never()).create(any(File.class));
    }

    @Test
    void testGetOrCreateFolderCreatesFolderWhenNoneExists() throws Exception {
        Drive driveService = mock(Drive.class);
        Drive.Files filesApi = mock(Drive.Files.class);
        Drive.Files.List listRequest = mock(Drive.Files.List.class);
        Drive.Files.Create createRequest = mock(Drive.Files.Create.class);

        FileList emptyResult = new FileList();
        emptyResult.setFiles(Collections.emptyList());

        File createdFolder = new File();
        createdFolder.setId("new-folder-id");

        when(driveService.files()).thenReturn(filesApi);
        when(filesApi.list()).thenReturn(listRequest);
        when(listRequest.setQ(anyString())).thenReturn(listRequest);
        when(listRequest.setSpaces(anyString())).thenReturn(listRequest);
        when(listRequest.setFields(anyString())).thenReturn(listRequest);
        when(listRequest.execute()).thenReturn(emptyResult);

        when(filesApi.create(any(File.class))).thenReturn(createRequest);
        when(createRequest.setFields(anyString())).thenReturn(createRequest);
        when(createRequest.execute()).thenReturn(createdFolder);

        String folderId = GoogleDriveConfig.getOrCreateFolder(driveService, "NewProfile");

        assertEquals("new-folder-id", folderId, "A newly created folder's id must be returned");
        verify(filesApi).create(argThat(f ->
                "NewProfile".equals(f.getName()) && "application/vnd.google-apps.folder".equals(f.getMimeType())));
    }

    @Test
    void testGetOrCreateFolderCreatesFolderWhenSearchResultHasNullFilesList() throws Exception {
        // FileList.getFiles() defaults to null when never set - getOrCreateFolder must
        // treat that the same as "not found" (fall through to create) rather than NPE.
        Drive driveService = mock(Drive.class);
        Drive.Files filesApi = mock(Drive.Files.class);
        Drive.Files.List listRequest = mock(Drive.Files.List.class);
        Drive.Files.Create createRequest = mock(Drive.Files.Create.class);

        FileList nullFilesResult = new FileList();

        File createdFolder = new File();
        createdFolder.setId("created-after-null");

        when(driveService.files()).thenReturn(filesApi);
        when(filesApi.list()).thenReturn(listRequest);
        when(listRequest.setQ(anyString())).thenReturn(listRequest);
        when(listRequest.setSpaces(anyString())).thenReturn(listRequest);
        when(listRequest.setFields(anyString())).thenReturn(listRequest);
        when(listRequest.execute()).thenReturn(nullFilesResult);

        when(filesApi.create(any(File.class))).thenReturn(createRequest);
        when(createRequest.setFields(anyString())).thenReturn(createRequest);
        when(createRequest.execute()).thenReturn(createdFolder);

        String folderId = GoogleDriveConfig.getOrCreateFolder(driveService, "AnotherProfile");

        assertEquals("created-after-null", folderId);
    }
}

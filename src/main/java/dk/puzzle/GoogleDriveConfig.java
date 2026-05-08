package dk.puzzle;

import com.google.api.client.auth.oauth2.Credential;
import com.google.api.client.extensions.java6.auth.oauth2.AuthorizationCodeInstalledApp;
import com.google.api.client.extensions.jetty.auth.oauth2.LocalServerReceiver;
import com.google.api.client.googleapis.auth.oauth2.GoogleAuthorizationCodeFlow;
import com.google.api.client.googleapis.auth.oauth2.GoogleClientSecrets;
import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport;
import com.google.api.client.http.javanet.NetHttpTransport;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.gson.GsonFactory;
import com.google.api.client.util.store.FileDataStoreFactory;
import com.google.api.services.drive.Drive;
import com.google.api.services.drive.DriveScopes;

import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Collections;
import java.util.List;

public class GoogleDriveConfig {
    private static final String APPLICATION_NAME = "Eternity II Solver";
    private static final JsonFactory JSON_FACTORY = GsonFactory.getDefaultInstance();

    // Mappen hvor programmet gemmer sin "husk-mig" login-nøgle
    private static final String TOKENS_DIRECTORY_PATH = "tokens";

    // Vi beder KUN om tilladelse til at se/redigere de filer, programmet selv har oprettet. (Sikkerhed!)
    private static final List<String> SCOPES = Collections.singletonList(DriveScopes.DRIVE_FILE);

    public static Drive getDriveService() throws Exception {
        final NetHttpTransport HTTP_TRANSPORT = GoogleNetHttpTransport.newTrustedTransport();

        // 1. Læs din credentials.json fil
        InputStream in = GoogleDriveConfig.class.getResourceAsStream("/credentials.json");
        if (in == null) {
            throw new FileNotFoundException(">>> [FEJL] Kunne ikke finde /credentials.json. Ligger den i src/main/resources?");
        }

        GoogleClientSecrets clientSecrets = GoogleClientSecrets.load(JSON_FACTORY, new InputStreamReader(in));

        // 2. Sæt sikkerheds-flowet op
        GoogleAuthorizationCodeFlow flow = new GoogleAuthorizationCodeFlow.Builder(
                HTTP_TRANSPORT, JSON_FACTORY, clientSecrets, SCOPES)
                .setDataStoreFactory(new FileDataStoreFactory(new java.io.File(TOKENS_DIRECTORY_PATH)))
                .setAccessType("offline")
                .build();

        // 3. Første gang: Åbner browseren på port 8888 for login.
        // Næste gang: Læser den bare nøglen fra "tokens" mappen.
        LocalServerReceiver receiver = new LocalServerReceiver.Builder().setPort(8888).build();
        Credential credential = new AuthorizationCodeInstalledApp(flow, receiver).authorize("user");

        // 4. Byg og returner selve Drive-motoren!
        return new Drive.Builder(HTTP_TRANSPORT, JSON_FACTORY, credential)
                .setApplicationName(APPLICATION_NAME)
                .build();
    }
}
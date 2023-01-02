import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.stream.Collectors;

public class Sender {
    public static void main(String[] args) throws IOException, URISyntaxException, InterruptedException {
//        String path = "/Users/haoxuanwang/Downloads/remote_test.jpeg";
        String path = "/Users/haoxuanwang/Traffic-Light-Detection-Using-YOLOv3/preview_images/test3.jpeg";

        File file = new File(path);
        byte[] fileContent = Files.readAllBytes(file.toPath());
        String file_hex = bytesToHex(fileContent);
        HashMap<String, String> data = new HashMap();
        data.put("image", file_hex);
        String body = mapToJson(data);
        String url = "https://navigasion.eastus2.inference.ml.azure.com/score";
        String api_key = "kwTPY0LbPgzumjod4tCdy8WfPVFGPf06";
        HashMap<String, String> headers_map = new HashMap();
        String headers = mapToJson(headers_map);
        HttpRequest request = HttpRequest.newBuilder()
                .uri(new URI(url))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer "+ api_key)
                .header("azureml-model-deployment", "traffic-light")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();

        HttpClient client = HttpClient.newHttpClient();

        HttpResponse<String> response = client.send(request,
                HttpResponse.BodyHandlers.ofString());
        System.out.println(response.body());
    }

    public static String bytesToHex(byte[] in) {
        final StringBuilder builder = new StringBuilder();
        for(byte b : in) {
            builder.append(String.format("%02x", b));
        }
        return builder.toString();
    }

    public static String mapToJson(HashMap<String, String> map) {
        return map.entrySet().stream()
                .map(e -> String.format("\t\"%s\":\"%s\"", e.getKey(), e.getValue()))
                .collect(Collectors.joining(",\n", "{\n", "\n}"));
    }
}

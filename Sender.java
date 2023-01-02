import java.nio.file.Path;

public class Sender {
    public static void main(String[] args) {
      String path = "/Users/haoxuanwang/Downloads/remote_test.jpeg";
      File file = new File(path);
      byte[] fileContent = Files.readAllBytes(file.toPath());
      String file_hex = bytesToHex(fileContent);
      HashMap<String, String> data = new HashMap();

    }
    public static String bytesToHex(byte[] in) {
      final StringBuilder builder = new StringBuilder();
      for(byte b : in) {
          builder.append(String.format("%02x", b));
      }
      return builder.toString();
    }
    
}
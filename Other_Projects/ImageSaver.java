import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ImageSaver {

    private static final String USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3";
    private static final String BING_IMAGE_SEARCH_URL = "https://www.bing.com/images/search?q=";

    public static void main(String[] args) {
        String searchTerm = "kitten"; // replace with your search term
        int numImagesToDownload = 10; // replace with the number of images you want to download

        try {
            downloadImages(searchTerm, numImagesToDownload);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void downloadImages(String searchTerm, int numImagesToDownload) throws IOException {
        List<String> imageUrls = getImageUrls(searchTerm, numImagesToDownload);
        int i = 0;

        for (String imageUrl : imageUrls) {
            String fileName = imageUrl.substring(imageUrl.lastIndexOf('/') + 1);
            File outputFile = new File("image" + i + ".png");

            byte[] imgData = Jsoup.connect(imageUrl).userAgent(USER_AGENT).timeout(5000).ignoreContentType(true)
                    .execute().bodyAsBytes();
            FileOutputStream fos = new FileOutputStream(outputFile);
            fos.write(imgData);
            fos.close();
            i += 1;
        }
    }

    private static List<String> getImageUrls(String searchTerm, int numImagesToDownload) throws IOException {
        List<String> imageUrls = new ArrayList<>();

        String url = BING_IMAGE_SEARCH_URL + searchTerm;
        Document doc = Jsoup.connect(url).userAgent(USER_AGENT).get();

        Elements imgElements = doc.select("a[href]");

        for (Element imgElement : imgElements) {
            String imgUrl = imgElement.attr("href");
            if (imgUrl.contains("/images/search?view=detail")) {
                String newUrl = "https://bing.com" + imgUrl;
                Document newDoc = Jsoup.connect(newUrl).userAgent(USER_AGENT).get();
                Elements newImgElements = doc.select("img[src]");
                for (Element newImgElement : newImgElements) {
                    String newImgUrl = newImgElement.attr("src");
                    System.out.println(newImgUrl);
                    if (newImgUrl.startsWith("https://")) {
                        imageUrls.add(newImgUrl);

                    }
                }
            }
            if (imageUrls.size() >= numImagesToDownload) {
                break;
            }
        }

        return imageUrls;
    }
}
package gr.uoa.di.ai.file.parsers;

public interface FileParser {
    public static String RESOURCE = "http://ai.di.uoa.gr/da4dte/resource/";
    public static String URI = "http://ai.di.uoa.gr/da4dte/ontology/";

    public String parse(String path);
}

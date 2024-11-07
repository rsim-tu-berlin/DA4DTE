package gr.uoa.di.ai;

public class MarineRegionParser {
    public static void main( String[] args ){
        JsonMRParser parser = new JsonMRParser(args[0]);
        parser.parse();
    }
}

package gr.uoa.di.ai.gost;

import gr.uoa.di.ai.gost.aggregators.AreaAggFactory;
import org.apache.jena.sparql.expr.aggregate.AggregateRegistry;

import java.util.HashMap;
import java.util.HashSet;

public class GeoDictionary {

    static GeoDictionary geoDictionary = new GeoDictionary();
    HashMap<String,String> dictionary;
    HashSet<String> geoVocab;
    HashSet<String> wktVocab;
    HashMap<String,String> varMap;

    private GeoDictionary(){
        //Initialize the dictionary with the geosparql boolean
        //functions and their corresponding sparql predicates
        this.dictionary = initializeGeoFunctions();
        this.varMap = new HashMap<>();
        this.geoVocab = initializeGeoVocabulary();
        this.wktVocab = initializeWKTVocabulary();
        initializeAggregates();
    }

    private HashMap<String, String> initializeGeoFunctions(){
        HashMap<String,String> hashMap = new HashMap<>();
        hashMap.put("http://www.opengis.net/def/function/geosparql/sfWithin","http://www.opengis.net/ont/geosparql#sfWithin");
        hashMap.put("http://www.opengis.net/def/function/geosparql/sfTouches","http://www.opengis.net/ont/geosparql#sfTouches");
        hashMap.put("http://www.opengis.net/def/function/geosparql/sfIntersects","http://www.opengis.net/ont/geosparql#sfIntersects");
        hashMap.put("http://www.opengis.net/def/function/geosparql/sfCrosses","http://www.opengis.net/ont/geosparql#sfCrosses");
        hashMap.put("http://www.opengis.net/def/function/geosparql/sfContains","http://www.opengis.net/ont/geosparql#sfContains");
        hashMap.put("http://www.opengis.net/def/function/geosparql/sfOverlaps","http://www.opengis.net/ont/geosparql#sfOverlaps");
        hashMap.put("http://www.opengis.net/def/function/geosparql/sfEquals","http://www.opengis.net/ont/geosparql#sfEquals");
        hashMap.put("http://www.opengis.net/def/function/geosparql/sfCovers","http://www.opengis.net/ont/geosparql#sfCovers");

        hashMap.put("http://strdf.di.uoa.gr/ontology#within","http://www.opengis.net/ont/geosparql#sfWithin");
        hashMap.put("http://strdf.di.uoa.gr/ontology#touches","http://www.opengis.net/ont/geosparql#sfTouches");
        hashMap.put("http://strdf.di.uoa.gr/ontology#intersects","http://www.opengis.net/ont/geosparql#sfIntersects");
        hashMap.put("http://strdf.di.uoa.gr/ontology#crosses","http://www.opengis.net/ont/geosparql#sfCrosses");
        hashMap.put("http://strdf.di.uoa.gr/ontology#contains","http://www.opengis.net/ont/geosparql#sfContains");
        hashMap.put("http://strdf.di.uoa.gr/ontology#overlaps","http://www.opengis.net/ont/geosparql#sfOverlaps");
        hashMap.put("http://strdf.di.uoa.gr/ontology#equals","http://www.opengis.net/ont/geosparql#sfEquals");

        return hashMap;
    }

    private HashSet<String> initializeGeoVocabulary(){
        HashSet<String> hashSet =  new HashSet<>();
        hashSet.add("http://www.example.com/hasGeometry");
        hashSet.add("http://www.opengis.net/ont/geosparql#hasGeometry");
        return hashSet;
    }

    private HashSet<String> initializeWKTVocabulary(){
        HashSet<String> hashSet =  new HashSet<>();
        hashSet.add("http://www.example.com/asWKT");
        hashSet.add("http://www.opengis.net/ont/geosparql#asWKT");
        return hashSet;
    }

    private void initializeAggregates(){
        AggregateRegistry.init();
        AggregateRegistry.register("http://strdf.di.uoa.gr/ontology#area",new AreaAggFactory());
    }

    public static String getMaterialized(String function){
        return geoDictionary.dictionary.get(function);
    }

    public static boolean getGeometryName(String geometry){
        return geoDictionary.geoVocab.contains(geometry);
    }

    public static boolean getWKTName(String wkt){
        return geoDictionary.wktVocab.contains(wkt);
    }

    public static void setMapping(String key,String value){
        geoDictionary.varMap.put(key,value);
    }

    //Follow the mapping chain to the original entity
    public static String getMapping(String key){
        String geo = geoDictionary.varMap.get(key);
        String wkt = geoDictionary.varMap.get(geo);
        return wkt==null?geo:wkt;
    }
}

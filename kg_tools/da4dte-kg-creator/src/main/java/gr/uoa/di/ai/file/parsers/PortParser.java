package gr.uoa.di.ai.file.parsers;

import gr.uoa.di.ai.transformers.GeoJsonToWktTransformer;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.FileReader;
import java.io.IOException;
import java.util.Map;

public class PortParser implements FileParser{
    GeoJsonToWktTransformer transformer;

    public PortParser(){
        transformer = new GeoJsonToWktTransformer();
    }

    //Method to parse a poi .json file and extract triples from it
    public String parse(String path){
        // parsing file "JSONExample.json"
        Object obj = null;
        try {
            obj = new JSONParser().parse(new FileReader(path));
            // typecasting obj to JSONObject
            JSONObject jo = (JSONObject) obj;

            //Create a string builder
            StringBuilder builder = new StringBuilder();

            // Get id
            String id = (String) jo.get("id");
            builder.append(createID(id));

            // get poi properties
            Map properties = ((Map)jo.get("properties"));

            String name = (String) properties.get("name");
            if(name!=null) builder.append(createGADMTriple(id,"hasGADM_Name","\""+name.replaceAll("\"","'")+"\""));
            name = (String) properties.get("name:en");
            if(name!=null) builder.append(createGADMTriple(id,"hasGADM_Name","\""+name.replaceAll("\"","'")+"\"@en"));
            name = (String) properties.get("name:de");
            if(name!=null) builder.append(createGADMTriple(id,"hasGADM_Name","\""+name.replaceAll("\"","'")+"\"@de"));
            name = (String) properties.get("name:fr");
            if(name!=null) builder.append(createGADMTriple(id,"hasGADM_Name","\""+name.replaceAll("\"","'")+"\"@fr"));

            String amenity = (String) properties.get("amenity");
            if(amenity!=null) builder.append(createTriple(id,"hasAmenity","\""+amenity.replaceAll("\"","'")+"\""));

            String isHarbour = (String) properties.get("harbour");
            if(isHarbour!=null) builder.append(createTriple(id,"isHarbour","\""+isHarbour+"\""));

            String harbourCategory = (String) properties.get("seamark:harbour:category");
            if(harbourCategory!=null) builder.append(createTriple(id,"hasHarbourCategory","\""+harbourCategory.replaceAll("\"","'")+"\""));

            String seamark_type = (String) properties.get("seamark:type");
            if(seamark_type!=null) builder.append(createTriple(id,"hasSeamarkType","\""+seamark_type.replaceAll("\"","'")+"\""));

            String source = (String) properties.get("source");
            if(source!=null) builder.append(createTriple(id,"hasSource","\""+source+"\""));

            if(jo.get("geometry")!=null){
                String geometry = jo.get("geometry").toString();
                String wkt = transformer.transformGeoJson(geometry);
                builder.append(createGeometry(id,wkt));
            }

            return builder.toString();
        } catch (IOException | ParseException e) {
            e.printStackTrace();
            return "";
        }
    }

    private String createID(String id){
        return "<"+FileParser.RESOURCE+id+"> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://ai.di.uoa.gr/da4dte/ontology/port> .\n";
    }

    private String createTriple(String id, String property, String value){
        return "<"+FileParser.RESOURCE+id+"> <http://ai.di.uoa.gr/da4dte/ontology/" + property+"> " + value + " .\n";
    }

    private String createGADMTriple(String id, String property, String value){
        return "<"+FileParser.RESOURCE+id+"> <http://kr.di.uoa.gr/yago2geo/ontology/" + property + "> " + value + " .\n";
    }

    private String createGeometry(String id, String wkt){
        String triples = "<"+FileParser.RESOURCE+id+"> <http://www.opengis.net/ont/geosparql#hasGeometry> <"+FileParser.RESOURCE+"Geometry_port_"+id+"> .\n";
        triples += "<"+FileParser.RESOURCE+"Geometry_port_"+id+"> <http://www.opengis.net/ont/geosparql#asWKT> " +
                "\"<http://www.opengis.net/def/crs/EPSG/0/4326> " + wkt +"\"^^<http://www.opengis.net/ont/geosparql#wktLiteral> .\n";
        return triples;
    }
}
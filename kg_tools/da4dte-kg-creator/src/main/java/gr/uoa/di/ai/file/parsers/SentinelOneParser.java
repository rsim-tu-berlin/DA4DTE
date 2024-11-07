package gr.uoa.di.ai.file.parsers;

import gr.uoa.di.ai.transformers.GeoJsonToWktTransformer;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.FileReader;
import java.io.IOException;
import java.util.Map;

public class SentinelOneParser implements FileParser{

    GeoJsonToWktTransformer transformer;

    public SentinelOneParser(){
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

            //Image class properties

            String hasPlatform = (String) properties.get("platformShortName");
            if(hasPlatform!=null) builder.append(createTriple(id,"hasPlatform","\""+hasPlatform+"\""));

            String hasProductType = (String) properties.get("productType");
            if(hasProductType!=null) builder.append(createTriple(id,"hasProductType","\""+hasProductType+"\""));

            String hasTimestamp = (String) properties.get("datetime");
            if(hasTimestamp!=null) builder.append(createTriple(id,"hasTimestamp","\""+hasTimestamp+"\"^^<http://www.w3.org/2001/XMLSchema#dateTime>"));

            Map assets = ((Map)jo.get("assets"));
            Map quicklook = ((Map)assets.get("QUICKLOOK"));
            if(quicklook!=null) {
                String hasThumbnail = (String) quicklook.get("href");
                if (hasThumbnail != null) builder.append(createTriple(id, "hasThumbnail", "\"" + hasThumbnail + "\""));
            }

            //Sentinel-1 specific properties
            String hasPolarisationChannels = (String) properties.get("polarisationChannels");
            if(hasPolarisationChannels!=null) builder.append(createTriple(id,"hasPolarisationChannels","\""+hasPolarisationChannels+"\""));

            String hasOrigin = (String) properties.get("origin");
            if(hasOrigin!=null) builder.append(createTriple(id,"hasOrigin","\""+hasOrigin+"\""));

            String hasAuthority = (String) properties.get("authority");
            if(hasAuthority!=null) builder.append(createTriple(id,"hasAuthority","\""+hasAuthority+"\""));

            String hasTimeliness = (String) properties.get("timeliness");
            if(hasTimeliness!=null) builder.append(createTriple(id,"hasTimeliness","\""+hasTimeliness+"\""));

            Long hasOrbitNumber = (Long) properties.get("orbitNumber");
            if(hasOrbitNumber!=null) builder.append(createTriple(id,"hasOrbitNumber","\""+hasOrbitNumber+"\"^^<http://www.w3.org/2001/XMLSchema#integer>"));

            Long hasSliceNumber = (Long) properties.get("sliceNumber");
            if(hasSliceNumber!=null) builder.append(createTriple(id,"hasSliceNumber","\""+hasSliceNumber+"\"^^<http://www.w3.org/2001/XMLSchema#integer>"));

            String hasProcessingLevel = (String) properties.get("processingLevel");
            if(hasProcessingLevel!=null) builder.append(createTriple(id,"hasProcessingLevel","\""+hasProcessingLevel+"\""));

            String operationalMode = (String) properties.get("operationalMode");
            if(operationalMode!=null) builder.append(createTriple(id,"operationalMode","\""+operationalMode+"\""));

            String orbitDirection = (String) properties.get("orbitDirection");
            if(orbitDirection!=null) builder.append(createTriple(id,"orbitDirection","\""+orbitDirection+"\""));

            // WKT geometry creation
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
        return "<"+FileParser.RESOURCE+id+"> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://ai.di.uoa.gr/da4dte/ontology/sentinel1> .\n";
    }

    private String createTriple(String id, String property, String value){
        return "<"+FileParser.RESOURCE+id+"> <http://ai.di.uoa.gr/da4dte/ontology/" + property+"> " + value + " .\n";
    }

    private String createGeometry(String id, String wkt){
        String triples = "<"+FileParser.RESOURCE+id+"> <http://www.opengis.net/ont/geosparql#hasGeometry> <"+FileParser.RESOURCE+"Geometry_sentinel1_"+id+"> .\n";
        triples += "<"+FileParser.RESOURCE+"Geometry_sentinel1_"+id+"> <http://www.opengis.net/ont/geosparql#asWKT> " +
                "\"<http://www.opengis.net/def/crs/EPSG/0/4326> " + wkt +"\"^^<http://www.opengis.net/ont/geosparql#wktLiteral> .\n";
        return triples;
    }
}

package gr.uoa.di.ai.file.parsers;

import gr.uoa.di.ai.transformers.GeoJsonToWktTransformer;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.FileReader;
import java.io.IOException;
import java.util.Map;

public class SentinelTwoParser implements FileParser{

    GeoJsonToWktTransformer transformer;

    public SentinelTwoParser(){
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

            String hasPlatform = (String) properties.get("platform");
            if(hasPlatform!=null) builder.append(createTriple(id,"hasPlatform","\""+hasPlatform+"\""));

            String hasProductType = (String) properties.get("s2:product_type");
            if(hasProductType!=null) builder.append(createTriple(id,"hasProductType","\""+hasProductType+"\""));

            String hasTimestamp = (String) properties.get("datetime");
            if(hasTimestamp!=null) builder.append(createTriple(id,"hasTimestamp","\""+hasTimestamp+"\"^^<http://www.w3.org/2001/XMLSchema#dateTime>"));

            JSONArray links = ((JSONArray)jo.get("links"));
            for(Object link: links.stream().toList()){
                JSONObject item = (JSONObject) link;
                if(item.get("rel").equals("thumbnail")){
                    String thumbnail = (String) item.get("href");
                    if(thumbnail!=null) builder.append(createTriple(id,"hasThumbnail","\""+thumbnail+"\""));
                }
                if(item.get("rel").equals("derived_from")){
                    String derivedFrom = (String) item.get("href");
                    if(derivedFrom!=null) builder.append(createTriple(id,"derivedFrom","\""+derivedFrom+"\""));
                }
                if(item.get("rel").equals("license")){
                    String hasLicense = (String) item.get("href");
                    if(hasLicense!=null) builder.append(createTriple(id,"hasLicense","\""+hasLicense+"\""));
                }
            }

            //Sentinel-2 specific properties
            Object hasCloudCover = (Object) properties.get("eo:cloud_cover");
            if(hasCloudCover!=null) builder.append(createTriple(id,"hasCloudCover","\""+hasCloudCover+"\"^^<http://www.w3.org/2001/XMLSchema#double>"));

            Object hasMediumProbaCloudsPercentage = (Object) properties.get("s2:medium_proba_clouds_percentage");
            if(hasMediumProbaCloudsPercentage!=null) builder.append(createTriple(id,"hasMediumProbaCloudsPercentage","\""+hasMediumProbaCloudsPercentage+"\"^^<http://www.w3.org/2001/XMLSchema#double>"));

            Object hasNotVegetatedPercentage = (Object) properties.get("s2:not_vegetated_percentage");
            if(hasNotVegetatedPercentage!=null) builder.append(createTriple(id,"hasNotVegetatedPercentage","\""+hasNotVegetatedPercentage+"\"^^<http://www.w3.org/2001/XMLSchema#double>"));

            Object hasHighProbaCloudsPercentage = (Object) properties.get("s2:high_proba_clouds_percentage");
            if(hasHighProbaCloudsPercentage!=null) builder.append(createTriple(id,"hasHighProbaCloudsPercentage","\""+hasHighProbaCloudsPercentage+"\"^^<http://www.w3.org/2001/XMLSchema#double>"));

            String inConstellation = (String) properties.get("constellation");
            if(inConstellation!=null) builder.append(createTriple(id,"inConstellation","\""+inConstellation+"\""));

            Object hasThinCirrusPercentage = (Object) properties.get("s2:thin_cirrus_percentage");
            if(hasThinCirrusPercentage!=null) builder.append(createTriple(id,"hasThinCirrusPercentage","\""+hasThinCirrusPercentage+"\"^^<http://www.w3.org/2001/XMLSchema#double>"));

            Object hasSnowIcePercentage = properties.get("s2:snow_ice_percentage");
            if(hasSnowIcePercentage!=null) builder.append(createTriple(id,"hasSnowIcePercentage","\""+hasSnowIcePercentage+"\"^^<http://www.w3.org/2001/XMLSchema#double>"));

            Object hasVegetationPercentage = properties.get("s2:vegetation_percentage");
            if(hasVegetationPercentage!=null) builder.append(createTriple(id,"hasVegetationPercentage","\""+hasVegetationPercentage+"\"^^<http://www.w3.org/2001/XMLSchema#double>"));

            Object hasUnclassifiedPercentage = (Object) properties.get("s2:unclassified_percentage");
            if(hasUnclassifiedPercentage!=null) builder.append(createTriple(id,"hasUnclassifiedPercentage","\""+hasUnclassifiedPercentage+"\"^^<http://www.w3.org/2001/XMLSchema#double>"));

            Object hasSunAzimuth = (Object) properties.get("view:sun_azimuth");
            if(hasSunAzimuth!=null) builder.append(createTriple(id,"hasSunAzimuth","\""+hasSunAzimuth+"\"^^<http://www.w3.org/2001/XMLSchema#double>"));

            Object hasWaterPercentage = (Object) properties.get("s2:water_percentage");
            if(hasWaterPercentage!=null) builder.append(createTriple(id,"hasWaterPercentage","\""+hasWaterPercentage+"\"^^<http://www.w3.org/2001/XMLSchema#double>"));



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
        return "<"+FileParser.RESOURCE+id+"> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://ai.di.uoa.gr/da4dte/ontology/sentinel2> .\n";
    }

    private String createTriple(String id, String property, String value){
        return "<"+FileParser.RESOURCE+id+"> <http://ai.di.uoa.gr/da4dte/ontology/" + property+"> " + value + " .\n";
    }

    private String createGeometry(String id, String wkt){
        String triples = "<"+FileParser.RESOURCE+id+"> <http://www.opengis.net/ont/geosparql#hasGeometry> <"+FileParser.RESOURCE+"Geometry_sentinel2_"+id+"> .\n";
        triples += "<"+FileParser.RESOURCE+"Geometry_sentinel2_"+id+"> <http://www.opengis.net/ont/geosparql#asWKT> " +
                "\"<http://www.opengis.net/def/crs/EPSG/0/4326> " + wkt +"\"^^<http://www.opengis.net/ont/geosparql#wktLiteral> .\n";
        return triples;
    }
}

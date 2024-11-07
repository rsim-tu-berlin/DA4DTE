package gr.uoa.di.ai;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.geotools.geojson.geom.GeometryJSON;
import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.io.WKTWriter;

import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class JsonMRParser {
    String inputFile;
    String outputFile;
    String id;
    private static String RESOURCE = "http://ai.di.uoa.gr/da4dte/resource/";
    private static String URI = "http://ai.di.uoa.gr/da4dte/ontology/";

    public JsonMRParser(String inputFile){
        this.inputFile = inputFile;
        outputFile = inputFile.replaceAll("\\.geojson",".nt");
        this.id=null;
    }

    public void parse(){
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));

            JSONParser geojson = new JSONParser();
            Object geojsonObj = geojson.parse(new FileReader(inputFile));
            JSONObject geojsonObject = (JSONObject)geojsonObj;

            JSONArray features = (JSONArray) geojsonObject.get("features");

            for(Object feature: features){
                JSONObject feature_json = (JSONObject) ((JSONObject) feature).get("properties");

                // Get the water body id
                this.id = feature_json.get("ID").toString();
                writer.write(buildId());

                //Get the water body MR
                int mrgid =  Integer.parseInt(feature_json.get("MRGID").toString());
                writer.write(buildMRGID(mrgid));

                //Get the water body's area
                int area = Integer.parseInt(feature_json.get("area").toString());
                writer.write(buildArea(area));

                //Get the water body's name
                String name = feature_json.get("NAME").toString();
                writer.write(buildName(name));


                GeometryJSON gjson = new GeometryJSON();
                JSONObject geometry_json = (JSONObject) ((JSONObject) feature).get("geometry");

                Geometry geometry = gjson.read(geometry_json.toJSONString());

                // Convert to WKT
                WKTWriter wktWriter = new WKTWriter();
                String wkt = wktWriter.write(geometry);
                writer.write(buildGeometry(wkt));
            }

            writer.close();
        } catch (IOException | ParseException e) {
            e.printStackTrace();
        }
    }

    private String buildId(){
        return "<"+RESOURCE+"sea_"+this.id+"> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <" + URI+"sea> .\n";
    }

    private String buildName(String name){
        return "<"+RESOURCE+"sea_"+this.id+"> <http://kr.di.uoa.gr/yago2geo/ontology/hasGADM_Name> \"" + name +"\" .\n";
    }

    private String buildMRGID(int mrgid){
        return "<"+RESOURCE+"sea_"+this.id+"> <"+ URI +"hasMRGID> \"" + mrgid +"\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n";
    }

    private String buildArea(int area){
        return "<"+RESOURCE+"sea_"+this.id+"> <"+ URI +"hasArea> \"" + area +"\"^^<http://www.w3.org/2001/XMLSchema#double> .\n";
    }

    private String buildGeometry(String wkt){
        String result = "<"+RESOURCE+"sea_"+this.id+"> <http://www.opengis.net/ont/geosparql#hasGeometry> <"+RESOURCE+"Geometry_sea_"+this.id+"> .\n";
        result += "<"+RESOURCE+"Geometry_sea_"+this.id+"> <http://www.opengis.net/ont/geosparql#asWKT> \"<http://www.opengis.net/def/crs/EPSG/0/4326> "+wkt+"\"^^<http://www.opengis.net/ont/geosparql#wktLiteral> .\n";
        return result;
    }
}

package gr.uoa.di.ai.file.parsers;

import gr.uoa.di.ai.transformers.GeoJsonToWktTransformer;
import org.apache.commons.lang3.ArrayUtils;
import org.geotools.geometry.jts.JTS;
import org.geotools.referencing.CRS;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.locationtech.jts.geom.*;
import org.locationtech.jts.io.WKTReader;
import org.locationtech.jts.operation.linemerge.LineMerger;
import org.opengis.referencing.FactoryException;
import org.opengis.referencing.crs.CoordinateReferenceSystem;
import org.opengis.referencing.operation.MathTransform;
import org.opengis.referencing.operation.TransformException;

import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class RiverParser implements FileParser{
    GeoJsonToWktTransformer transformer;

    public RiverParser(){
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
            if(name!=null) builder.append(createGADMTriple(id,"hasGADM_Name","\""+name+"\""));
            name = (String) properties.get("name:en");
            if(name!=null) builder.append(createGADMTriple(id,"hasGADM_Name","\""+name+"\"@en"));
            name = (String) properties.get("name:de");
            if(name!=null) builder.append(createGADMTriple(id,"hasGADM_Name","\""+name+"\"@de"));
            name = (String) properties.get("name:fr");
            if(name!=null) builder.append(createGADMTriple(id,"hasGADM_Name","\""+name+"\"@fr"));

            String destination = (String) properties.get("destination");
            if(destination!=null) builder.append(createTriple(id,"hasDestination","\""+destination+"\""));

            String waterwayType = (String) properties.get("waterway");
            if(waterwayType!=null) builder.append(createTriple(id,"waterwayType","\""+waterwayType+"\""));

            String wikidata = (String) properties.get("wikidata");
            if(wikidata!=null) builder.append(createWikiLink(id,wikidata));

            List<String> linestrings = new ArrayList<>();
            if(jo.get("geometry")!=null){
                Map geometry = ((Map)jo.get("geometry"));
                JSONArray geometries = (JSONArray) geometry.get("geometries");
                for(Object geom: geometries){
                    String wkt = this.getGeometry((JSONObject) geom);
                    if(wkt!=null)
                        linestrings.add(wkt);
                }
                builder.append(createGeometry(id,mergeLineStrings(linestrings)));
            }


            return builder.toString();
        } catch (IOException | ParseException e) {
            e.printStackTrace();
            return "";
        }
    }

    public String mergeLineStrings(List<String> geometries){
        WKTReader reader = new WKTReader();
        LineMerger merger = new LineMerger();
        ArrayList<LineString> lineStringArrayList = new ArrayList<>();

        for (String wkt : geometries) {
            Geometry geom = null;
            try {
                geom = reader.read(wkt.replace("\"",""));
            } catch (org.locationtech.jts.io.ParseException e) {
                e.printStackTrace();
            }
            merger.add(geom);
            lineStringArrayList.add((LineString) geom);
        }

        Collection<LineString> collection = merger.getMergedLineStrings();

        Collection<Polygon> polygons = new ArrayList<>();
        for(LineString l:collection){
            Coordinate[] points = l.getCoordinates();

            ArrayList<Coordinate> soln = new ArrayList<>();
            //store initial points
            soln.addAll(Arrays.asList(points));
            // reverse the list
            ArrayUtils.reverse(points);
            // for each point move offset metres right
            for (Coordinate c:points) {
                soln.add(new Coordinate(c.x, c.y));
            }
            // close the polygon
            soln.add(soln.get(0));
            // create polygon
            GeometryFactory gf = new GeometryFactory();
            Polygon polygon = gf.createPolygon(soln.toArray(new Coordinate[] {}));
            polygons.add(polygon);
        }
        Polygon[] polygonsArray = polygons.toArray(new Polygon[polygons.size()]);
        GeometryFactory factory = new GeometryFactory();
        MultiPolygon geometryCollection = factory.createMultiPolygon(polygonsArray);

        return geometryCollection.toText();
    }

    public String getGeometry(JSONObject jsonObject){
        //Get geometry and form wkt
        String type = (String)jsonObject.get("type");

        StringBuilder wkt = new StringBuilder();
        if(type.equals("LineString")){
            wkt.append("\"LINESTRING(");

            JSONArray coordinates = (JSONArray) jsonObject.get("coordinates");
            for (Object coordinate : coordinates) {
                JSONArray values = (JSONArray)coordinate;
                String x = String.valueOf((Double)values.get(0));
                String y = String.valueOf((Double)values.get(1));

                wkt.append(x).append(" ").append(y).append(",");
            }
            wkt.deleteCharAt(wkt.length()-1);
            wkt.append(")\"");
        }
        else if(type.equals("Point")){
            wkt.append("\"LINESTRING(");
            JSONArray coordinates = (JSONArray) jsonObject.get("coordinates");
            String x = String.valueOf((Double)coordinates.get(0));
            String y = String.valueOf((Double)coordinates.get(1));

            wkt.append(x).append(" ").append(y).append(",");
            wkt.append(x).append(" ").append(y).append(",");

            wkt.deleteCharAt(wkt.length()-1);
            wkt.append(")\"");
        }

        String sourceCRS = "EPSG:"+4326;
        String targetCRS = "EPSG:4326";

        // Parse the source and target CRS
        String wkt_4326 = null;
        try {
            CoordinateReferenceSystem source = CRS.decode(sourceCRS,true);
            CoordinateReferenceSystem target = CRS.decode(targetCRS,true);
            // Get the transform from the source CRS to the target CRS
            MathTransform transform = CRS.findMathTransform(source, target);

            WKTReader wktReader = new WKTReader();
            Geometry geom = wktReader.read(wkt.toString().replace("\"",""));
            Geometry transformedGeometry = JTS.transform(geom,transform);

            wkt_4326 = "\"" + transformedGeometry.toText() + "\"^^<http://www.opengis.net/ont/geosparql#wktLiteral>";
        } catch (org.locationtech.jts.io.ParseException | FactoryException | TransformException e) {
            e.printStackTrace();
        }
        return wkt_4326;
    }

    private String createID(String id){
        return "<"+FileParser.RESOURCE+id+"> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://ai.di.uoa.gr/da4dte/ontology/river> .\n";
    }

    private String createTriple(String id, String property, String value){
        return "<"+FileParser.RESOURCE+id+"> <http://ai.di.uoa.gr/da4dte/ontology/" + property+"> " + value + " .\n";
    }

    private String createGADMTriple(String id, String property, String value){
        return "<"+FileParser.RESOURCE+id+"> <http://kr.di.uoa.gr/yago2geo/ontology/" + property + "> " + value + " .\n";
    }

    private String createWikiLink(String id, String link){
        return "<"+FileParser.RESOURCE+id+"> <http://ai.di.uoa.gr/da4dte/ontology/wikilink> <https://www.wikidata.org/wiki/" + link+"> .\n";
    }

    private String createGeometry(String id, String wkt){
        String triples = "<"+FileParser.RESOURCE+id+"> <http://www.opengis.net/ont/geosparql#hasGeometry> <"+FileParser.RESOURCE+"Geometry_river_"+id+"> .\n";
        triples += "<"+FileParser.RESOURCE+"Geometry_river_"+id+"> <http://www.opengis.net/ont/geosparql#asWKT> " +
                "\"<http://www.opengis.net/def/crs/EPSG/0/4326> " + wkt +"\"^^<http://www.opengis.net/ont/geosparql#wktLiteral> .\n";
        return triples;
    }
}

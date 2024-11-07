import org.semanticweb.yars.nx.Node;
import org.semanticweb.yars.nx.parser.NxParser;
import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.io.ParseException;
import org.locationtech.jts.io.WKTReader;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

public class GeometryLocator {
    String path;
    String geoFile;
    String duplicates;

    GeometryLocator(String path){
        this.path = path;
        this.geoFile = path.replace(".nt","_sorted.nt");
        this.duplicates = null;
    }

    public void selectGeometries(String geoPredicate){
        try {
            File ntFile = new File(geoFile);
            ntFile.createNewFile();
            FileWriter writer = new FileWriter(geoFile);


            FileInputStream is = new FileInputStream(this.path);
            NxParser nxp = new NxParser();
            nxp.parse(is);

            for (Node[] nx : nxp){
                if(geoPredicate.equals(nx[1].toString()))
                    writer.write(nx[0] + " " + nx[1] + " " + nx[2] + " .\n");
            }

            writer.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void findDuplicates(){
        try {
            this.duplicates = this.path.replace(".nt",".duplicates");
            File ntFile = new File(duplicates);
            ntFile.createNewFile();
            FileWriter writer = new FileWriter(duplicates);


            FileInputStream is = new FileInputStream(this.geoFile);
            NxParser nxp = new NxParser();
            nxp.parse(is);

            HashMap<String,String> map = new HashMap<>();
            for (Node[] nx : nxp){
                String entity = map.get(nx[0].toString());
                if(entity==null)
                    map.put(nx[0].toString(),nx[2].toString());
                else{
                    if(entity.contains("Geometry_gadm")) {
                        writer.write(entity + "\n");
                        map.replace(nx[0].toString(),entity,nx[2].toString());
                    }
                    else{
                        writer.write(nx[2].toString()+"\n");
                    }
                }
            }

            writer.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void removeDuplicates(){
        try {
            //Load duplicates into a hashset
            HashSet<String> duplicateSet = new HashSet<>();
            BufferedReader reader = new BufferedReader(new FileReader(this.duplicates));
            String line = reader.readLine();
            while (line != null) {
                duplicateSet.add(line);
                line = reader.readLine();
            }
            reader.close();

            String finalFile = this.path.replace(".nt","_cleaned.nt");
            File ntFile = new File(finalFile);
            ntFile.createNewFile();
            FileWriter writer = new FileWriter(finalFile);


            FileInputStream is = new FileInputStream(this.path);
            NxParser nxp = new NxParser();
            nxp.parse(is);

            for (Node[] nx : nxp){
                if(!duplicateSet.contains(nx[0].toString()) && !duplicateSet.contains(nx[2].toString())){
                    writer.write(nx[0] + " " + nx[1] + " " + nx[2] + " .\n");
                }
            }

            writer.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void removeDuplicatesByArea(){
        try {
            String fileName = this.path.replace(".nt","_cleaned_by_area.nt");
            File ntFile = new File(fileName);
            ntFile.createNewFile();
            FileWriter writer = new FileWriter(fileName);


            FileInputStream is = new FileInputStream(this.path);
            NxParser nxp = new NxParser();
            nxp.parse(is);

            HashMap<String,String> mapGeo = new HashMap<>();
            WKTReader wktReader = new WKTReader();

            for (Node[] nx : nxp){
                if(nx[1].toString().equals("<http://www.opengis.net/ont/geosparql#asWKT>")){
                    if(mapGeo.get(nx[0].toString())==null) {
                        mapGeo.put(nx[0].toString(), nx[2].toString());
                    }
                    else{
                        Geometry g1 = wktReader.read(mapGeo.get(nx[0].toString()).replace("\"",""));
                        Geometry g2 = wktReader.read(nx[2].toString().replace("\"",""));
                        double area1 = g1.getArea();
                        double area2 = g2.getArea();
                        if(area1>area2) mapGeo.put(nx[0].toString(),"\""+g1.toText()+"\"");
                        else mapGeo.put(nx[0].toString(),"\""+g2.toText()+"\"");
                    }
                }
                else{
                    writer.write(nx[0] + " " + nx[1] + " " + nx[2] + " .\n");
                }
            }

            // Iterating HashMap through for loop
            for (Map.Entry<String, String> set : mapGeo.entrySet()) {
                writer.write(set.getKey() + " "
                        + "<http://www.opengis.net/ont/geosparql#asWKT>" + " " + set.getValue() + " .\n");
            }


            writer.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ParseException e) {
            e.printStackTrace();
        }
    }
}

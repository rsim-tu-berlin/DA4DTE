import org.locationtech.jts.io.ParseException;
import org.locationtech.jts.io.WKTReader;
import org.opengis.referencing.FactoryException;
import org.opengis.referencing.NoSuchAuthorityCodeException;
import org.opengis.referencing.operation.TransformException;
import org.semanticweb.yars.nx.Node;
import org.semanticweb.yars.nx.parser.NxParser;
import org.geotools.geometry.jts.JTS;
import org.geotools.referencing.CRS;
import org.locationtech.jts.geom.Geometry;
import org.opengis.referencing.crs.CoordinateReferenceSystem;
import org.opengis.referencing.operation.MathTransform;

import java.io.*;
import java.util.HashMap;

public class WKTTransformer {

    String path;
    String crs;

    public WKTTransformer(String path){
        this.path = path;
        this.crs = "<http://www.opengis.net/def/crs/EPSG/0/4326>";
    }

    public void transform(){
        try {
            String fileName = this.path.replace(".nt","_crs.nt");
            File ntFile = new File(fileName);
            ntFile.createNewFile();
            FileWriter writer = new FileWriter(fileName);


            FileInputStream is = new FileInputStream(this.path);
            NxParser nxp = new NxParser();
            nxp.parse(is);

            for (Node[] nx : nxp){
                String s = nx[0].toString();
                String p = nx[1].toString();
                String o = nx[2].toString();
                if(nx[1].toString().equals("<http://www.opengis.net/ont/geosparql#asWKT>")){
                    if(!o.contains(this.crs)){
                        o = o.substring(1);
                        o = "\""+this.crs+"  "+o;
                    }
                }
                writer.write(s + " " + p + " " + o + " .\n" );
            }

            writer.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void dirtyTransform(){
        try {
            String fileName = this.path.replace(".nt","_no_crs.nt");
            File ntFile = new File(fileName);
            ntFile.createNewFile();
            FileWriter writer = new FileWriter(fileName);


            FileInputStream is = new FileInputStream(this.path);
            NxParser nxp = new NxParser();
            nxp.parse(is);

            for (Node[] nx : nxp){
                String s = nx[0].toString();
                String p = nx[1].toString();
                String o = nx[2].toString();
                if(nx[1].toString().equals("<http://www.opengis.net/ont/geosparql#asWKT>")){
                    if(o.contains(this.crs)){
                        o = o.replace("\"<http://www.opengis.net/def/crs/EPSG/0/4326>","");
                        o = o.trim();
                        o = "\"" + o;
                    }
                }
                writer.write(s + " " + p + " " + o + " .\n" );
            }

            writer.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void transformCRS(){
        try {
            String sourceCRS = "EPSG:2100";
            String targetCRS = "EPSG:4326";
            String fileName = this.path.replace(".nt","_4326.nt");
            File ntFile = new File(fileName);
            ntFile.createNewFile();
            FileWriter writer = new FileWriter(fileName);


            // Parse the source and target CRS
            CoordinateReferenceSystem source = CRS.decode(sourceCRS);
            CoordinateReferenceSystem target = CRS.decode(targetCRS);

            // Get the transform from the source CRS to the target CRS
            MathTransform transform = CRS.findMathTransform(source, target);

            WKTReader wktReader = new WKTReader();

            FileInputStream is = new FileInputStream(this.path);
            NxParser nxp = new NxParser();
            nxp.parse(is);

            for (Node[] nx : nxp){
                String s = nx[0].toString();
                String p = nx[1].toString();
                String o = nx[2].toString();
                if(nx[1].toString().equals("<http://www.opengis.net/ont/geosparql#asWKT>")){
                    Geometry geom = wktReader.read(nx[2].toString().replace("\"",""));
                    Geometry transformedGeometry = JTS.transform(geom,transform);
                    o = "\"" + transformedGeometry.toText() + "\"";
                }
                writer.write(s + " " + p + " " + o + " .\n" );
            }

            writer.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        } catch (IOException | NoSuchAuthorityCodeException | ParseException e) {
            e.printStackTrace();
        } catch (FactoryException e) {
            e.printStackTrace();
        } catch (TransformException e) {
            e.printStackTrace();
        }
    }
}

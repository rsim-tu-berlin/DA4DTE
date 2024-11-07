import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.io.ParseException;
import org.locationtech.jts.io.WKTReader;

import java.io.*;

public class GeometryValidator {
    String path;

    public GeometryValidator(String path){
        this.path = path;
    }

    public void correct(){
        try {
            String fileName = this.path.replace(".tsv","_valid_geom.tsv");
            File ntFile = new File(fileName);
            ntFile.createNewFile();
            FileWriter writer = new FileWriter(fileName);
            writer.write("WKT\tEntity\n");

            WKTReader wktReader = new WKTReader();

            BufferedReader reader = new BufferedReader(new FileReader(this.path));
            String line = reader.readLine();
            line = reader.readLine();
            while (line != null) {
                String[] elems = line.split("\t",2);
                String geoString = elems[0].replace("\"","");
                Geometry geom = wktReader.read(geoString);
                Geometry valid =  geom.buffer(0);
                writer.write("\""+ valid.toText() + "\"\t" + elems[1] + "\n");
                line = reader.readLine();
            }

            writer.close();
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ParseException e) {
            e.printStackTrace();
        }
    }
}

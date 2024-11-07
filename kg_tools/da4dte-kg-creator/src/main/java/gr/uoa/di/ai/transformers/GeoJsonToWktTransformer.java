package gr.uoa.di.ai.transformers;

import org.geotools.geojson.geom.GeometryJSON;
import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.io.WKTWriter;

import java.io.IOException;

public class GeoJsonToWktTransformer {
    WKTWriter writer;
    public GeoJsonToWktTransformer(){
        this.writer = new WKTWriter();
    }

    public String transformGeoJson(String geojson){
        GeometryJSON gjson = new GeometryJSON();
        Geometry geometry = null;
        try {
            geometry = gjson.read(geojson);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        WKTWriter writer = new WKTWriter();
        return writer.write(geometry);
    }
}

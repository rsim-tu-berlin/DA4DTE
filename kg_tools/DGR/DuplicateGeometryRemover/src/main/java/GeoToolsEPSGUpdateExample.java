import org.geotools.referencing.CRS;
import org.opengis.referencing.crs.CoordinateReferenceSystem;
import org.opengis.util.FactoryException;

public class GeoToolsEPSGUpdateExample {
    public static void updateEPSGDatabase() {
        // Update the GeoTools EPSG database using the latest version available on the EPSG registry website
        CRS.getAuthorityFactory(true);
    }
}

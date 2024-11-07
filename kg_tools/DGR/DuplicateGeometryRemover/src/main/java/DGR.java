public class DGR {
    public static void main(String[] args) {
        if(args[0].equals("Deduplicate")) {
            GeometryLocator geometryLocator = new GeometryLocator(args[1]);
            geometryLocator.selectGeometries("<http://www.opengis.net/ont/geosparql#hasGeometry>");
            geometryLocator.findDuplicates();
            geometryLocator.removeDuplicates();
        }
        // TODO - transform geometries into crs 4326
        else if(args[0].equals("AddEPSGTag")){
            WKTTransformer wktTransformer = new WKTTransformer(args[1]);
            wktTransformer.transform();
        }
        // TODO - remove all epsg values form wkts, not only 4326
        else if(args[0].equals("RemoveEPSGTag")){
            WKTTransformer wktTransformer = new WKTTransformer(args[1]);
            wktTransformer.dirtyTransform();
        }
        else if(args[0].equals("NTriplesToTSV")){
            ResourceMap resourceMap = new ResourceMap(args[1]);
            resourceMap.ntToTSV();
        }
        else if(args[0].equals("JedAISpatialMap")){
            ResourceMap resourceMap = new ResourceMap(args[1]);
            resourceMap.mapEntities(args[2],args[3]);
        }
        else if(args[0].equals("ValidateGeometry")){
            GeometryValidator geometryValidator = new GeometryValidator(args[1]);
            geometryValidator.correct();
        }
        else if(args[0].equals("DeduplicateByArea")){
            GeometryLocator geometryLocator = new GeometryLocator(args[1]);
            geometryLocator.removeDuplicatesByArea();
        }
        // TODO - merge with TransformTo4326 - right now only works with 2100 to 4326
        else if(args[0].equals("TransformTo4326")){
            WKTTransformer wktTransformer = new WKTTransformer(args[1]);
            wktTransformer.transformCRS();
        }
        else if(args[0].equals("KeepTriples")){
            ResourceMap resourceMap = new ResourceMap(args[1]);
            resourceMap.keepTripleFromDictionary(args[2]);
        }
    }
}

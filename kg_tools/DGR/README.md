# DGR

A toolchain for manipulating geospatial rdf data

## Compilation

1. Requires maven 3.6.3 and Java 8

2. Compile using maven

       mvn clean package shade:shade

3. Run with the following commands:

### Geometry Deduplication based on first apperance

    java -cp target/DuplicateGeometryRemover-1.0-SNAPSHOT.jar DGR Deduplicate [path_to_nt]

Output will be a file in the same location as the source file with the _sorted.nt extension

### Add EPSG tag to wkt geometries

    java -cp target/DuplicateGeometryRemover-1.0-SNAPSHOT.jar DGR AddEPSGTag [path_to_nt]

Output will be a file in the same location as the source file with the _crs.nt extension

### Remove EPSG tag from wkt geometries

    java -cp target/DuplicateGeometryRemover-1.0-SNAPSHOT.jar DGR RemoveEPSGTag [path_to_nt]

Output will be a file in the same location as the source file with the _no_crs.nt extension

### Transform a .nt file into a tsv file with two columns. The first column contains the wkt geometry and second the entity of the geometry

    java -cp target/DuplicateGeometryRemover-1.0-SNAPSHOT.jar DGR NTriplesToTSV [path_to_nt]

Output will be a file in the same location as the source file with the geo_only.tsv extension

### Mapping the results of the JedAI-Spatial toolchain to the original entities

    java -cp target/DuplicateGeometryRemover-1.0-SNAPSHOT.jar DGR JedAISpatialMap [path_to_nt] [source_tsv] [target_tsv]

  Output will be a file in the same location as the source file with the _map.nt extension

### Validate and correct wkt geometries. If geometry is not valid, ogc:buffer(0) is used to correct it.

    java -cp target/DuplicateGeometryRemover-1.0-SNAPSHOT.jar DGR ValidateGeometry [path_to_tsv]

  Output will be a file in the same location as the source file with the valid_geom.tsv extension

 ### Geometry Deduplication based on largest area

    java -cp target/DuplicateGeometryRemover-1.0-SNAPSHOT.jar DGR DeduplicateByArea [path_to_nt]

Output will be a file in the same location as the source file with the _cleaned_by_area.nt extension

### Transform given geometries to EPSG:4326 ( currently working only for EPSG:2100 )

    java -cp target/DuplicateGeometryRemover-1.0-SNAPSHOT.jar DGR TransformTo4326 [path_to_nt]

Output will be a file in the same location as the source file with the _4326.nt extension

## Operating with JedAI-Spatial

1. Transform the .nt files using NTriplesToTSV
2. Transform the .tsv file into .csv files using cs.py script (replace the input file and output file names in the script)
3. After you get the results from the JedAI-Spatial workflows get the mappings using JedAISpatialMap

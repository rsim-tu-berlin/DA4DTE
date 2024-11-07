# GoST

GoST -  **G**eoSPARQL t**o** **S**PARQL **T**ranspiler

GoST is a transpiler converting GeoSPARQL queries using geospatial topological funtions, to SPARQL queries using the respective materialized relationships. The project is built using Java 8, Maven 3.6.3 and Jena ARQ .

 Compilation:
 
    mvn clean package shade:shade
    
Execution:

    java -cp target\GoST-1.0-SNAPSHOT.jar gr.uoa.di.ai.gost.Transpiler query [output_file]

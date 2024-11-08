# DA4DTE Knowledge Graph Tools

The tools outlined above focus on the construction, deployment, usability, and optimized performance of the knowledge graph developed for the purposes of the DA4DTE project.

## Repository Information

### Knowledge graph construction tools

- `da4dte-kg-creator/`: A tool for parsing and transforming input JSON data on Sentinel-1, Sentinel-2, points of interest (POIs), rivers, and ports, provided by eGeos, into N-Triples format for integration into the knowledge graph.
- `marineregions-parser/`: A tool for parsing and formating input NTriple files data from the Marine Regions datasource to integrate sea sectors into the knowledge graph.

### Deployment Tool

- `da4dte-kg-docker/`: A dedicated Dockerfile for creating and provisioning a GraphDB endpoint preloaded with DA4DTE knowledge graph data.

### Optimization Tools

- `DGR/`: A toolchain for parsing and creating geospatial triples from various data sources. It was used for the purposes of this project alongside [JedAI-spatial](https://github.com/AI-team-UoA/JedAI-spatial), to materiliaze geospatial relationships between geoentities in the DA4DTE knowledge graph.
- `GoST/` 👻: A GeoSPARQL-to-SPARQL transpiler that automatically converts GeoSPARQL queries into equivalent SPARQL queries, targeting materialized relationships within the knowledge graph.

## Requirements 

All tools required maven 3.6.3 and Java 17 to run

## Compilation

All tools can be compiled with the command:

    mvn clean package shade:shade

## Usage: da4dte-kg-creator

Run the following command after successfully compiling the project:

    java -cp target/da4dte-kg-creator-1.0-SNAPSHOT.jar gr.uoa.di.ai.Main [poi_input_dir] [poi_output] [port_input_dir] [port_output] [river_input_dir] [river_output] [s1_input_dir] [s1_output] [s2_input_dir] [s2_output]

## Usage: marineregions-parser

    java -cp target/marineregions-parser-1.0-SNAPSHOT.jar gr.uoa.di.ai.MarineRegionParser [marine_regions_input_file]

### Usage: da4dte-kg-docker

Info can be found inside the `da4dte-kg-docker/` directory or the [project's repository](https://github.com/AI-team-UoA/da4dte-kg-docker)

### Usage for Optimization Tools

- GosT: Info can be found in the [project's repository](https://github.com/AI-team-UoA/GoST)
- DGR: Info can be found in the [project's repository](https://github.com/KwtsPls/DGR)

## Team & Authors

<img align="right" src="https://github.com/AI-team-UoA/.github/blob/main/AI_LOGO.png?raw=true" alt="ai-team-uoa" width="200"/>

- [Sergios-Anestis Kefalidis](http://users.uoa.gr/~skefalidis/), Research Associate at the University of Athens, Greece
- [Kostas Plas](https://www.madgik.di.uoa.gr/el/people/msc-student/kplas), Research Associate at the University of Athens, Greece
- [Manolis Koubarakis](https://cgi.di.uoa.gr/~koubarak/), Professor at the University of Athens, Greece

This engine was developed by the [AI-Team](https://ai.di.uoa.gr) of the Department of Informatics and Telecommunications at the University of Athens.

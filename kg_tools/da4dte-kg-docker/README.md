# DA4DTE KG DOCKER

Docker to deploy the DA4DTE knowledge graph, using Graphdb.

## Docker setup

Create a directory named data:

      mkdir data

Transfer the .gz knowledge graph datafiles inside the new directory:

      mv [path_to_kg_files] data/

To build the docker image run:

      docker build -t graphdb .

To run the docker container image run:

      sudo docker run --name graphdb-container -p 7200:7200 graphdb

A GraphDB instance will be online on http://localhost:7200

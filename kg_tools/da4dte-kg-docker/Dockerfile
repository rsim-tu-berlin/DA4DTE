################################
# Dockerfile for DA4DTE's KG   #
# DI @ UoA                     #
#                              #
# Java 11                      #
# graphdb 10.6.3               #
################################

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PORT 7200
ENV GRAPH_DB_VERSION 10.6.3

# INSTALL PREREQUISITIES
RUN apt-get update \
 && apt-get install -y \
    wget \
    openjdk-11-jdk \
    curl \
    unzip \
    gzip \
    gnupg2 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Download and extract graphdb
RUN wget "https://maven.ontotext.com/repository/owlim-releases/com/ontotext/graphdb/graphdb/${GRAPH_DB_VERSION}/graphdb-${GRAPH_DB_VERSION}-dist.zip" \
	&& unzip graphdb-${GRAPH_DB_VERSION}-dist.zip

# Copy the knowledge graph data
COPY data/* .

# Extract the knowledge graph data
RUN gzip -d da4dte.nt.gz

RUN gzip -d images.nt.gz

RUN gzip -d non_satellite_mat_reduced_map.nt.gz

RUN gzip -d s1_mat_intersects_only_map.nt.gz

RUN gzip -d s2_mat_intersects_only_map.nt.gz

RUN gzip -d da4dte_en_labels_unique.nt.gz

RUN gzip -d seas.nt.gz

RUN gzip -d s1_seas_mat_map.nt.gz

RUN gzip -d s2_seas_mat_map.nt.gz

RUN gzip -d seas_da4dte_mat_map.nt.gz

RUN gzip -d da4dte_areas.nt.gz

COPY template.ttl . 

EXPOSE 7200
EXPOSE 7300

RUN ./graphdb-10.6.3/bin/importrdf preload -c template.ttl da4dte.nt da4dte_en_labels_unique.nt images.nt non_satellite_mat_reduced_map.nt s1_mat_intersects_only_map.nt s2_mat_intersects_only_map.nt seas.nt s1_seas_mat_map.nt s2_seas_mat_map.nt seas_da4dte_mat_map.nt da4dte_areas.nt
ENTRYPOINT ["./graphdb-10.6.3/bin/graphdb"]

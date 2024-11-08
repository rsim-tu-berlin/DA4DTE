############################
# Dockerfile for Strabon   #
# RK-team                  #
# Madgik                   #
# DI @ UoA                 #
#                          #
# java8                    #
# maven 3.6.3              #
# git 1.9.1                #
# tomcat 9.0.71            #
# postgres 12.13           #
# postgis 3.3.2            #
############################


FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && \
    apt-get install -y && \
    apt-get install -y gnupg2 && \
    apt-get install -y curl

# Get all necessary packages for gpu utilization
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

RUN apt-get update && \
    apt-get install -y nvidia-container-runtime && \
    apt-get install -y nvidia-container-toolkit

# Install gzip
RUN apt-get install -y gzip
RUN apt-get install -y wget

# Install Python 3.10 (if needed)
RUN apt-get install -y python3.10

# Install pip (if needed)
RUN apt-get install -y python3-pip

# Copy the requirements.txt file into the container
COPY conda_env.yml /tmp/

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda

# Add Miniconda to the PATH
ENV PATH="/opt/miniconda/bin:${PATH}"

# Update Conda
RUN conda update -n base -c defaults conda
RUN conda install python=3.10

# Install Python dependencies
RUN conda env create -f /tmp/conda_env.yml

# Clean up APT when done
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Extract deploy.ckpt
COPY lit4rsvqa.py .
COPY vqa_server.py .
COPY deploy.ckpt.gz .
COPY templates ./templates 
RUN gzip -d deploy.ckpt.gz

# Deploy and run vqa_server
SHELL ["conda", "run", "-n", "lit4rsvqa", "/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "-n", "lit4rsvqa", "python3","vqa_server.py","8080","deploy.ckpt"]

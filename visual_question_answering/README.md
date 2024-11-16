# DA4DTE-VQA

This repository contains the code used to train and deploy the Visual Question Answering engine for [DA4DTE](https://eo4society.esa.int/projects/da4dte/). It uses [RSVQAxBEN](https://github.com/syvlo/RSVQAxBEN) for training therefore only Sentinel-2 images (10-bands) are supported.

## Input

The model takes input in the form of a question in natural language (English) and a Sentinel-2 image with 10 bands and 120x120 dimensions. 

##### The bands must be combined as follows:

```
["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"]
```

## Setup

To setup with Conda:

`$ conda create --name <env> --file requirements.txt`

## Training

Run `lit4rsvqa.py` with the appropriate options. For more information: 

`$ lit4rsvqa.py --help`

## Web Server

Run `vqa_server.py` with the appropriate options (port and model checkpoint are required). For more information: 

`$ vqa_server.py --help`

## Docker setup

### Enable nvidia-container-toolkit

Configure the repository:

      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee                   /etc/apt/sources.list.d/nvidia-container-toolkit.list \
      && sudo apt-get update

Install the NVIDIA Container Toolkit packages:

      sudo apt-get install -y nvidia-container-toolkit

Configure the container runtime by using the nvidia-ctk command:

      sudo nvidia-ctk runtime configure --runtime=docker

Restart the Docker daemon:

      sudo systemctl restart docker

### Build and run through Dockerfile

Enter the `docker` directory:

      cd docker/

To build the docker image run:

      sudo docker build -t vqa .

To run the docker container image run:

      sudo docker run --gpus all --name vqa-container -p 5000:8080 vqa

After a short delay vqa_server will be online on http://localhost:5000

### Acknowledgement

The engine uses the system presented in:

```bibtex
@INPROCEEDINGS{10281674,
    author={Hackel, Leonard and Clasen, Kai Norman and Ravanbakhsh, Mahdyar and Demir, Begüm},
    booktitle={IGARSS 2023 - 2023 IEEE International Geoscience and Remote Sensing Symposium}, 
    title={LIT-4-RSVQA: Lightweight Transformer-Based Visual Question Answering in Remote Sensing}, 
    year={2023},
    volume={},
    number={},
    pages={2231-2234},
    doi={10.1109/IGARSS52108.2023.10281674}
}
```


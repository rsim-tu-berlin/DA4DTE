# Search-by-Image Engines

In this repository, the software for search-by-image engines is documented. Three scripts execute the main parts of this sub-project:
- Pretraining: [`main_train.py`](./main_train.py)
- Deep Hashing Module: [`main_hashify.py`](./main_hashify.py)
- Evaluation: [`main_retrieval.py`](./main_retrieval.py)


## Datasets

There are two datasets used to train search-by-image engines, which are [BigEarthNet-MM](https://bigearth.net/) (BEN) and vessel detection dataset (VDD). In the corresponding folders, implementations of torch datasets are provided. VDD directly operates on TIF-files and BEN expects a LMDB file, which can be obtained by following the guidelines [here](https://github.com/kai-tub/bigearthnet_encoder). For VDD, we create custom query/archive splits by using the [`gen_query_archive.ipynb`](./src/vessel_dataset/gen_query_archive.ipynb) notebook.


## Pretraining

#### Documentation

This repository contains three pretraining methods, namely Cross-Modal Masked Autoencoder (CMMAE) [1], CMMAE-Vessel and DUCH [2]. Each method is implemented as a combination of a LightningModule ([`cmmae.py`](./src/cmmae.py), [`cmmae_vessel.py`](./src/cmmae_vessel.py), [`duch.py`]((./src/duch.py))), where the actual training happens and losses are applied, and an encoder backbone ([`vit_cmmae.py`](./src/vit_cmmae.py), [`vit_cmmae_vessel.py`](./src/vit_cmmae_vessel.py), [`vit_duch.py`](./src/vit_duch.py)), which is used as the feature extractor. Note that this part only deals with feature extraction and deep hashing modules are added and trained in a second step.

The current workflow trains a CMMAE (using [`cmmae.py`](./src/cmmae.py), [`vit_cmmae.py`](./src/vit_cmmae.py)) on BEN. Parameters for the backbone, the method itself, data-related options, optimizer (scheduler) coefficients can be adjusted in a yaml file, `cfg.yaml`. Four entries _have to_ be completed:
- The training progress is tracked on [Weights & Biases](https://wandb.ai/) (Wandb). To this end, the `wandb.entity` and `wandb.project` fields have to be entered in the `wandb` attribute.
- For training, [BigEarthNet-MM](https://bigearth.net/) is required. The dataloader requires the LMDB format which is explained [here](http://docs.kai-tub.tech/bigearthnet_encoder/intro.html). Finally, the `data.root_dir` should point to the directory containing the LMDB file and `data.split_dir` should point to the directory containing CSV-file splits of the dataset.

To train DUCH on BEN or CMMAE-Vessel on VDD, some parts of the pretraining script have to adjusted:
- In [`main_train.py`](./main_train.py), the correct dataset has to be instantiated. Implementations for both datasets are available, as mentioned in the datasets section. Upon changing the dataset, make sure that the corresponding hyperparameters are set correctly in the `cfg.data.[]` part and paths are updated accordingly.
- In [`main_train.py`](./main_train.py), the correct method has to be instantiated (one of [`cmmae.py`](./src/cmmae.py), [`cmmae_vessel.py`](./src/cmmae_vessel.py), [`duch.py`]((./src/duch.py))). Also here, update the backbone- and method-kwargs according to the implementation at hand.

To illustrate necessary changes in the config file, we provide a template for CMMAE-Vessel on VDD in `cfg_vessel-template.yaml`.

#### Code

To run the pretraining, execute
```bash
> python main_train.py --config-path '/path/to/config/' --config-name 'cfg.yaml'
```
Logs are automatically added to the Wandb project while meta-data and checkpoints are stored under `./trained_models/`.


## Deep Hashing Module

#### Documentation

After pretraining, we can train the deep hashing module on top of feature extractors. The deep hashing module usually comprises fully-connected layers and a _tanh()_ function. The deep hashing module is already part of backbones but may need to be un-commented. The [`main_hashify.py`](./main_hashify.py) script is currently set up for training the deep hashing module of CMMAEs for BEN. Applying it to different methods/datasets can be achieved by exchanging the dataset and instantiating (loading) different models. Note that different models may need the application of different losses, for instance, `calc_zerone_loss` is designed for VDD and some hashing losses require Sentinel-1 (S1) / Sentinel-2 (S2) features while others only work on a single modality.

#### Code

To run the deep hashing module, execute the corresponding script as
```bash
> python main_hashify.py
```


## Evaluation Search By Image-Engines

#### Documentation

To evaluate search by image-engines (a task also denoted as image retrieval), run the [`main_retrieval.py`](./main_retrieval.py) script. The current implementation evaluates CMMAE models. By exchanging the dataset/backbone, other image-retrieval methods can be assessed. Note that some metrics are not informative for binary vessel detection.

The two required flags are
- name of the folder, which contains the model checkpoint to be evaluated
- the GPU device number used for inference.

#### Code

For instance, a model stored under `./trained_models/abcd1234/` can be evaluated with

```bash
python retrieval.py abcd1234 0
```


## Model Weights

We shared model weights suitable for [VDD](https://tubcloud.tu-berlin.de/s/k6TzgWyazPCt4qP), [BEN-DUCH](https://tubcloud.tu-berlin.de/s/mMYbXrqCXcyaMM2) and [BEN-CMMAE](https://tubcloud.tu-berlin.de/s/iMqnGn4tG6XmaEA). To load, follow the code snippet below.

```python
# BEN
cfg = OmegaConf.load('./cfg.yaml')
model = CMMAEBackbone(**cfg.backbone.kwargs)

state_dict = torch.load('./checkpoints/weights_ben.ckpt', map_location="cpu")['state_dict']
model.load_state_dict(state_dict, strict=True)

# VDD
cfg = OmegaConf.load('./cfg.yaml')
model = CMMAEBackboneVessel()

state_dict = torch.load('./checkpoints/weights_vessel.ckpt', map_location="cpu")['state_dict']
model.load_state_dict(state_dict, strict=True)
```


# Acknowledgment

This software was developed by [RSiM](https://rsim.berlin/) of [BIFOLD](https://bifold.berlin) and [TU Berlin](https://tu.berlin).

- [Jakob Hackstein](https://rsim.berlin/team/members/jakob-hackstein)
- [Genc Hoxha](https://rsim.berlin/team/members/genc-hoxha)
- [Begum Demir](https://rsim.berlin/team/members/begum-demir)

For questions, requests and concerns, please contact [Jakob Hackstein via mail](mailto:hackstein@tu-berlin.de)

---

> [1] Hackstein, Jakob, Gencer Sumbul, Kai Norman Clasen, and Begüm Demir. "Exploring Masked Autoencoders for Sensor-Agnostic Image Retrieval in Remote Sensing." arXiv preprint arXiv:2401.07782 (2024).

> [2] Mikriukov, Georgii, Mahdyar Ravanbakhsh, and Begüm Demir. "Deep unsupervised contrastive hashing for large-scale cross-modal text-image retrieval in remote sensing." arXiv preprint arXiv:2201.08125 (2022).

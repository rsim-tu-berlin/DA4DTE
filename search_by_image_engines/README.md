# Search-by-image Engines

In this repository, the software for search-by-image engines is documented. Three scripts execute the main parts of this sub-project:
- Pretraining: `main_pretrain.py`
- Deep Hashing Module: `main_hashify.py`
- Evaluation: `main_retrieval.py`


## Datasets

There are two datasets used for image-to-image use cases, which are [BigEarthNet-MM](https://bigearth.net/)  (BEN) and vessel detection dataset (VDD). In the corresponding folders, implementations of torch datasets are provided. VDD directly operates on tif-files and BEN expects a LMDB file, which can be obtained by following the guidelines [here](http://docs.kai-tub.tech/bigearthnet_encoder/intro.html). For VDD, we create custom query/archive splits by using the `gen_query_archive` notebook.


## Pretraining

#### Documentation

This repository contains three pretraining methods, namely CMMAE, CMMAE-Vessel and DUCH. Each method is implemented as a combination of a LightningModule (`cmmae.py`, `cmmae_vessel.py`, `duch.py`), where the actual training happens and losses are applied, and an encoder backbone (`vit_cmmae.py`, `vit_cmmae_vessel.py`, `vit_duch.py`), which is used as the feature extractor. Note that this part only deals with feature extraction and deep hashing modules are added and trained in a second step.

The current workflow trains a CMMAE (using `cmmae.py`, `vit_cmmae.py`) on BEN. Parameters for the backbone, the method itself, data-related options, optimizer (scheduler) coefficients can be adjusted in a yaml file, `cfg.yaml`. Four entries _have to_ be completed:
- The training progress is tracked on [Weights & Biases](https://wandb.ai/). To this end, the `wandb.entity` and `wandb.project` fields have to be entered in the `wandb` attribute.
- For training, [BigEarthNet-MM](https://bigearth.net/) is required. The dataloader requires the LMDB format which is explained [here](http://docs.kai-tub.tech/bigearthnet_encoder/intro.html). Finally, the `data.root_dir` should point to the directory containing the LMDB file and `data.split_dir` should point to the directory containing CSV-file splits of the dataset.

To train DUCH on BEN or CMMAE-Vessel on VDD, some parts of the pretraining script have to adjusted:
- In main_train.py, the correct dataset has to be instantiated. Implementations for both datasets are available, as mentioned in the datasets section. Upon changing the dataset, make sure that the corresponding hparams are set correctly in the `cfg.data.[]` part and paths are updated accordingly.
- In main_train.py, the correct method has to be instantiated (one of `cmmae.py`, `cmmae_vessel.py`, `duch.py`). Also here, update the backbone- and method-kwargs according to the implementation at hand.

To illustrate necessary changes in the config file, we provide a template for CMMAE-Vessel on VDD in `cfg_vessel-template.yaml`.

#### Code

To run the pretraining, execute
```bash
> python main_pretrain.py --config-path '/path/to/config/' --config-name 'cfg.yaml'
```
Logs are automatically added to the Wandb project, meta-data and checkpoints are stored under `./trained_models/`.


## Deep Hashing Module

#### Documentation

After pretraining, we can train the deep hashing module on top of feature extractors. The deep hashing module usually comprises fully-connected layers and a tanh() function. The deep hashing module is already part of backbones but may need to be un-commented. The `main_hashify.py` script is currently set up for training the deep hashing module of CMMAEs for BEN. Applying it to different methods/datasets can be achieved by exchanging the dataset and instantiating (loading) different models. Note that different models may need the application of different losses, for instance, `calc_zerone_loss` is designed for VDD and some hashing losses require S1/S2 features while others only work on a single modality.

#### Code

To run the deep hashing module, execute the corresponding script as
```bash
> python main_hashify.py
```


## Evaluation Image Retrieval

#### Documentation

To compute image retrieval results, run the `main_retrieval.py` script. The current implementation evaluates CMMAE models. By exchanging the dataset/backbone, other image-retrieval methods can be assessed. Note that some metrics are not informative for binary vessel detection.

The two required flags are
- name of the folder, which contains the model checkpoint to be evaluated
- the GPU device number used for inference.

#### Code

For instance, a model stored under `./trained_models/abcd1234/` can be evaluated with

```bash
python retrieval.py abcd1234 0
```


## Model Weights

We shared model weights VDD and BEN methods. To load follow the code snippet below.

```python
# BEN
cfg = OmegaConf.load(f'./cfg.yaml')
model = CMMAEBackbone(**cfg.backbone.kwargs)

state_dict = torch.load(f'./checkpoints/weights_ben.ckpt', map_location="cpu")['state_dict']
model.load_state_dict(state_dict, strict=True)

# VDD
cfg = OmegaConf.load(f'./cfg.yaml')
model = CMMAEBackboneVessel()

state_dict = torch.load(f'./checkpoints/weights_vessel.ckpt', map_location="cpu")['state_dict']
model.load_state_dict(state_dict, strict=True)
```
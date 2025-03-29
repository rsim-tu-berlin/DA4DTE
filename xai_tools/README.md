# xAI Tools

In this repository, the software for two xAI methods called LRP [1] and BiLRP [2] is contained. The first method explains class-decisions by highlighting the area that was attended to form a classification decision, and the second draws arrows between two images which lead to similar representations in embedding space. We show two implementations according to two different models and datasets. For the vessel-detection dataset, we demonstrate LRP and BiLRP, and for [BigEarthNet-MM](https://bigearth.net/) (BEN), we demonstrate LRP only since BiLRP is computationally too expensive. 

#### Documentation

For the xAI-code, the models to be explained are re-implemented such that they only use layers from [`layers_ours.py`](./xai_vessel/src/layers_ours.py) ([`layers_ours.py`](./xai_bigearthnet/src/layers_ours.py)). Here, the vanilla implementations of torch modules like `Linear`, `MaxPool2D`, `...` are extended by a relevance propagation function, that allows to transfer relevance back from the neuron of a classification head to the input image. We convert our models accordingly and re-use [open-source LRP code](https://github.com/hila-chefer/Transformer-Explainability) to compute explanations. Commonly, other models and methods are easily transformed to the [`layers_ours.py`](./xai_vessel/src/layers_ours.py) by just replacing `import torch.nn as nn` with `from src.layers_ours import *`.

#### Code

This sub-repo contains two folders for LRP and BiLRP for the vessel-dataset and LRP for BEN. The process of obtaining explanations should be straightforward by following these implementations and demo files. The respective model weights which are assumed to be in `./weights/` can be downloaded [here (BEN)](https://tubcloud.tu-berlin.de/s/cq7ydatipHLqaNT) and [here (vessel)](https://tubcloud.tu-berlin.de/s/is3GXDFL8LiLsG3).

## Acknowledgment

This software was developed by [RSiM](https://rsim.berlin/) of [BIFOLD](https://bifold.berlin) and [TU Berlin](https://tu.berlin).

- [Jakob Hackstein](https://rsim.berlin/team/members/jakob-hackstein)
- [Genc Hoxha](https://rsim.berlin/team/members/genc-hoxha)
- [Begum Demir](https://rsim.berlin/team/members/begum-demir)

For questions, requests and concerns, please contact [Jakob Hackstein via mail](mailto:hackstein@tu-berlin.de)

---

> [1] Alexander Binder, Gregoire Montavon, Sebastian Lapuschkin, Klaus-Robert Müller, and Wojciech Samek. Layer-wise relevance propagation for neural networks with local renormalization layers. In International Conference on Artificial Neural Networks, pages 63–71. Springer, 2016

> [2] Eberle, Oliver, Jochen Büttner, Florian Kräutli, Klaus-Robert Müller, Matteo Valleriani, and Grégoire Montavon. "Building and interpreting deep similarity models." IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 2020
# Explainability Tools

In this repository, the software for two explainability methods called LRP and BiLRP is contained. The first method explains class-decisions by highlighting the area that was attended to form a classification decision, and the second draws arrows between two images which lead to similar representations in embedding space. We show two implementations according to two different models/datasets. For the vessel-detection dataset, we demonstrate LRP and BiLRP, and for BigEarthNet, we demonstrate LRP only since BiLRP is computationally too expensive. 

#### Documentation

For the explainability-code, the models to be explained are re-implemented such that they only use layers from `layers_ours.py`. Here, the vanilla implementations of torch modules like `Linear`, `MaxPool2D`, ... are extended by a relevance propagation function, that allows to transfer relevance back from the neuron of a classification head to the input image. We convert our models accordingly and re-use public LRP/BiLRP code to compute explanations. Commonly, other models and methods are easily transformed to the `layers_ours.py` by just replacing `import torch.nn as nn` by `from src.layers_ours import *`.

#### Code

This sub-repo contains two folders for LRP+BiLRP for vessel and LRP for BEN. The process of obtaining explanations should be straightforward by following these implementations and demo files. The respective model weights which are assumed to be in `./weights/` can be downloaded from the shared TUB-Cloud, as they are too big for GitHub.
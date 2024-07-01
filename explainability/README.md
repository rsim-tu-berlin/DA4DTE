# Explainability Tools

In this repository, the software for LRP and BiLRP. The first method explains class-decisions by highlighting the area that was attended to form a decision, and the second draws arrows between two images which lead to similar representations in embedding space. The current implementation is based on vessel detection dataset, but can be extended by replacing the model (weights) and the dataset.

#### Documentation

For this code, the models to be explained are re-implemented such that they only use layers from `layers_ours.py`. Here, the modules of torch implementations of `Linear`, `MaxPool2D`, ... are extended by a relevance propagation function, that allows to go back from the neuron of a classification head to the input image. Commonly, other methods are easily transformed to the `layers_ours.py` by just replacing `import torch.nn as nn` by `from src.layers_ours import *`.

#### Code

This sub-repo contains two notebooks to demonstrate both LRP and BiLRP. The process of obtaining explanations should be straightforward by following these implementations. The model weights which are assumed to be in `./weights/` can be downloaded from the shared TUB-Cloud, as they are too big for GitHub.
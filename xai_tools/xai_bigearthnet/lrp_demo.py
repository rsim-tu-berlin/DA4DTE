import os
import numpy as np
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import tifffile

from src.model import vit_base
from src.LRP import LRP

from xai_misc import (
    get_classwise_explanation,
    raw_sentinel1_to_RGB_format,
    raw_sentinel2_to_RGB_format,
    raw_sentinel12_to_MODEL_format,
)

########################################################
# Initalization of constants, environment variables
# and explainer-module
########################################################

# TODO set CUDA device according to your hardware setup
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-bcfdbd83-be78-533e-a986-25fcc5fa1a6f"
dev = "cuda:0"

# we blend heatmaps and original image in the visualization of
# the explanation - this parameter determines the strength of heatmap
HEATMAP_WEIGHT = 0.5

# initialize backbone and load weights
ckpt_path = "./weights/lrp_weights.ckpt"
backbone = vit_base(
    patch_size=15,
    img_size=120,
    in_chans=12,
    pre_encoder_depth=10,
    shared_encoder_depth=2,
)
state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
ret = backbone.load_state_dict(state, strict=True)
backbone = backbone.to(dev).eval()
print("Loading weights:", ret)

# initialize explainer module and link to backbone model
IMG_SIZE = 120
PATCH_SIZE = 15
NUM_TOKENS = int((IMG_SIZE / PATCH_SIZE) ** 2) + 1
BATCH_SIZE = 1
EYE = torch.eye(NUM_TOKENS).expand(BATCH_SIZE, NUM_TOKENS, NUM_TOKENS).to(dev)

lrp = LRP(backbone, EYE)


########################################################
# Data handling:
# - Load raw Sentinel1 and Sentinel2 image patches (from BEN)
# - Convert to RGB format for visualization
# - Convert to model input format
########################################################

sentinel12_img_path = "./raw_tifs/S2A_MSIL2A_20170613T101031_25_69.tif"
sentinel12_img = tifffile.imread(sentinel12_img_path)

# split stacked S2+S1 image into S1 and S2 parts
sentinel1_img = sentinel12_img[:, :, 10:]
sentinel2_img = sentinel12_img[:, :, :10]

# move from numpy to torch tensor
sentinel1_img = torch.from_numpy(sentinel1_img).float().to(dev)
sentinel2_img = torch.from_numpy(sentinel2_img).float().to(dev)
sentinel12_img = torch.from_numpy(sentinel12_img).float().to(dev)

# Swap axes from 'Height x Width x Channel' -> 'Channel x Height x Width'
sentinel1_img = sentinel1_img.permute(2, 1, 0)
sentinel2_img = sentinel2_img.permute(2, 1, 0)
sentinel12_img = sentinel12_img.permute(2, 1, 0)

sentinel1_img_RGB = raw_sentinel1_to_RGB_format(sentinel1_img)
sentinel2_img_RGB = raw_sentinel2_to_RGB_format(sentinel2_img)
sentinel12_img_MODEL = raw_sentinel12_to_MODEL_format(sentinel12_img)

########################################################
# Generate explanations for an S1 input:

# For this demo, we load example files (stacked S2+S1) from the ./raw_tifs/ folder, as they are provided from BEN.
# The Sentinel image patch is fed into the LRP-explainer module with corresponding modality flag and heatmaps is generated.
########################################################


def explanation_plot(sentinel12_img_MODEL, img_RGB, lrp, backbone, modality):
    """Compute explanations for input and given modality and plots the results.

    Args:
        sentinel12_img_MODEL (torch.Tensor): 12-channel tensor, i.e. normalized stacked S2 + S1 channels
        img_RGB (np.ndarray): 3-channel visualization of the input image
        lrp (_type_):
        backbone (_type_):
        modality (str): either "s2" or "s1"
    """

    # generate classwise explanation
    heatmap_cls_prob_list = get_classwise_explanation(
        lrp, backbone, sentinel12_img_MODEL, modality=modality
    )

    fig, axs = plt.subplots(1, len(heatmap_cls_prob_list) + 1, figsize=(15, 5))
    for ax in axs.ravel():
        ax.axis("off")

    axs[0].imshow(img_RGB)
    axs[0].set_title("Original Input", fontsize=12)

    for i, (heatmap, cls, prob) in enumerate(heatmap_cls_prob_list):
        print(f"Top-{i} (class-names, probs):", cls, prob)

        # blend heatmap and original image
        alpha = np.clip(prob, 0.25, HEATMAP_WEIGHT)
        heatmap_blended = (1 - alpha) * (img_RGB) + alpha * heatmap

        axs[i + 1].imshow(heatmap_blended)
        axs[i + 1].set_title(
            f"{cls[:25] + ('...' if len(cls) > 25 else '')}", fontsize=12
        )

    return fig


print("Demonstrate S1 explanation.")
fig = explanation_plot(sentinel12_img_MODEL, sentinel1_img_RGB, lrp, backbone, "s1")
fig.savefig("s1_clswise_explained.png", bbox_inches="tight")
plt.show()
plt.clf()

print("Demonstrate S2 explanation.")
fig = explanation_plot(sentinel12_img_MODEL, sentinel2_img_RGB, lrp, backbone, "s2")
fig.savefig("s2_clswise_explained.png", bbox_inches="tight")
plt.show()
plt.clf()

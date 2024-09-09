import os
import torch
import matplotlib.pyplot as plt

from src.model import vit_base
from src.utils import get_explanation, load_example_files, generate_caption, imshow
from src.LRP import LRP


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
# Generate explanations for an S1 input:
# For the explanations we need two files:
#   1) the Sentinel image patch (stacked S2 + S1) as a tensor to compute heatmaps
#   2) 8bit image (pixel values in [0,255]) of a RGB-like version of the image patch, to be enriched by the heatmap
#      (can be RGB-channels from S2 or a visualization of S1-channels, e.g. (VV, VH, VV/VH))
#
# For this demo, we load examples files from the ./examples folder via the function load_example_files(...),
# but an actual use case may need to implement a project-specific function to load these files.
#
# The Sentinel image patch is fed into the LRP-explainer module with corresponding modality
# flag and a heatmap is generated. The explanation is then a blended version of heatmap and RGB visual.
########################################################
print("Demonstrate S1 explanation.")
MODALITY = "s1"

# load example files
s1_patch_name, s2_patch_name = (
    "S1B_IW_GRDH_1SDV_20170613T044822_34VER_69_34",
    "S2A_MSIL2A_20170613T101032_69_34",
)
sentinel_img, s1_rgb, s2_rgb = load_example_files(s1_patch_name, s2_patch_name)
sentinel_img = sentinel_img.to(dev)
print(f"The sentinel_img has shape {sentinel_img.shape}, type {sentinel_img.dtype}")
print(f"Corresponding RGB versions of S1/S2 have shape {s1_rgb.shape}")


# plot original S1, without heatmap
imshow(s1_rgb)
plt.title("Original Input")
plt.show()
plt.savefig("s1_original.png", bbox_inches="tight")
plt.clf()

# generate explanation
heatmap, topk_classes = get_explanation(lrp, backbone, sentinel_img, modality=MODALITY)

# blend heatmap and original image
heatmap_blended = (1 - HEATMAP_WEIGHT) * (s1_rgb / 255) + HEATMAP_WEIGHT * heatmap
print("Top-k (class-names, probs):", topk_classes)


# NOTE choose a caption that fits best into the UI / workflow
caption = generate_caption(topk_classes, add_probability=True)
# caption = generate_caption(topk_classes, add_probability=False)
# caption = {'label': 'Relevant Areas', 'fontsize': 14}

# plot explaination
imshow(heatmap_blended)
plt.title(**caption)
plt.show()
plt.savefig("s1_explained.png", bbox_inches="tight")
plt.clf()

########################################################
# Generate explanations for an S2 input, similar to S1-demonstration above but
# we adjust variables to select s2-explanation
########################################################
print("\nDemonstrate S2 explanation.")
MODALITY = "s2"

# load example files
s1_patch_name, s2_patch_name = (
    "S1B_IW_GRDH_1SDV_20170717T064605_29UPV_70_10",
    "S2A_MSIL2A_20170717T113321_70_10",
)
sentinel_img, s1_rgb, s2_rgb = load_example_files(s1_patch_name, s2_patch_name)
sentinel_img = sentinel_img.to(dev)
print(f"The sentinel_img has shape {sentinel_img.shape}, type {sentinel_img.dtype}")
print(f"Corresponding RGB versions of S1/S2 have shape {s1_rgb.shape}")

# plot original S1, without heatmap
imshow(s2_rgb)
plt.title("Original Input")
plt.show()
plt.savefig("s2_original.png", bbox_inches="tight")
plt.clf()

# generate explanation
heatmap, topk_classes = get_explanation(lrp, backbone, sentinel_img, modality=MODALITY)

# blend heatmap and original image
heatmap_blended = (1 - HEATMAP_WEIGHT) * (s2_rgb / 255) + HEATMAP_WEIGHT * heatmap
print("Top-k (class-names, probs):", topk_classes)

# NOTE choose a caption that fits best into the UI / workflow
caption = generate_caption(topk_classes, add_probability=True)
# caption = generate_caption(topk_classes, add_probability=False)
# caption = {'label': 'Relevant Areas', 'fontsize': 14}

# plot explaination
imshow(heatmap_blended)
plt.title(**caption)
plt.show()
plt.savefig("s2_explained.png", bbox_inches="tight")
plt.clf()

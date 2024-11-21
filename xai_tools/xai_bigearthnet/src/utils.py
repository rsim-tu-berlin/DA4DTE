import torch
import torchvision

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.layers_ours import *

# cls index to class name
valid_labels_classification = [
    "Agro-forestry areas",  # 0
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",  # 5
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with\n    significant areas "
    "of natural vegetation",  # 10
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",  # 15
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric",
]

# normalization, has to be applied to all samples fed into models
normalize = torchvision.transforms.Normalize(
    mean=[
        429.9430203,
        614.21682446,
        590.23569706,
        2218.94553375,
        950.68368468,
        1792.46290469,
        2075.46795189,
        1594.42694882,
        1009.32729131,
        2266.46036911,
        -19.29044597721542,
        -12.619993741972035,
    ],
    std=[
        572.41639287,
        582.87945694,
        675.88746967,
        1365.45589904,
        729.89827633,
        1096.01480586,
        1273.45393088,
        1079.19066363,
        818.86747235,
        1356.13789355,
        5.464428464912864,
        5.115911777546365,
    ],
)


def get_explanation(lrp, model, img, modality, k=3, cmap="jet"):
    """Compute heatmap for given img and specified modality. In addition, the
    top-k classes + probabilities are returned for eventual visualiation purposes.

    Args:
        lrp (LRP): LRP explainer module
        model (nn.Module): CSMAE
        img (tensor): Sentinel image (S2 + S1 stacked)
        modality (str): modality of img, either 's1' or 's2'
        k (int, optional): Defaults to 3.
        cmap (str, optional): Defaults to "jet".

    Returns:
        - Heatmap of relevant areas for feature extraction
        - List of tuples with topk-many elements of (class-names, probabilities)
    """

    cmap = plt.get_cmap(cmap)
    batch = img.unsqueeze(0)

    logits = getattr(model, f"forward_{modality}")(batch).squeeze(0)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    topk_class_values, topk_class_indices = probs.topk(k, dim=0)
    topk_class_values = topk_class_values.detach().tolist()
    topk_class_names = [valid_labels_classification[x] for x in topk_class_indices]

    mask = np.zeros((120, 120))
    for j, class_index in enumerate(topk_class_indices):
        explained = generate_lrp_mask(lrp, modality, img, class_indices=[class_index])
        mask += explained

    mask /= mask.max()
    heatmap = cmap(mask)[:, :, :3]

    return heatmap, list(zip(topk_class_names, topk_class_values))


def load_example_files(s1_patch_name, s2_patch_name):
    """Helper function for this demo. Load files to be used in the LRP demo.

    Args:
        patch_name (str): patch-name

    Returns: Tuple of three elements:
        - sentinel_img (torch.tensor): 12-channel tensor, i.e. stacked S2 + S1 channels
        - s1_rgb (np.array): visualization of S1 channels (RGB version of S1)
        - s2_rgb (np.array): visualization of S2 channels (RGB version of S2)
    """

    def load_npy_into_torch(file_name):
        return normalize(
            torch.from_numpy(np.load(f"./examples/{file_name}.npy")).permute(2, 1, 0)
        )

    def load_png(file_name):
        return np.array(Image.open(f"./examples/{file_name}.png"))

    sentinel_img = load_npy_into_torch(s2_patch_name)

    s1_rgb, s2_rgb = load_png(s1_patch_name), load_png(s2_patch_name)

    return sentinel_img, s1_rgb, s2_rgb


def generate_caption(topk_classes, add_probability=True):
    """Generate cpations from topk-(class-name,prob) elements.

    Args:
        topk_classes (List[Tuples]): List with (class-name, probability) elements
        add_probability (bool, optional): Defaults to True.

    Returns:
        Dict: dict to be fed into title() function of plt
    """

    if add_probability:
        title = [
            f"{i}) {cls} ({prob*100:.2f})%"
            for i, (cls, prob) in enumerate(topk_classes, 1)
        ]
    else:
        title = [f"{i}) {cls}" for i, (cls, _) in enumerate(topk_classes, 1)]

    caption = "\n".join(title)

    return {"label": caption, "loc": "left"}


def imshow(img):
    """For convenience: visualization of images always without axis"""
    plt.imshow(img)
    plt.axis("off")


def generate_lrp_mask(
    lrp, modality, original_image, class_indices, use_thresholding=False
):
    r_list = []

    for class_index in class_indices:
        r = lrp.run_relprop(
            modality,
            original_image.unsqueeze(0).cuda(),
            method="transformer_attribution",
            index=class_index,
        ).detach()
        r = r.reshape(1, 1, 8, 8)
        r = torch.nn.functional.interpolate(r, scale_factor=15, mode="bilinear")
        r = r.reshape(120, 120).data.cpu().numpy()
        r = (r - r.min()) / (r.max() - r.min())

        if use_thresholding:
            r = r * 255
            r = r.astype(np.uint8)
            ret, r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            r[r == 255] = 1

        r_list.append(r)

    r = np.mean(r_list, axis=0)
    return r

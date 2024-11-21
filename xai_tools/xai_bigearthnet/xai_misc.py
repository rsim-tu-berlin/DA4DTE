import torch
from src.utils import valid_labels_classification, generate_lrp_mask
import numpy as np
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

MEAN = [
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
]

STD = [
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
]


def get_classwise_explanation(lrp, model, img, modality, k=3, cmap="jet"):
    cmap = plt.get_cmap(cmap)
    batch = img.unsqueeze(0)

    logits = getattr(model, f"forward_{modality}")(batch).squeeze(0)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    topk_class_values, topk_class_indices = probs.topk(k, dim=0)
    topk_class_values = topk_class_values.detach().tolist()
    topk_class_names = [valid_labels_classification[x] for x in topk_class_indices]

    heatmap_per_class = []

    mask = np.zeros((120, 120))
    for j, class_index in enumerate(topk_class_indices):
        explained = generate_lrp_mask(lrp, modality, img, class_indices=[class_index])
        mask += explained

        heatmap_per_class.append(cmap(explained)[:, :, :3])

    return list(zip(np.array(heatmap_per_class), topk_class_names, topk_class_values))


def raw_sentinel1_to_RGB_format(sentinel1: torch.Tensor) -> np.ndarray:
    """Receives a sentinel1 tensor and returns a 3-channel visualization of it in HxWxC order."""

    assert sentinel1.shape[0] == 2, f"Expect two channels but got {sentinel1.shape[0]}"
    assert torch.is_tensor(sentinel1), f"Expect tensor but got {type(sentinel1)}"

    sentinel1 = sentinel1.permute(1, 2, 0).detach().cpu().numpy()

    def s(x):
        return (x + abs(x.min())) / (abs(x.min()) + abs(x.max()))

    img_vh = s((sentinel1[:, :, [0]] + 19.29) / 5.46)  # 0
    img_vv = s((sentinel1[:, :, [1]] + 12.61) / 5.11)  # 1
    img_vvvh = s(img_vh - img_vv)

    img_vvvh = np.dstack([img_vv, img_vh, img_vvvh])
    img_vvvh = np.clip(img_vvvh * 1.1, 0, 1)

    return img_vvvh


def raw_sentinel2_to_RGB_format(sentinel2: torch.Tensor) -> np.ndarray:
    """Receives a sentinel2 tensor and returns a 3-channel visualization of it in HxWxC order."""

    assert sentinel2.shape[0] == 10, f"Expect ten channels but got {sentinel2.shape[0]}"
    assert torch.is_tensor(sentinel2), f"Expect tensor but got {type(sentinel2)}"

    sentinel2 = sentinel2.permute(1, 2, 0).detach().cpu().numpy()
    sentinel_RGB = sentinel2[:, :, [2, 1, 0]]

    return np.clip(sentinel_RGB / 1000, 0, 1)


def raw_sentinel12_to_MODEL_format(
    sentinel12: torch.Tensor, dev="cuda:0"
) -> torch.Tensor:
    """Receives a sentinel2+1 tensor and a normalized tensor to be used in models."""

    # We assume bands in this order
    # "B02","B03","B04","B08","B05","B06","B07","B11","B12","B8A","VH","VV",
    mean = torch.Tensor(MEAN).to(dev)
    std = torch.Tensor(STD).to(dev)
    return transforms.Normalize(mean, std)(sentinel12)

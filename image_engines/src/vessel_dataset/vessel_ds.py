import numpy as np
import os
from pathlib import Path
import json
import tifffile as tiff

import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch

from src.utils import Messages


class VesselDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split_dir,
        split="query",
        transform=None,
        max_img_idx=None,
        img_size=(4, 128, 128),
        include_idx=False,
        include_patch_name=False,
    ):
        """Implementation for the vesel detection dataset, compatible for torch dataloaders.

        Args:
            root_dir (str): Path that contains 'vessels_dataset_annotations.json' and a folder called 'tif', where all images are stored.
            split_dir (str): Path that contains splits called 'query_patches.json', 'archive_patches.json', 'zero_patches.json'. See notebook
            for generation of these json files.
            transform (torch.transforms, optional): Transformation applied to images. Defaults to None.
            max_img_idx (int, optional): Maximum number of images in the dataset. Defaults to None.
            img_size (tuple, optional): Defaults to (4, 128, 128).
            include_idx (bool, optional): Whether to return the img_idx. Defaults to False.
            include_patch_name (bool, optional): Whether to return the patch-name (filename). Defaults to False.
        """

        super().__init__()
        self.root_dir = root_dir
        self.tif_dir = f"{root_dir}/tif"
        self.json_file = f"{root_dir}/vessels_dataset_annotations.json"
        with open(self.json_file) as f:
            d = json.load(f)
            self.annotations_dict = d

        self.split = split
        self.split_dir = split_dir

        self.transform = transform
        self.img_size = img_size
        self.include_idx = include_idx
        self.include_patch_name = include_patch_name

        self.read_channels = img_size[0]

        # Select all patches specified in splits/query_patches.json
        if self.split == "query":
            query_patches_json = f"{self.split_dir}/query_patches.json"
            with open(query_patches_json) as json_data:
                d = json.load(json_data)
                query_patches = d["query_patches"]
                json_data.close()
            self.patches = [
                Path(file).stem
                for file in Path(self.tif_dir).glob("*.tif")
                if Path(file).stem in query_patches
            ]

        # Select all patches specified in splits/archive_patches.json
        elif self.split == "archive":
            archive_patches_json = f"{self.split_dir}/archive_patches.json"
            with open(archive_patches_json) as json_data:
                d = json.load(json_data)
                archive_patches = d["archive_patches"]
                json_data.close()
            self.patches = [
                Path(file).stem
                for file in Path(self.tif_dir).glob("*.tif")
                if Path(file).stem in archive_patches
            ]

        elif self.split == "all":
            zero_patches_json = f"{self.split_dir}/zero_patches.json"
            with open(zero_patches_json) as json_data:
                d = json.load(json_data)
                zero_patches = d["zero_patches"]
                json_data.close
            self.patches = [
                Path(file).stem
                for file in Path(self.tif_dir).glob("*.tif")
                if Path(file).stem not in zero_patches
            ]
        else:
            assert False, "False split parameter '{self.split}'"

        # sort list for reproducibility
        self.patches.sort()
        Messages.hint(f"    {self.split} {len(self.patches)} patches indexed")

        if max_img_idx is None and self.split not in [
            "all",
            "balanced",
            "train",
            "query",
            "archive",
        ]:
            max_img_idx = 32

        if (
            max_img_idx is not None
            and max_img_idx < len(self.patches)
            and max_img_idx != -1
        ):
            self.patches = self.patches[:max_img_idx]
        Messages.hint(f"    {self.split} {len(self.patches)} filtered patches indexed")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        key = self.patches[idx]
        img = tiff.imread(f"{self.tif_dir}/{key}.tif").astype(np.float32)

        assert np.any(img), "Image must have non-zero elements."

        d = self.annotations_dict[key]
        num_bboxes = 0

        assert "annotations" in d
        bboxes = d["annotations"]
        num_bboxes = len(bboxes)
        assert num_bboxes < 142

        idx = int(num_bboxes > 0)
        label = torch.zeros(2)
        label[idx] = 1

        if img is None:
            print(f"Cannot load {key}")
            raise ValueError

        img = torch.as_tensor(np.array(img, copy=True))
        img = img.permute(2, 0, 1)

        if self.transform:
            img = self.transform(img)

        if self.include_patch_name:
            return (
                (idx, img, label, key)
                if self.split in ["train", "query", "archive", "balanced"]
                or self.include_idx
                else (img, label, key)
            )
        else:
            return (
                (idx, img, label)
                if self.split in ["train", "query", "archive", "balanced"]
                or self.include_idx
                else (img, label)
            )

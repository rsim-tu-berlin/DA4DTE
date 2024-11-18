import argparse
from .base_config import BaseConfig

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--dataset",
    default="ucm",
    help="ucm or rsicd or vessel_dataset",
    type=str,
    metavar="DATASET_NAME",
)
parser.add_argument(
    "--img-aug",
    default="no_aug",
    type=str,
    metavar="IMG_AUG_SET",
    help="image transform set: see 'image_aug_transform_sets' variable for available sets",
)
parser.add_argument(
    "--crop-size",
    default=[220, 220],
    nargs=2,
    type=int,
    metavar=("CROP_H", "CROP_W"),
    help="crop size for 'center_crop' and 'random_crop'",
)
parser.add_argument(
    "--rot-deg",
    default=[-10, -5],
    nargs=2,
    type=int,
    metavar=("ROT_DEG_MIN", "ROT_DEG_MAX"),
    help="random rotation degrees range for 'rotation_cc'",
)
parser.add_argument(
    "--blur-val",
    default=[3.0, 3.0, 1.1, 1.3],
    nargs=4,
    type=float,
    help="gaussian blur parameters for 'blur_cc'",
    metavar=("KERNEL_W", "KERNEL_H", "SIGMA_MIN", "SIGMA_MAX"),
)
parser.add_argument(
    "--jit-str",
    default=[0.5],
    nargs=1,
    type=float,
    metavar="JITTER_STRENGTH",
    help="color jitter strength for 'jitter_cc'",
)
# parser.add_argument('--gpu', default=0,  type=str, help='selecte between 0,1,2,3,4,5,6:0,6:1,6:2,6:3,7:0,7:1'
#                     )

args = parser.parse_args()

dataset = args.dataset
img_aug = args.img_aug

acs = args.crop_size
ard = args.rot_deg
abv = args.blur_val
ajs = args.jit_str


class ConfigImgPrep(BaseConfig):
    if dataset == "ucm":
        image_emb_batch_size = 525  # 1/4 of dataset
        image_folder_preload = False  # Load all images before running (will be loading on request otherwise)
        image_aug_number = 1  # number generated augmented images

    if dataset == "rsicd":
        image_emb_batch_size = 500
        image_folder_preload = False  # Load all images before running (will be loading on request otherwise)
        image_aug_number = 1  # number generated augmented images

    if dataset == "vessel_dataset":
        image_emb_batch_size = 256
        image_folder_preload = False  # Load all images before running (will be loading on request otherwise)
        image_aug_number = 10  # number generated augmented images

    img_aug_set = (
        img_aug  # name of image transform set to select, check image_aug_transform_sets
    )
    # transforms will be applied in the dictionary order
    # 'blur' gaussian blur with given kernel size and sigma range
    # 'rotation' random rotation in given ranges of degrees
    # 'affine' random affine transform in given ranges of degrees
    # 'center_crop' center crop with given dimensions
    # 'random_crop' random crop with given dimensions
    # 'jitter' color jittering strength
    # 'each_img_random' - random transform from list for each image
    # 'no_aug' - no augmentation (only resize to 224x224)

    image_aug_transform_sets = {
        "center_crop_only": {"center_crop": tuple(acs)},
        "random_crop_only": {"random_crop": tuple(acs)},
        "aug_center": {
            "blur": ((3, 3), (1.1, 1.3)),
            "rotation": [(-10, -5)],  # [(-10, -5), (5, 10)]
            "center_crop": tuple(acs),
        },
        "aug_random": {
            "blur": ((3, 3), (1.1, 1.3)),
            "rotation": [(-10, -5)],  # [(-10, -5), (5, 10)]
            "random_crop": tuple(acs),
        },
        "rotation_cc": {"rotation": [tuple(ard)], "center_crop": tuple(acs)},
        "blur_cc": {
            "blur": (tuple([int(i) for i in abv[:2]]), tuple(abv[2:])),
            "center_crop": tuple(acs),
        },
        "jitter_cc": {"jitter": ajs[0], "center_crop": tuple(acs)},
        "each_img_random": ["rotation_cc", "blur_cc", "jitter_cc"],
        "no_aug": {},
    }

    if img_aug_set == "rotation_cc":
        image_emb_aug_file = "/data/image_emb_{}_aug_{}_{}_{}.h5".format(
            dataset.upper(), img_aug_set, ard[0], ard[1]
        )
    elif img_aug_set == "blur_cc":
        image_emb_aug_file = "/data/image_emb_{}_aug_{}_{}_{}.h5".format(
            dataset.upper(),
            img_aug_set,
            "_".join([str(int(i)) for i in abv[:2]]),
            "_".join([str(i) for i in abv[2:]]),
        )
    elif img_aug_set == "jitter_cc":
        image_emb_aug_file = "/data/image_emb_{}_aug_{}_{}.h5".format(
            dataset.upper(), img_aug_set, ajs[0]
        )
    else:
        image_emb_aug_file = "/data/image_emb_{}_aug_{}.h5".format(
            dataset.upper(), img_aug_set
        )
    image_emb_file = "/data/image_emb_{}_aug.h5".format(dataset.upper())

    def __init__(self, args):
        super(ConfigImgPrep, self).__init__(args)


cfg = ConfigImgPrep(args)

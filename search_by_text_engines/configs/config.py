import argparse
from .base_config import BaseConfig
from utils import get_image_file_names


parser = argparse.ArgumentParser(description='')
parser.add_argument('--test', default=False, help='run test', action='store_true')
parser.add_argument('--bit', default=128, help='hash code length', type=int)
parser.add_argument('--model', default='DUCH', help='model type', type=str)
parser.add_argument('--epochs', default=100, help='training epochs', type=int)
parser.add_argument('--tag', default='test', help='model tag (for save path)', type=str)
parser.add_argument('--dataset', default='ucm', help='ucm or rsicd', type=str)
parser.add_argument('--preset', default='default', help='data presets, see available in config.py', type=str)
parser.add_argument('--alpha', default=0.01, help='alpha hyperparameter (La)', type=float) #0.01
parser.add_argument('--beta', default=0.001, help='beta hyperparameter (Lq)', type=float)
parser.add_argument('--gamma', default=0.01, help='gamma hyperparameter (Lbb)', type=float)
parser.add_argument('--contrastive-weights', default=[1.0, 1.0, 1.0], type=float, nargs=3,
                    metavar=('INTER', 'INTRA_IMG', 'INTRA_TXT'), help='contrastive loss component weights')

parser.add_argument('--img-aug-emb', default=None, type=str, help='overrides augmented image embeddings file (u-curve)')

args = parser.parse_args()

dataset = args.dataset
preset = args.preset

alpha = args.alpha
beta = args.beta
gamma = args.gamma
contrastive_weights = args.contrastive_weights


class ConfigModel(BaseConfig):
    preset = preset.lower()

    if preset == 'no_aug':
        # default for texts
        image_emb_for_model = "/data/image_emb_{}_aug_center_crop_only.h5".format(dataset.upper())
        caption_emb_for_model = "/data/caption_emb_{}_aug.h5".format(dataset.upper())

        image_emb_aug_for_model = "/data/image_emb_{}_aug_center_crop_only.h5".format(dataset.upper())
        caption_emb_aug_for_model = "/data/caption_emb_{}_aug.h5".format(dataset.upper())

        dataset_json_for_model = "/data/augmented_{}.json".format(dataset.upper())
    elif preset == 'each_img_random':
        # default for texts
        image_emb_for_model = "/data/image_emb_{}_aug_center_crop_only.h5".format(dataset.upper())
        caption_emb_for_model = "/data/caption_emb_{}_aug.h5".format(dataset.upper())

        image_emb_aug_for_model = "/data/image_emb_{}_aug_each_img_random.h5".format(dataset.upper())
        caption_emb_aug_for_model = "/data/caption_emb_{}_aug_rb.h5".format(dataset.upper())

        dataset_json_for_model = "/data/augmented_{}.json".format(dataset.upper())
    elif preset in ['img_aug_center', 'baseline', 'img_aug_center_txt_rb', 'default']:
        # default for texts
        print ('default set up')
        image_emb_for_model = "/data/image_emb_{}_aug_center_crop_only.h5".format(dataset.upper())
        caption_emb_for_model = "/data/caption_emb_{}_aug.h5".format(dataset.upper())

        image_emb_aug_for_model = "/data/image_emb_{}_aug_aug_center.h5".format(dataset.upper())
        caption_emb_aug_for_model = "/data/caption_emb_{}_aug_rb.h5".format(dataset.upper())

        dataset_json_for_model = "/data/augmented_{}.json".format(dataset.upper())
    elif preset == 'img_aug_center_txt_bt_chain':
        # default for texts
        image_emb_for_model = "/data/image_emb_{}_aug_center_crop_only.h5".format(dataset.upper())
        caption_emb_for_model = "/data/caption_emb_{}_aug.h5".format(dataset.upper())

        image_emb_aug_for_model = "/data/image_emb_{}_aug_aug_center.h5".format(dataset.upper())
        caption_emb_aug_for_model = "/data/caption_emb_{}_aug_bt_chain.h5".format(dataset.upper())

        dataset_json_for_model = "/data/augmented_{}.json".format(dataset.upper())
    elif preset == 'img_aug_random':
        image_emb_for_model = "/data/image_emb_{}_aug_center_crop_only.h5".format(dataset.upper())
        caption_emb_for_model = "/data/caption_emb_{}_aug.h5".format(dataset.upper())

        image_emb_aug_for_model = "/data/image_emb_{}_aug_aug_random.h5".format(dataset.upper())
        caption_emb_aug_for_model = "/data/caption_emb_{}_aug_rb.h5".format(dataset.upper())

        dataset_json_for_model = "/data/augmented_{}.json".format(dataset.upper())
    elif preset == 'img_aug_random_only':
        image_emb_for_model = "/data/image_emb_{}_aug_center_crop_only.h5".format(dataset.upper())
        caption_emb_for_model = "/data/caption_emb_{}_aug.h5".format(dataset.upper())

        image_emb_aug_for_model = "/data/image_emb_{}_aug_random_crop_only.h5".format(dataset.upper())
        caption_emb_aug_for_model = "/data/caption_emb_{}_aug_rb.h5".format(dataset.upper())

        dataset_json_for_model = "/data/augmented_{}.json".format(dataset.upper())
    elif preset == 'txt_aug_rb':
        # default for images
        image_emb_for_model = "/data/image_emb_{}_aug_center_crop_only.h5".format(dataset.upper())
        caption_emb_for_model = "/data/caption_emb_{}_aug.h5".format(dataset.upper())

        image_emb_aug_for_model = "/data/image_emb_{}_aug_aug_center.h5".format(dataset.upper())
        caption_emb_aug_for_model = "/data/caption_emb_{}_aug_rb.h5".format(dataset.upper())

        dataset_json_for_model = "/data/augmented_{}.json".format(dataset.upper())
    elif preset == 'txt_aug_bt_prob':
        image_emb_for_model = "/data/image_emb_{}_aug_center_crop_only.h5".format(dataset.upper())
        caption_emb_for_model = "/data/caption_emb_{}_aug.h5".format(dataset.upper())

        image_emb_aug_for_model = "/data/image_emb_{}_aug_aug_center.h5".format(dataset.upper())
        caption_emb_aug_for_model = "/data/caption_emb_{}_aug_bt_prob.h5".format(dataset.upper())

        dataset_json_for_model = "/data/augmented_{}.json".format(dataset.upper())
    elif preset == 'txt_aug_bt_chain':
        image_emb_for_model = "/data/image_emb_{}_aug_center_crop_only.h5".format(dataset.upper())
        caption_emb_for_model = "/data/caption_emb_{}_aug.h5".format(dataset.upper())

        image_emb_aug_for_model = "/data/image_emb_{}_aug_aug_center.h5".format(dataset.upper())
        caption_emb_aug_for_model = "/data/caption_emb_{}_aug_bt_chain.h5".format(dataset.upper())

        dataset_json_for_model = "/data/augmented_{}.json".format(dataset.upper())
    else:
        raise Exception('Nonexistent preset: {}'.format(preset))

    if args.img_aug_emb is not None:
        image_emb_aug_for_model = args.img_aug_emb

    if dataset == 'ucm':
        label_dim = 21

    if dataset == 'rsicd':
        label_dim = 31
    if dataset == 'vessel_dataset':
        label_dim = 52

# sto cambiando da qui; Provare ad avere un unico dataset loader
    # dataset_train_split = 0.90  # part of all data, that will be used for training
    # # (1 - dataset_train_split) - evaluation data
    # dataset_query_split = 0.10  # part of evaluation data, that will be used for query
    # # (1 - dataset_train_split) * (1 - dataset_query_split) - retrieval data

    model_type = 'DUCH'
    batch_size = 256
    image_dim = 512
    text_dim = 768
    hidden_dim = 1024 * 4
    hash_dim = 128

    lr = 0.0001
    max_epoch = 100
    valid = True  # validation
    valid_freq = 100  # validation frequency (epochs)
    alpha = alpha  # adv loss
    beta = beta  # quant loss
    gamma = gamma  # bb loss
    contrastive_weights = contrastive_weights  # [inter, intra_img, intra_txt]

    retrieval_map_k = 8

    tag = 'test'

    def __init__(self, args):
        super(ConfigModel, self).__init__(args)
        self.test = args.test
        self.hash_dim = args.bit
        self.model_type = args.model
        self.max_epoch = args.epochs
        self.tag = args.tag


cfg = ConfigModel(args)

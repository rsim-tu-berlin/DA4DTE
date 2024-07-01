import argparse
from .base_config import BaseConfig

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', default='ucm', help='ucm or rsicd or vessel_dataset', type=str, metavar='DATASET_NAME')
parser.add_argument('--txt-aug', default='backtranslation-prob', type=str, metavar='TXT_AUG_TYPE',
                    help="image transform set: 'rule-based', 'backtranslation-prob', 'backtranslation-chain'")

args = parser.parse_args()

dataset = args.dataset
txt_aug = args.txt_aug


class ConfigTxtPrep(BaseConfig):
    if dataset == 'ucm':
        caption_token_length = 64  # use hardcoded number, max token length for clean data is 26

    if dataset == 'rsicd':
        caption_token_length = 64  # use hardcoded number, max token length for clean data is 40
    if dataset == 'vessel_dataset':
        caption_token_length = 75  # use hardcoded number, max token length for clean data is 74

    caption_hidden_states = 4  # Number of last BERT's hidden states to use
    caption_hidden_states_operator = 'sum'  # "How to combine hidden states: 'sum' or 'concat'"

    caption_aug_rb_glove_sim_threshold = 0.65
    caption_aug_rb_bert_score_threshold = 0.75

    # caption_aug_type - augmentation method 'prob' or 'chain'
    # caption_aug_method:
    #   'prob' - sentence translated to a random lang with prob proportional to weight from captions_aug_bt_lang_weights
    #   'chain' - translates in chain en -> lang1 -> lang2 -> ... -> en (weight values are ignored)
    if txt_aug.startswith('rule'):
        caption_aug_type = 'rule-based'
        caption_aug_method = None
    else:
        caption_aug_type = 'backtranslation'
        if txt_aug.endswith('prob'):
            caption_aug_method = 'prob'
        else:
            caption_aug_method = 'chain'

    # available models and laguages: https://huggingface.co/Helsinki-NLP
    # en -> choice(languages) -> en; choice_probability_for_language = lang_weight / sum(lang_weights)
    caption_aug_bt_lang_weights = {'es': 1, 'de': 1}  # {lang1: lang_weight1, lang2: lang_weight2, ...}
    # caption_aug_bt_lang_weights = {'ru': 1, 'bg': 1}

    caption_aug_dataset_json = "/data/augmented_{}.json".format(dataset.upper())
    caption_emb_file = "/data/caption_emb_{}_aug.h5".format(dataset.upper())
    caption_emb_aug_file_rb = "/data/caption_emb_{}_aug_rb.h5".format(dataset.upper())
    caption_emb_aug_file_bt_prob = "/data/caption_emb_{}_aug_bt_prob.h5".format(dataset.upper())
    caption_emb_aug_file_bt_chain = "/data/caption_emb_{}_aug_bt_chain.h5".format(dataset.upper())

    def __init__(self, args):
        super(ConfigTxtPrep, self).__init__(args)


cfg = ConfigTxtPrep(args)

from utils import read_json, write_json, get_captions
from configs.config_txt_aug import cfg

from transformers import MarianMTModel, MarianTokenizer
import copy
import nltk
import gensim.downloader as api
from bert_score import BERTScorer
from tqdm import tqdm
import numpy as np
import random
import torch
import gc
import pickle
import os


def remove_tokens(data):
    """
    Removes 'tokens' key from caption record, if exists; halves the size of the file

    :param data: original data
    :return: data without tokens
    """
    for img in data['images']:
        for sent in img['sentences']:
            try:
                sent.pop("tokens")
            except:
                pass
    return data


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def untokenize(tokens):
    return " ".join(tokens)


def count_total_failed_aug(captions, aug_captions):
    counter = 0
    for c, a in zip(captions, aug_captions):
        c = untokenize(tokenize(c))
        if c == a:
            counter += 1
    return counter / len(captions)


def count_total_failed_aug_per_class(captions, aug_captions, cls):
    counts = []
    captions = [captions[i:i + 500] for i in range(0, len(captions), 500)]
    aug_captions = [aug_captions[i:i + 500] for i in range(0, len(aug_captions), 500)]
    for cap, aug_cap in zip(captions, aug_captions):
        failed_aug = count_total_failed_aug(cap, aug_cap)
        counts.append(failed_aug)
    out_dict = {}
    for k, v in cls.items():
        out_dict[v] = counts[int(k)]
    return out_dict


def check_aug_distribution():
    aug_data = read_json(cfg.caption_aug_dataset_json)
    cls = aug_data['classes']
    captions, aug_captions_rb, aug_captions_bt_prob, aug_captions_bt_chain = get_captions(aug_data)
    if cfg.caption_aug_type == 'backtranslation':
        if cfg.caption_aug_method == 'prob':
            aug_captions = aug_captions_bt_prob
        else:
            aug_captions = aug_captions_bt_chain
    else:
        aug_captions = aug_captions_rb
    total_failed_aug = count_total_failed_aug(captions, aug_captions)
    total_failed_aug_per_class = count_total_failed_aug_per_class(captions, aug_captions, cls)

    print("Total failed augmentations: {:.3f}%".format(total_failed_aug * 100))
    print("Total failed augmentations per class: ", total_failed_aug_per_class)


class TextAugmenterRuleBased:

    def __init__(self, glove_sim_threshold, bert_score_threshold):
        print('Initializing RULE-BASED text AUGMENTER')

        self.bert_score_threshold = bert_score_threshold
        self.glove_sim_threshold = glove_sim_threshold

        print("Loading GloVe")
        self.model = api.load("glove-wiki-gigaword-200")
        print("Loading POS tagger")
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        print("Loading BERTScorer")
        self.scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        print (("Loading BERTScorer"))

    def augment_data(self, clean_data):

        augmented_data = copy.deepcopy(clean_data)

        for img in tqdm(augmented_data['images']):
            for sent in img['sentences']:
                tokens = tokenize(sent['raw'])
                processed_tokens = self.process_tokens(tokens)
                processed_setnence = untokenize(processed_tokens)
                sent['aug_rb'] = processed_setnence

        return augmented_data

    def process_tokens(self, tokens):

        new_tokens = []
        pos_tags = nltk.pos_tag(tokens)

        for idx, (token, tag) in enumerate(pos_tags):
            if tag in ['NN', 'NNS', 'NNP', 'NNPS']:  # noun
                replacement = self.replace_token(idx, tokens, tag)
                new_tokens.append(replacement)
            elif tag == 'EX' and pos_tags[idx + 1][1] == 'VBZ':  # there is -> it is
                replacement = 'it'
                new_tokens.append(replacement)
            elif tag in ['VB', 'VBD', 'VBG', 'VBN']:  # verb
                replacement = self.replace_token(idx, tokens, tag)
                new_tokens.append(replacement)
            else:
                new_tokens.append(token)
        return new_tokens

    def replace_token(self, idx, tokens, tag):
        similar_tokens = self.get_similar_tokens(tokens[idx])
        replacement = self.check_similars(idx, tokens, similar_tokens, tag)
        if replacement is None:
            return tokens[idx]
        else:
            return replacement

    def get_similar_tokens(self, token):
        try:
            return self.model.most_similar(token)
        except:
            return None

    def check_similars(self, idx, tokens, similar_tokens, tag):

        if similar_tokens is None:
            return None

        candidate_tokens = self.get_candidate_tokens(idx, tokens, similar_tokens, tag)

        if len(candidate_tokens) == 0:
            return None

        references = [untokenize(tokens)] * len(candidate_tokens)
        candidates = self.get_candidates(idx, tokens, candidate_tokens)

        p, r, f1 = self.scorer.score(candidates, references)

        replacement = self.select_replacement_from_candidates(f1, candidate_tokens, method='argmax')

        return replacement

    def get_candidate_tokens(self, idx, tokens, similar_tokens, tag):
        candidates = []
        for similar_token, score in similar_tokens:
            if score < self.glove_sim_threshold:
                continue
            new_tokens = copy.deepcopy(tokens)
            new_tokens[idx] = similar_token
            new_pos = nltk.pos_tag(new_tokens)
            new_tag = new_pos[idx][1]
            if new_tag == tag:
                candidates.append((similar_token, score))
        return candidates

    @staticmethod
    def get_candidates(idx, tokens, similar_tokens):
        candidates = []
        for similar_token, score in similar_tokens:
            new_tokens = copy.deepcopy(tokens)
            new_tokens[idx] = similar_token
            candidates.append(untokenize(new_tokens))
        return candidates

    def select_replacement_from_candidates(self, f1, candidate_tokens, method='argmax'):
        f1 = f1.detach().cpu().numpy()
        if method == 'random_with_threshold':
            valid_f1_ids = []
            for i, f in enumerate(f1):
                if f > self.bert_score_threshold:
                    valid_f1_ids.append(i)
            if len(valid_f1_ids) == 0:
                return None
            replacement = candidate_tokens[random.choice(valid_f1_ids)]
            return replacement[0]
        else:
            replacement = candidate_tokens[np.argmax(f1)]
            return replacement[0]


class TextAugmenterBacktranslation:

    def __init__(self, lang_weights):
        print('Initializing BACKTRANSLATION text AUGMENTER')
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.langs, self.lang_weights = list(lang_weights.keys()), list(lang_weights.values())
        # self.models, self.tokenizers = self.load_lang_models_and_tokenizers()

    def augment_data(self, clean_data):

        augmented_data = copy.deepcopy(clean_data)

        sents = []
        for img in tqdm(augmented_data['images']):
            for sent in img['sentences']:
                sents.append(sent['raw'])

        sents = sents
        if cfg.caption_aug_method == 'chain':
            tr = self.backtranslate_chain(sents)
        else:
            tr = self.backtranslate(sents)

        counter = 0
        for img in tqdm(augmented_data['images']):
            for sent in img['sentences']:
                try:
                    sent['aug_bt_{}'.format(cfg.caption_aug_method)] = tr[counter]
                    counter += 1
                except:
                    break

        return augmented_data

    @staticmethod
    def load_lang_model_and_tokenizer(src, tgt, use_romance=False):
        model_name = 'Helsinki-NLP/opus-mt-{}-{}'.format(src, tgt)
        if use_romance:
            #tgt = 'ROMANCE' if src == 'en' else tgt
            #src = 'ROMANCE' if tgt == 'en' else src
            model_name = 'Helsinki-NLP/opus-mt-{}-{}'.format(src, tgt)
        print('LOADING:', model_name)
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def select_languages(self, texts, langs):
        texts = np.array(texts)
        langs = np.array(langs)
        lang_idxs = {}
        lang_txt = {}

        for l in self.langs:
            lang_idxs[l] = np.where(langs == l)
            lang_txt[l] = texts[lang_idxs[l]]

        return lang_txt, lang_idxs

    def backtranslate(self, sents):

        langs = random.choices(self.langs, weights=self.lang_weights, k=len(sents))
        lang_txt, lang_idxs = self.select_languages(sents, langs)

        translated = {}
        backtranslated = {}

        # forward
        for tgt in self.langs:
            translated, backtranslated = self.translations_pickle_load(translated, backtranslated)
            if tgt in translated:
                continue
            src = 'en'
            txt = [str(t) for t in lang_txt[tgt]]
            translated[tgt] = self.translate(txt, src, tgt)
            self.translations_pickle_save(translated, backtranslated)

        # backward
        for src in self.langs:
            translated, backtranslated = self.translations_pickle_load(translated, backtranslated)
            if src in backtranslated:
                continue
            tgt = 'en'
            txt = translated[src]
            backtranslated[src] = self.translate(txt, src, tgt)
            self.translations_pickle_save(translated, backtranslated)

        t, b = self.translations_pickle_load(translated, backtranslated)
        result = np.zeros_like(sents)

        for l in self.langs:
            result[lang_idxs[l]] = np.array(backtranslated[l])

        return [str(s).lower() for s in result]

    def backtranslate_chain(self, sents):
        languages_chain = self.langs
        languages_chain.append('en')
        languages_chain.insert(0, 'en')

        translated = sents
        translated_dict = {}

        for i in range(1, len(languages_chain)):
            translated_dict = self.translations_pickle_load(translated_dict)[0]
            src = languages_chain[i - 1]
            tgt = languages_chain[i]
            step_name = '{}-{}'.format(src, tgt)
            if step_name in translated_dict:
                translated = translated_dict[step_name]
                continue
            translated = self.translate(translated, src, tgt)
            translated_dict[step_name] = translated
            self.translations_pickle_save(translated_dict)

        return translated

    @staticmethod
    def translations_pickle_load(*args):
        file_name = "./data/translations_{}_{}.pkl".format(cfg.dataset, cfg.caption_aug_method)
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                a = pickle.load(f)
                return a
        else:
            return args

    @staticmethod
    def translations_pickle_save(*args):
        file_name = "./data/translations_{}_{}.pkl".format(cfg.dataset, cfg.caption_aug_method)
        with open(file_name, 'wb') as f:
            pickle.dump(args, f)

    def translate(self, txts, src, tgt):

        with torch.no_grad():

            model, tokenizer = self.load_lang_model_and_tokenizer(src, tgt, use_romance=True)

            model = model.to(self.device)
            model.eval()

            batches = self.prep_batches(txts)

            translated = []

            for batch in tqdm(batches, desc="{} -> {}".format(src, tgt)):
                translated.extend(self.translate_batch(model, tokenizer, batch))

            self.clear_gpu_cache(model, tokenizer)

            return translated

    @staticmethod
    def prep_batches(original_texts, batch_size=8):
        batches = []
        for i in range((len(original_texts) // batch_size) + 1):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size

            if batch_end >= len(original_texts):
                batch_end = len(original_texts)

            batches.append(original_texts[batch_start:batch_end])
        return batches

    def translate_batch(self, model, tokenizer, batch):
        tokens = tokenizer(batch, return_tensors='pt', truncation=True,
                           padding='max_length', max_length=128).to(self.device)

        tr = model.generate(**tokens)
        dec = tokenizer.batch_decode(tr, skip_special_tokens=True)

        self.clear_gpu_cache(tokens, tr)

        return dec

    @staticmethod
    def clear_gpu_cache(*objs):
        for o in objs:
            del o
        torch.cuda.empty_cache()
        gc.collect()

new_mapping = {
        "0": "GPU-3523f28d-a20d-b020-6229-fe23eee5a105",
        "1": "GPU-2389f231-f2ec-2f4b-0878-ffd79787f412",
        "2": "GPU-a4185d1a-d26c-f749-8073-7b5558423a4c",
        "3": "GPU-00ff7e48-7006-93b7-0127-0d1d2522a625",
        "4": "GPU-85dc2634-c618-6771-e513-06a4289cf0b5",
        "5": "GPU-300de5b9-703c-5e2c-dc98-02ca380f1446",
        
        "6:0": "MIG-ba51d0b6-ae95-5a2d-b2a6-636a246495a1",  # MIG 3g.40gb
        "6:1": "MIG-032c9e9d-8350-5552-8138-b055b2ebe447",  # MIG 2g.20gb
        "6:2": "MIG-5b302910-e4da-563c-8f1b-dff9dec32d2e",  # MIG 1g.10gb
        "6:3": "MIG-48d4aeb9-4840-51ba-9ffb-a83d1f5db588",  # MIG 1g.10gb
        
        "7:0": "MIG-bcfdbd83-be78-533e-a986-25fcc5fa1a6f",  # MIG 3g.40gb
        "7:1": "MIG-62586c97-df16-56a0-af10-68c67b41898e",  # MIG 4g.40gb
}

if __name__ == '__main__':
    print("AUGMENT CAPTIONS")
    random.seed(cfg.seed)
    # torch.set_num_threads(8)
    os.environ['CUDA_VISIBLE_DEVICES'] = new_mapping['4']
    device = torch.device(cfg.cuda_device if torch.cuda.is_available() else "cpu")
    print ('hey hey',device)

    # check_aug_distribution()

    # read original captions (clean data)
    clean_data = read_json(cfg.dataset_json_file)

    if cfg.caption_aug_type == 'backtranslation':
        ta = TextAugmenterBacktranslation(cfg.caption_aug_bt_lang_weights)
    else:
        ta = TextAugmenterRuleBased(cfg.caption_aug_rb_glove_sim_threshold, cfg.caption_aug_rb_bert_score_threshold)

    # add noise to captions
    augmented_data = ta.augment_data(clean_data)

    # output noisy captions
    write_json(cfg.caption_aug_dataset_json, remove_tokens(augmented_data))

    check_aug_distribution()

    print("DONE\n\n\n")

from utils import read_json, write_hdf5, get_captions
from configs.config_txt_aug import cfg

import transformers
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from images_augment import new_mapping
import os


class CaptionsDataset(torch.utils.data.Dataset):
    """
    Dataset for captions
    """

    def __init__(self, captions, tokenizer, max_len=128):
        self.captions = captions
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        item = self.encode_caption(self.captions[idx])
        return idx, item["input_ids"].squeeze(), item["attention_mask"].squeeze()

    def __len__(self):
        return len(self.captions)

    def encode_caption(self, caption):
        """
        Encode caption with model's tokenizer

        :param caption: caption

        :return: tokenized caption
        """
        return self.tokenizer.encode_plus(
            caption,
            max_length=self.max_len,
            padding="max_length",
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )


def get_embeddings(model, dataloader, device, num_hidden_states=4, operation="sum"):
    """
    Get embeddings

    :param model: model
    :param dataloader: data loader with captions
    :param device: CUDA device
    :param num_hidden_states: number of last BERT's hidden states to use
    :param operation: how to combine last hidden states of BERT: 'concat' or 'sum'

    :return: embeddings
    """

    with torch.no_grad():  # no need to call Tensor.backward(), saves memory
        model = model.to(device)  # to gpu (if presented)

        batch_outputs = []
        hs = [i for i in range(-(num_hidden_states), 0)]
        len_hs = len(hs) * 768 if (operation == "concat") else 768
        print("(Last) Hidden states to use:", hs, " -->  Embedding size:", len_hs)

        for idx, input_ids, attention_masks in tqdm(
            dataloader, desc="Getting Embeddings (batches): "
        ):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            out = model(input_ids=input_ids, attention_mask=attention_masks)
            hidden_states = out["hidden_states"]
            last_hidden = [hidden_states[i] for i in hs]

            if operation == "sum":
                # stack list of 3D-Tensor into 4D-Tensor
                # 3D [(batch_size, tokens, 768)] -> 4D (hidden_states, batch_size, tokens, 768)
                hiddens = torch.stack(last_hidden)
                # sum along 0th dimension -> 3D (batch_size, tokens, output_dim)
                resulting_states = torch.sum(hiddens, dim=0).squeeze()
            elif operation == "concat":
                # concat list of 3D-Tensor into 3D-Tensor
                # 3D [(batch_size, tokens, 768)] -> 3D (batch_size, tokens, 768 * list_length)
                resulting_states = torch.cat(tuple(last_hidden), dim=2)
            else:
                raise Exception("unknown operation " + str(operation))

            # token embeddings to sentence embedding via token embeddings averaging
            # 3D (batch_size, tokens, resulting_states.shape[2]) -> 2D (batch_size, resulting_states.shape[2])
            sentence_emb = torch.mean(resulting_states, dim=1).squeeze()
            batch_outputs.append(sentence_emb)

        # vertical stacking (along 0th dimension)
        # 2D [(batch_size, resulting_states.shape[2])] -> 2D (num_batches * batch_size, resulting_states.shape[2])
        output = torch.vstack(batch_outputs)
        embeddings = (
            output.cpu().numpy()
        )  # return to cpu (or do nothing), convert to numpy
        print("Embeddings shape:", embeddings.shape)
        return embeddings


def get_max_token_length(tokenizer, captions):
    """
    Get size of largest tokenized caption

    :param tokenizer: tokenizer
    :param captions: captions

    :return: size of largest tokenized
    """
    max_length = 128
    token_lens = []
    for c in tqdm(captions, desc="Searching for max token length: "):
        toks = tokenizer.encode(c, max_length=max_length, truncation=True)
        token_lens.append(len(toks))
    max_token_length = max(token_lens)
    print("Max token length:", max_token_length)
    return max_token_length


def embed_captions(captions):
    """
    Generate embeddings from caption list.

    :param captions: captions

    :return: embeddings list
    """
    # load tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    max_token_length = get_max_token_length(tokenizer, captions)
    print(max_token_length)
    max_token_length = max_token_length
    # max_token_length = cfg.caption_token_length

    # create dataset and dataloader
    captions_dataset = CaptionsDataset(captions, tokenizer, max_len=max_token_length)
    captions_dataloader = DataLoader(captions_dataset, batch_size=64, shuffle=False)

    # get pretrained BERT
    bert = transformers.BertModel.from_pretrained(
        "bert-base-uncased", output_hidden_states=True
    )

    # get BERT embeddings
    embeddings = get_embeddings(
        bert,
        captions_dataloader,
        device,
        num_hidden_states=cfg.caption_hidden_states,
        operation=cfg.caption_hidden_states_operator,
    )
    return embeddings


if __name__ == "__main__":
    print("CREATE CAPTION EMBEDDINGS FROM DATASET FILE")

    # device
    torch.set_num_threads(8)
    os.environ["CUDA_VISIBLE_DEVICES"] = new_mapping["6:1"]
    # device = torch.device(cfg.cuda_device if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read captions from JSON file
    data = read_json(cfg.dataset_json_file)

    # get captions
    captions, aug_captions_rb, aug_captions_bt_prob, aug_captions_bt_chain = (
        get_captions(data)
    )
    # print (captions)

    # generate embeddings
    embeddings = embed_captions(captions)

    # save embeddings
    write_hdf5(cfg.caption_emb_file, embeddings.astype(np.float32), "caption_emb")

    if len(aug_captions_rb) > 0:
        aug_emb_rb = embed_captions(aug_captions_rb)
        write_hdf5(
            cfg.caption_emb_aug_file_rb, aug_emb_rb.astype(np.float32), "caption_emb"
        )

    if len(aug_captions_bt_prob) > 0:
        aug_emb_bt_prob = embed_captions(aug_captions_bt_prob)
        write_hdf5(
            cfg.caption_emb_aug_file_bt_prob,
            aug_emb_bt_prob.astype(np.float32),
            "caption_emb",
        )

    if len(aug_captions_bt_chain) > 0:
        aug_emb_bt_chain = embed_captions(aug_captions_bt_chain)
        write_hdf5(
            cfg.caption_emb_aug_file_bt_chain,
            aug_emb_bt_chain.astype(np.float32),
            "caption_emb",
        )

    print("DONE\n\n\n")

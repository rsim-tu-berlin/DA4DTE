from models.DUCH import DUCH
import torch
import torchvision
import os
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet152, ResNet152_Weights

import transformers
from PIL import Image
from torchvision import transforms


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

os.environ['CUDA_VISIBLE_DEVICES'] = new_mapping['4']
cuda_device = 'cuda'
device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
print ('device', device)

from torch.utils.data import DataLoader
from tqdm import tqdm

'*****************************************dataloader for catpions and bert embedding*********************************************'

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
        return idx, item['input_ids'].squeeze(), item['attention_mask'].squeeze()

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
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt')


def get_embeddings_txt(model, dataloader, device, num_hidden_states=4, operation='sum'):
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
        len_hs = len(hs) * 768 if (operation == 'concat') else 768
        print('(Last) Hidden states to use:', hs, ' -->  Embedding size:', len_hs)

        

        for idx, input_ids, attention_masks in tqdm(dataloader, desc='Getting Embeddings (batches): '):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            out = model(input_ids=input_ids, attention_mask=attention_masks)
            hidden_states = out['hidden_states']
            last_hidden = [hidden_states[i] for i in hs]
            

            if operation == 'sum':
                # stack list of 3D-Tensor into 4D-Tensor
                # 3D [(batch_size, tokens, 768)] -> 4D (hidden_states, batch_size, tokens, 768)
                hiddens = torch.stack(last_hidden)
                # sum along 0th dimension -> 3D (batch_size, tokens, output_dim)
                resulting_states = torch.sum(hiddens, dim=0).squeeze()
            elif operation == 'concat':
                # concat list of 3D-Tensor into 3D-Tensor
                # 3D [(batch_size, tokens, 768)] -> 3D (batch_size, tokens, 768 * list_length)
                resulting_states = torch.cat(tuple(last_hidden), dim=2)
            else:
                raise Exception('unknown operation ' + str(operation))

            # token embeddings to sentence embedding via token embeddings averaging
            # 3D (batch_size, tokens, resulting_states.shape[2]) -> 2D (batch_size, resulting_states.shape[2])
            sentence_emb = torch.mean(resulting_states, dim=1).squeeze()
            print (sentence_emb.shape)
            
            batch_outputs.append(sentence_emb)

        # vertical stacking (along 0th dimension)
        # 2D [(batch_size, resulting_states.shape[2])] -> 2D (num_batches * batch_size, resulting_states.shape[2])
        output = torch.vstack(batch_outputs)
        embeddings = output.cpu().numpy()  # return to cpu (or do nothing), convert to numpy
        print('Embeddings shape?:', embeddings.shape)
        return embeddings
    
def embed_captions(captions):
    """
    Generate embeddings from caption list.

    :param captions: captions

    :return: embeddings list
    """
    # load tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    # max_token_length = get_max_token_length(tokenizer, captions)
    # print (max_token_length)
    # # max_token_length = caption_token_length

    # create dataset and dataloader
    captions_dataset = CaptionsDataset(captions, tokenizer, max_len=75)
    captions_dataloader = DataLoader(captions_dataset, batch_size=batch_size, shuffle=False)

    # get pretrained BERT
    bert = transformers.BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

    # get BERT embeddings
    embeddings = get_embeddings_txt(bert, captions_dataloader, device, num_hidden_states=4,
                                operation='sum')
    return embeddings




'**********************************************ResNet Embedding************************************************************'

def get_resnet_model_for_embedding(model=None):
    """
    Remove the last layer to get embeddings

    :param model: pretrained model (optionally)

    :return: pretrained model without last (classification layer)
    """
    if model is None:
        # model = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False).eval()
        model = resnet152(weights =ResNet152_Weights.DEFAULT, progress = False).eval()

    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    # print (model)
    return model

def get_embeddings_img(model, img, device):
    """
    Get Embeddings

    :param model: model
    :param dataloader: data loader with images
    :param device: CUDA device
    :return:
    """
    with torch.no_grad():  # no need to call Tensor.backward(), saves memory
        model = model.to(device)  # to gpu (if presented)

        img = img.to(device)
        # model(img) 
        output = model(img)   # (batches, batch_size, output_dim) -> (batches * batch_size, output_dim)

        embeddings = output.squeeze().cpu().numpy()  # return to cpu (or do nothing), convert to numpy
        print('Embeddings shape:', embeddings.shape)
        return embeddings


'*********************************************load_single_image***********************************************'



def load_single_image(img_name):
    """
    Load single image from the disc by name.

    :return: PIL.Image array
    """
    # print (img_name)
    img= Image.open(img_name)
    # print(img)
    img.show()
    img = img.resize((128,128),resample= Image.Resampling.BILINEAR)
    # print(img)
    
    # import ipdb; ipdb.set_trace()

    return img

'**********************************************Load Duch Model*************************************************'

model_type = 'DUCH'
batch_size = 256
image_dim = 2048
text_dim = 768
hidden_dim = 1024 * 4
hash_dim = 128
path_model= 'checkpoints/vessel_dataset_'+str(hash_dim)+'_test_each_img_random'


def get_model():
        """
        Initialize model

        :returns: instance of NN model
        """
        return DUCH(image_dim, text_dim, hidden_dim, hash_dim)

def load_model(model, tag='best'):
        """
        Load model from the disk

        :param: tag: name tag
        """
        # model.load(os.path.join(path_model, model.module_name + '_' + str(tag) + '.pth'))
        # x = model.load(os.path.join(path_model, model.module_name + '_' + str(tag) + '.pth'))
        model.load_state_dict(torch.load(os.path.join(path_model, model.module_name + '_' + str(tag) + '.pth')))
        x=model.load_state_dict(torch.load(os.path.join(path_model, model.module_name + '_' + str(tag) + '.pth')))
        # print (x)
        return model





# get the duch module to generate hash codes
model_duch_get = get_model()

model_duch =load_model(model_duch_get)
model_duch.eval()






captions = ['two vessels',' '] #due to data loader this should be a list of at least two elemnts! 
embeddings_sentence = embed_captions(captions)

# /home/genc/DA4DTE/duch-master/duch-master/
dataset_image_folder_path = '/home/genc/DA4DTE/duch-master/duch-master/img_test/'

img =dataset_image_folder_path +'T10SEG_20190808T184921_TCI_crop_x-512_y-1408.png'


img_load = load_single_image(img)
# print (torchvision.transforms.functional.pil_to_tensor(img_load))
transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()


])

img_transform = transform(img_load).unsqueeze(0)

resnet = get_resnet_model_for_embedding() 
embeddings_img_real = get_embeddings_img(resnet,img_transform,device )

emb_torch_img = torch.from_numpy(embeddings_img_real)
emb_torch_img = torch.reshape(emb_torch_img,(1,embeddings_img_real.shape[0]))
hash_img = model_duch.generate_img_code(emb_torch_img)
hash_img=hash_img.sign()

embed_torch_txt =torch.from_numpy(embeddings_sentence[0]) #[0] to get the first caption!!
embed_torch_txt =torch.reshape(embed_torch_txt,(1,embeddings_sentence[0].shape[0]))

hash_sent = model_duch.generate_txt_code(embed_torch_txt)
hash_sent= hash_sent.sign()

hash_sent[hash_sent ==-1] =0
hash_img[hash_img==-1] = 0

print ('hash img',hash_img)
print ('hash_sent',hash_sent)
# hash_img2[hash_img2==-1] = 0
# convert to dictionary images and sentences; Key img name/sentence and item hashcode img/sentence
# sentence_dictionary
dict_sent_hash_zerone ={}
print (len(captions))
dict_sent_hash_zerone[captions[0]] = hash_sent[[0]].detach().cpu().numpy().tolist()
print ('hash codes for the sentence: **two vessels**=' , dict_sent_hash_zerone['two vessels'])
print (img)
dict_hash_img_zerone= {}

dict_hash_img_zerone[img] = hash_img[[0]].detach().cpu().numpy().tolist()
dict_hash_img_zerone

print ('hash codes for the image filename: **T10SEG_20190808T184921_TCI_crop_x-512_y-1408.png**=' , dict_hash_img_zerone[img])
# img_dictionary





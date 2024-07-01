import os

from utils import read_json, write_hdf5, get_image_file_names, shuffle_file_names_list, reshuffle_embeddings
from configs.config_img_aug import cfg

import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
# import PIL
from PIL import Image
import random
import tifffile as tiff
# import tifffile
from torchvision.models import resnet18, ResNet18_Weights
from skimage.transform import resize

class ImagesDataset(torch.utils.data.Dataset):
    """
    Dataset for images
    """

    def __init__(self, image_file_names, images_folder, img_transforms_dicts, img_aug_set):
        self.image_file_names = image_file_names
        self.images_folder = images_folder
        self.img_transforms_dicts = img_transforms_dicts
        self.img_aug_set = img_aug_set
        self.img_transforms = self.init_transforms()
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):

        img = self.load_single_image(self.image_file_names[idx])
        

        # if only one sequence in self.img_transforms - it will be always applied
        transform = random.choice(self.img_transforms)
        # transform = self.img_transforms[0]

        img_aug = transform(img)
        # print (img_aug.shape)
        return idx, img_aug, self.to_tensor(img)

    def __len__(self):
        return len(self.image_file_names)


# **********************tiff files 4 channles**************************************
    # def load_single_image(self, img_name):
    #     img =tiff.imread(os.path.join(self.images_folder, img_name))
        
    #     # print ('tiff',img)
    #     # img = resize(img,(128,128))

    #     # print ('skimage',img)
    #     # print (img.shape)
    #     img = img[:,:,[0,1,2]] #
    #     # print (img.shape)
    #     # if np.max (img) != np.min(img):
    #     # standard_dev = np.std(img[0])
    #     min_value,max_value = np.percentile(img,[2,98])
    #     # max_value = np.max(img[0])
    #     # min_value = np.min(img[0])
    #     img = img.clip(min_value,max_value)
    #     normalized_image  = ((img -min_value)  / (max_value -min_value))*255
        
    #     normalized_image = normalized_image.astype(np.uint8)


    #     # import ipdb; ipdb.set_trace()
    #     img = Image.fromarray(normalized_image)
    #     img = img.resize((128,128),resample= Image.Resampling.BILINEAR)
    #     # print (img_name)
        
    #     if not np.any(img):
    #         print ('zero file found',img_name)
    #     # assert np.any(img), "Image must have non-zero elements."
    #     # print (img.shape)
    #     # print (type(img))

    #     # print (img.shape)
    #     # print ('frame shape',frame_pil.shape)
    #     """
    #     Load single image from the disc by name.


    #     :return: PIL.Image array
    #     """
    #     return img
# ***********************png 3 channels
    def load_single_image(self, img_name):
        """
        Load single image from the disc by name.

        :return: PIL.Image array
        """
        img= Image.open(os.path.join(self.images_folder, img_name))
        # print(img)
        img = img.resize((128,128),resample= Image.Resampling.BILINEAR)
        # print(img)
        # import ipdb; ipdb.set_trace()

        return img


    def init_transforms(self):
        """
        Initialize transforms.

        :return: list of transforms sequences (may be only one), one will be selected and applied to image
        """
        if self.img_aug_set == 'each_img_random':
            return self.init_transforms_random()
        else:
            return [self.init_transforms_not_random(self.img_transforms_dicts[self.img_aug_set])]

    def init_transforms_random(self):
        """
        Initialize transforms randomly from transforms dictionary.

        :return: list of transforms sequences, one will be selected and applied to image
        """
        transforms = []
        for ts in self.img_transforms_dicts[self.img_aug_set]:
            transforms.append(self.init_transforms_not_random(self.img_transforms_dicts[ts]))
        return transforms

    @staticmethod
    def init_transforms_not_random(transform_dict):
        """
        Initialize transforms non-randomly from transforms dictionary.
        :param transform_dict: transforms dictionary from config file

        :return: sequence of transforms to apply to each image
        """
        def _rotation_transform(values):
            return torchvision.transforms.RandomChoice([torchvision.transforms.RandomRotation(val) for val in values])

        def _affine_transform(values):
            return torchvision.transforms.RandomChoice([torchvision.transforms.RandomAffine(val) for val in values])

        def _gaussian_blur_transform(values):
            return torchvision.transforms.GaussianBlur(*values)

        def _center_crop_transform(values):
            return torchvision.transforms.CenterCrop(values)

        def _random_crop_transform(values):
            return torchvision.transforms.RandomCrop(values)

        def _color_jittering(values):
            return torchvision.transforms.ColorJitter(0.8 * values, 0.8 * values, 0.8 * values, 0.2 * values)

        image_transform_funcs = {'rotation': _rotation_transform,
                                 'affine': _affine_transform,
                                 'blur': _gaussian_blur_transform,
                                 'center_crop': _center_crop_transform,
                                 'random_crop': _random_crop_transform,
                                 'jitter': _color_jittering}

        transforms_list = []

        for k, v in transform_dict.items():
            transforms_list.append(image_transform_funcs[k](v))
        transforms_list.append(torchvision.transforms.Resize((224,224)))
        transforms_list.append(torchvision.transforms.ToTensor())
        print ('transfrom lists',(transforms_list))
        return torchvision.transforms.Compose(transforms_list)


def get_embeddings(model, dataloader, device):
    """
    Get Embeddings

    :param model: model
    :param dataloader: data loader with images
    :param device: CUDA device
    :return:
    """
    with torch.no_grad():  # no need to call Tensor.backward(), saves memory
        model = model.to(device)  # to gpu (if presented)

        batch_outputs = []
        # print ('ciao')

        for idx, x, _ in tqdm(dataloader, desc='Getting Embeddings (batches): '):
            # print (idx)
            x = x.to(device)
            # print (x)
            batch_outputs.append(model(x))

        output = torch.vstack(batch_outputs)  # (batches, batch_size, output_dim) -> (batches * batch_size, output_dim)

        embeddings = output.squeeze().cpu().numpy()  # return to cpu (or do nothing), convert to numpy
        print('Embeddings shape:', embeddings.shape)
        return embeddings


# def get_resnet_model_for_embedding(model=None):
#     """
#     Remove the last layer to get embeddings

#     :param model: pretrained model (optionally)

#     :return: pretrained model without last (classification layer)
#     """
#     if model is None:
#         # model = torchvision.models.resnet18(pretrained=True)
#         # model = conve
#         num_channels = 4
#         model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT, progress=False)
#         # here we change the input to 4 channels
#         num_filters = model.conv1.out_channels
#         kernel_size = model.conv1.kernel_size
#         stride = model.conv1.stride
#         padding = model.conv1.padding
#         conv1 = torch.nn.Conv2d(num_channels, num_filters, kernel_size=kernel_size, stride=stride, padding=padding)
#         original_weights = model.conv1.weight.data.mean(dim=1, keepdim=True)
#         conv1.weight.data = original_weights.repeat(1, num_channels, 1, 1)
#         model.conv1 = conv1


        
#     model = torch.nn.Sequential(*(list(model.children())[:-1]))
#     return model

# ******************************************three channels png******************************************************
def get_resnet_model_for_embedding(model=None):
    """
    Remove the last layer to get embeddings

    :param model: pretrained model (optionally)

    :return: pretrained model without last (classification layer)
    """
    if model is None:
        model = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False).eval()
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model


def show_rand_imgs(dataset):
    # visualization
    def imshow(imgs):
        fig = plt.figure(figsize=(len(imgs) * 3, 6))
        for i, img in enumerate(imgs):
            ax1 = fig.add_subplot(2, len(imgs), len(imgs) + i + 1)
            npimg1 = img[1].numpy()
            plt.imshow(np.transpose(npimg1, (1, 2, 0)))
            plt.axis('off')

            ax2 = fig.add_subplot(2, len(imgs), i + 1)
            npimg2 = img[2].numpy()
            plt.imshow(np.transpose(npimg2, (1, 2, 0)))
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join('plots', 'tmp.png'))

    import matplotlib.pyplot as plt
    choice = np.random.choice(len(dataset), 10)
    chosen = [dataset[c] for c in choice]
    imshow(chosen)

# mapping of MARS GPU
new_mapping = {
        "0": "GPU-3523f28d-a20d-b020-6229-fe23eee5a105",
        "1": "GPU-2389f231-f2ec-2f4b-0878-ffd79787f412",
        "2": "GPU-a4185d1a-d26c-f749-8073-7b5558423a4c",
        "3": "GPU-00ff7e48-7006-93b7-0127-0d1d2522a625",
        "4": "GPU-85dc2634-c618-6771-e513-06a4289cf0b5",
        "5": "GPU-300de5b9-703c-5e2c-dc98-02ca380f1446",
        
        "6:0": "MIG-ba51d0b6-ae95-5a2d-b2a6-636a246495a1",  # MIG 3g.40gb MIG-ba51d0b6-ae95-5a2d-b2a6-636a246495a1
        "6:1": "MIG-032c9e9d-8350-5552-8138-b055b2ebe447",  # MIG 2g.20gb
        "6:2": "MIG-5b302910-e4da-563c-8f1b-dff9dec32d2e",  # MIG 1g.10gb
        "6:3": "MIG-48d4aeb9-4840-51ba-9ffb-a83d1f5db588",  # MIG 1g.10gb
        
        "7:0": "MIG-bcfdbd83-be78-533e-a986-25fcc5fa1a6f",  # MIG 3g.40gb
        "7:1": "MIG-62586c97-df16-56a0-af10-68c67b41898e",  # MIG 4g.40gb
}

if __name__ == '__main__':
    print("CREATE AUGMENTED IMAGE EMBEDDINGS")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    print ('what is cuda here? ',cfg.cuda_device)
    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = new_mapping['2']
    device = torch.device(cfg.cuda_device if torch.cuda.is_available() else "cpu")
    print ('device', device)
    # torch.set_num_threads(8)

    # read captions from JSON file
    data = read_json(cfg.dataset_json_file)

    # get file names
    file_names = get_image_file_names(data)

    # shuffle images to avoid errors caused by batch normalization layer in ResNet18 (batch size shall also be big)
    file_names_permutated, permutations = shuffle_file_names_list(file_names)
    # print ('ketu jane emrat e imazheve',file_names_permutated[250])
    

    # create dataset and dataloader
    images_dataset = ImagesDataset(file_names_permutated,
                                   cfg.dataset_image_folder_path,
                                   cfg.image_aug_transform_sets, cfg.img_aug_set)
    # print (images_dataset)
    # show_rand_imgs(images_dataset)
    # import ipdb; ipdb.set_trace()
    images_dataloader = DataLoader(images_dataset, batch_size=cfg.image_emb_batch_size, shuffle=False)
    # images_dataloader = DataLoader(images_dataset, batch_size=200, shuffle=False)

    # load pretrained ResNet without last (classification layer)
    # resnet = get_resnet_model_for_embedding(torch.load(os.path.join('fine_tuned', 'resnet18ft.pth')))
    resnet = get_resnet_model_for_embedding()

    embeddings = get_embeddings(resnet, images_dataloader, device)

    # return embeddings back to original order of images
    embeddings_orig_order = reshuffle_embeddings(embeddings, permutations)


    # save embeddings (ketu)
    write_hdf5(cfg.image_emb_aug_file, embeddings_orig_order.astype(np.float32), 'image_emb')

    print("DONE\n\n\n")

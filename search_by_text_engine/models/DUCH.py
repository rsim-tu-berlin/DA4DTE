from torch import nn
import torch
import os


class DUCH(nn.Module):
    def __init__(self, image_dim, text_dim, hidden_dim, hash_dim):
        super(DUCH, self).__init__()
        self.module_name = "DUCH"
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.hash_dim = hash_dim

        self.image_module = nn.Sequential(
            nn.Linear(image_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hash_dim, bias=True),
            nn.Tanh(),
        )
        self.text_module = nn.Sequential(
            nn.Linear(text_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hash_dim, bias=True),
            nn.Tanh(),
        )

        # hash discriminator
        self.hash_dis = nn.Sequential(
            nn.Linear(self.hash_dim, self.hash_dim * 2, bias=True),
            nn.BatchNorm1d(self.hash_dim * 2),
            nn.ReLU(True),
            nn.Linear(self.hash_dim * 2, 1, bias=True),
        )

    def forward(self, *args):
        if len(args) == 4:
            res = self.forward_img_img_txt_txt(*args)
        elif len(args) == 3:
            res = self.forward_img_txt_txt(*args)
        elif len(args) == 2:
            res = self.forward_img_txt(*args)
        else:
            raise Exception("Method must take 2, 3 or 4 arguments")
        return res

    def forward_img_img_txt_txt(self, r_img1, r_img2, r_txt1, r_txt2):
        h_img1 = self.image_module(r_img1).squeeze()
        h_img2 = self.image_module(r_img2).squeeze()
        h_txt1 = self.text_module(r_txt1).squeeze()
        h_txt2 = self.text_module(r_txt2).squeeze()
        return h_img1, h_img2, h_txt1, h_txt2

    def forward_img_txt_txt(self, r_img, r_txt1, r_txt2):
        h_img = self.image_module(r_img).squeeze()
        h_txt1 = self.text_module(r_txt1).squeeze()
        h_txt2 = self.text_module(r_txt2).squeeze()
        return h_img, h_txt1, h_txt2

    def forward_img_txt(self, r_img, r_txt):
        h_img = self.image_module(r_img)
        h_txt = self.text_module(r_txt)
        return h_img, h_txt

    def generate_img_code(self, i):
        return self.image_module(i).detach()

    def generate_txt_code(self, t):
        return self.text_module(t).detach()

    def load(self, path, use_gpu=False):
        if not use_gpu:
            self.load_state_dict(
                torch.load(path, map_location=lambda storage, loc: storage)
            )
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None, path="./checkpoints", cuda_device=None):
        if not os.path.exists(path):
            os.makedirs(path)
        if cuda_device.type == "cpu":
            torch.save(self.state_dict(), os.path.join(path, name))
        else:
            with torch.cuda.device(cuda_device):
                torch.save(self.state_dict(), os.path.join(path, name))
        return name

    def discriminate_hash(self, h):
        return self.hash_dis(h).squeeze()


# LEO
# def regress(self, h):
#     return self.reg_head(h).squeeze()

from torch import nn
import torch
from torch.nn import functional as F

import math


class NTXentLoss(nn.Module):

    """
    Normalized Temperature-scaled Cross-entropy Loss (NTXent Loss).

    Contains single-modal and cross-modal implementations.
    """

    def __init__(self, temperature=1, eps=1e-6):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, *args, type='orig'):
        if type == 'cross':
            return self.forward_cross_modal(*args)
        if type == 'orig':
            return self.forward_orig(*args)
        if type == 'both':
            return self.forward_orig(*args), self.forward_cross_modal(*args)
        else:
            raise Exception("Wrong NTXent loss type, must be: 'cross', 'orig' or 'both'")

    def forward_cross_modal(self, mod1, mod2):
        """
        Cross-modal case:

        p - positive pair
        n - negative pair
        sim - cosine similarity

        ix - image modality feature number x
        tx - text modality feature number x

        Cross-modal case of NTXent doesn't consider similarities inside of the same modality

                        Similarities matrix: exp(sim(i, y))
                             +--+--+--+--+--+--+--+
                             |  |i1|i2|i3|t1|t2|t3|
         Modality            +--+--+--+--+--+--+--+
         Features            |i1|0 |0 |0 |p |n |n |
        +--+  +--+           +--+--+--+--+--+--+--+
        |i1|  |t1|           |i2|0 |0 |0 |n |p |n |
        +--+  +--+           +--+--+--+--+--+--+--+
        |i2|  |t2|  ------>  |i3|0 |0 |0 |n |n |p |
        +--+  +--+           +--+--+--+--+--+--+--+
        |i3|  |t3|           |t1|p |n |n |0 |0 |0 |
        +--+  +--+           +--+--+--+--+--+--+--+
                             |t2|n |p |n |0 |0 |0 |
                             +--+--+--+--+--+--+--+
                             |t3|n |n |p |0 |0 |0 |
                             +--+--+--+--+--+--+--+

        :param: mod1: features of the 1st modality
        :param: mod1: features of the 2nd modality
        :return: NTXent loss

        """
        # normalize for numerical stability
        mod1 = F.normalize(mod1)
        mod2 = F.normalize(mod2)

        out = torch.cat([mod1, mod2], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]

        cov = torch.mm(out, out.t().contiguous())  # cosine similarities matrix
        sim = torch.exp(cov / self.temperature)

        # mask for cross-modal case, nullifies certain regions (see docstring)
        zeros = torch.zeros(mod1.shape[0], mod1.shape[0]).to(sim.device)
        ones = torch.ones(mod1.shape[0], mod1.shape[0]).to(sim.device)
        mask = torch.hstack([torch.vstack([zeros, ones]), torch.vstack([ones, zeros])]).to(sim.device)

        sim = sim * mask

        # neg: [2 * batch_size]
        # negative pairs sum
        neg = sim.sum(dim=1)

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(mod1 * mod2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + self.eps)).sum()
        return loss

    def forward_orig(self, out_1, out_2):
        """
        Implementation taken from:
        https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py

        p - positive pair
        n - negative pair
        sim - cosine similarity
        e - Euler's number

        ix - value x of input feature vector i
        tx - value x of input feature vector t

                        Similarities matrix: exp(sim(i, y))
                             +--+--+--+--+--+--+--+
                             |  |i1|i2|i3|t1|t2|t3|
         Modality            +--+--+--+--+--+--+--+
         Features            |i1|e |n |n |p |n |n |
        +--+  +--+           +--+--+--+--+--+--+--+
        |i1|  |t1|           |i2|n |e |n |n |p |n |
        +--+  +--+           +--+--+--+--+--+--+--+
        |i2|  |t2|  ------>  |i3|n |n |e |n |n |p |
        +--+  +--+           +--+--+--+--+--+--+--+
        |i3|  |t3|           |t1|p |n |n |e |n |n |
        +--+  +--+           +--+--+--+--+--+--+--+
                             |t2|n |p |n |n |e |n |
                             +--+--+--+--+--+--+--+
                             |t3|n |n |p |n |n |e |
                             +--+--+--+--+--+--+--+

        :param out_1: input feature vector i
        :param out_2: input feature vector t
        :return: NTXent loss
        """
        out_1 = F.normalize(out_1)
        out_2 = F.normalize(out_2)

        out = torch.cat([out_1, out_2], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / self.temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^1 to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=self.eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + self.eps)).sum()
        return loss

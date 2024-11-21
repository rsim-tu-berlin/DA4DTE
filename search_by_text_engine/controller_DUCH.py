from models.DUCH import DUCH
from loss.contrastive_loss import NTXentLoss
from utils import write_pickle, calc_map_k, calc_map_rad, pr_curve, p_top_k
from torch import autograd
import torch
from torch.nn.functional import one_hot
from torch.optim import Adam

import time

import os


class ControllerDUCH:
    """
    Training/evaluation controller.
    """

    def __init__(self, log, cfg, dataloaders):
        self.since = time.time()
        self.logger = log
        self.cfg = cfg
        self.path = 'checkpoints/' + '_'.join([self.cfg.dataset, str(self.cfg.hash_dim), self.cfg.tag, self.cfg.preset])

        self.device = torch.device(self.cfg.cuda_device if torch.cuda.is_available() else "cpu")

        self.dataloader_quadruplets, self.dataloader_q, self.dataloader_db = dataloaders

        self.B, self.Hi1, self.Hi2, self.Ht1, self.Ht2 = self.init_hashes()

        self.losses = []

        self.maps_max = {'i2t': 0., 't2i': 0., 'i2i': 0., 't2t': 0., 'avg': 0.}
        self.maps = {'i2t': [], 't2i': [], 'i2i': [], 't2t': [], 'avg': []}

        self.model = self.get_model()
        self.optimizer_gen, self.optimizer_dis = self.get_optimizers()
        self.contr_loss = NTXentLoss()

    def init_hashes(self):
        """
        Initialize hash values (either zeros or random, see below)

        :return: initialized hash values
        """
        dataset_size = len(self.dataloader_quadruplets.dataset)
        B = torch.randn(dataset_size, self.cfg.hash_dim).sign().to(self.device)
        Hi1 = torch.zeros(dataset_size, self.cfg.hash_dim).sign().to(self.device)
        Hi2 = torch.zeros(dataset_size, self.cfg.hash_dim).sign().to(self.device)
        Ht1 = torch.zeros(dataset_size, self.cfg.hash_dim).sign().to(self.device)
        Ht2 = torch.zeros(dataset_size, self.cfg.hash_dim).sign().to(self.device)
        return B, Hi1, Hi2, Ht1, Ht2

    def get_model(self):
        """
        Initialize model

        :returns: instance of NN model
        """
        return DUCH(self.cfg.image_dim, self.cfg.text_dim, self.cfg.hidden_dim, self.cfg.hash_dim)

    def get_optimizers(self):
        """
        Initialize learning optimizers

        :returns: learning optimizers
        """
        optimizer_gen = Adam([
            {'params': self.model.image_module.parameters()},
            {'params': self.model.text_module.parameters()}
        ], lr=self.cfg.lr, weight_decay=0.0005)

        optimizer_dis = Adam(self.model.hash_dis.parameters(), lr=self.cfg.lr, betas=(0.5, 0.9), weight_decay=0.0001)
        return optimizer_gen, optimizer_dis

    def train_epoch(self, epoch):
        """
        Train model for 1 epoch

        :param: epoch: current epoch
        """
        t1 = time.time()

        self.model.to(self.device).train()

        e_losses_dict = {'adv': 0., 'bb': 0., 'quant': 0., 'ntxent': 0., 'sum': 0.}

        for i, (dataloader_idx, sample_idxs, img1, img2, txt1, txt2, label) in enumerate(self.dataloader_quadruplets):
            img1 = img1.to(self.device)  # original images
            img2 = img2.to(self.device)  # augmented images
            txt1 = txt1.to(self.device)  # original texts
            txt2 = txt2.to(self.device)  # augmented texts
            h_img1, h_img2, h_txt1, h_txt2 = self.model(img1, img2, txt1, txt2)


            # print(label)
            # import ipdb; ipdb.set_trace()

            self.Hi1[dataloader_idx, :] = h_img1
            self.Hi2[dataloader_idx, :] = h_img2
            self.Ht1[dataloader_idx, :] = h_txt1
            self.Ht2[dataloader_idx, :] = h_txt2

            current_batch_size = len(dataloader_idx)

            self.train_discriminator(h_img1, h_img2, h_txt1, h_txt2, current_batch_size)

            e_losses_dict = self.train_hash_network(h_img1, h_img2, h_txt1, h_txt2, dataloader_idx, e_losses_dict,label)

        self.losses.append(e_losses_dict)

        self.B = (((self.Hi1.detach() + self.Hi2.detach()) / 2 +
                   (self.Ht1.detach() + self.Ht2.detach()) / 2) / 2).sign()

        self.update_optimizer_params(epoch)

        delta_t = time.time() - t1
        s = '[{}/{}] Train: {:.3f}s, Losses: A: {adv:10.2f}, BB: {bb:10.2f}, Q: {quant:10.2f}, NTX: {ntxent:10.2f}'
        self.logger.info(s.format(epoch + 1, self.cfg.max_epoch, delta_t, **e_losses_dict))

    def train_discriminator(self, h_img1, h_img2, h_txt1, h_txt2, batch_size):
        """
        Train hash discriminator

        :param: h_img1: batch of image hashes #1 (original)
        :param: h_img2: batch of image hashes #2 (augmented)
        :param: h_txt1: batch of text hashes #1 (original)
        :param: h_txt2: batch of text hashes #2 (augmented)
        :param: batch_size: current batch size
        """
        self.optimizer_dis.zero_grad()
        self.train_discriminator_step(h_txt1, h_img1, batch_size)
        self.train_discriminator_step(h_txt2, h_img2, batch_size)
        self.optimizer_dis.step()

    def train_discriminator_step(self, v_m1, v_m2, batch_size):
        """
        Train hash discriminator for one step

        :param: v_m1: 'real' samples
        :param: v_m2: 'fake' samples
        :param: batch_size: current batch size
        """
        d_real = self.model.discriminate_hash(v_m1.detach())
        d_real = -torch.log(torch.sigmoid(d_real)).mean()
        d_real.backward()

        # train with fake (TXT)
        d_fake = self.model.discriminate_hash(v_m2.detach())
        d_fake = -torch.log(torch.ones(batch_size).to(self.device) - torch.sigmoid(d_fake)).mean()
        d_fake.backward()

        # train with gradient penalty (GP)

        # interpolate real and fake data
        alpha = torch.rand(batch_size, self.cfg.hash_dim).to(self.device)
        interpolates = alpha * v_m1.detach() + (1 - alpha) * v_m2.detach()
        interpolates.requires_grad_()
        disc_interpolates = self.model.discriminate_hash(interpolates)
        # get gradients with respect to inputs
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        # calculate penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10  # 10 is GP hyperparameter
        gradient_penalty.backward()

    def train_hash_network(self, h_img1, h_img2, h_txt1, h_txt2, ind, e_losses_dict,lables):
        """
        Train hashing network

        :param: h_img1: batch of image hashes #1 (original)
        :param: h_img2: batch of image hashes #2 (augmented)
        :param: h_txt1: batch of text hashes #1 (original)
        :param: h_txt2: batch of text hashes #2 (augmented)
        :param: ind: indexes of samples in current batch
        :param: e_losses_dict: dictionary with epoch losses

        :returns: updated dictionary with epoch losses
        """
        err, batch_losses_dict = self.calculate_losses(h_img1, h_img2, h_txt1, h_txt2, ind,lables)
        e_losses_dict = self.update_epoch_losses_dict(e_losses_dict, batch_losses_dict)

        self.optimizer_gen.zero_grad()
        err.backward()
        self.optimizer_gen.step()

        return e_losses_dict

    def calculate_losses(self, h_img1, h_img2, h_txt1, h_txt2, ind, lables):
        """
        Calculate losses
        :param: h_img1: batch of image hashes #1 (original)
        :param: h_img2: batch of image hashes #2 (augmented)
        :param: h_txt1: batch of text hashes #1 (original)
        :param: h_txt2: batch of text hashes #2 (augmented)
        :param: ind: indexes of samples in current batch

        :returns: total epoch loss and updated dictionary with epoch losses
        """

        loss_ntxent = self.calc_ntxent_loss(h_img1, h_img2, h_txt1, h_txt2)
        # loss of image/text regression concatanateate the image features and the text features regression LEO
        loss_adv_h = self.calc_adversarial_loss(h_txt1, h_img1) * self.cfg.alpha #h_txt1
        loss_quant = self.calc_quantization_loss(h_img1, h_img2, h_txt1, h_txt2, ind) * self.cfg.beta
        loss_bb = self.calc_bit_balance_loss(h_img1, h_img2, h_txt1, h_txt2) * self.cfg.gamma

        err = loss_ntxent + loss_adv_h + loss_quant + loss_bb #+ loss_binary_img +loss_binary_sent  +loss_binary_img_aug +loss_binary_sent_aug #loss of the regression


        e_losses = self.get_batch_losses_dict(loss_adv_h, loss_ntxent, loss_quant, loss_bb)

        return err, e_losses
    
    def calc_zerone_loss(self,feats_s2, label): # loss for supervised
        # print ('ciao')
        # import ipdb; ipdb.set_trace()
        zerone = torch.ones(feats_s2.size(0), feats_s2.size(1))
        # import ipdb; ipdb.set_trace()
        for i in range(len(label)):
            if label[i] == 0:
                zerone[i] = -1
            else:
                zerone[i] = 1
        

        zerone =zerone.to('cuda:0')

        assert zerone.shape == feats_s2.shape, f"{zerone.shape} and {feats_s2.shape}"
        diff = torch.absolute(zerone - feats_s2)
        # import ipdb; ipdb.set_trace()

        # print (diff)
        
        return diff.sum(axis=-1).mean()

    def calc_ntxent_loss(self, h_img1, h_img2, h_txt1, h_txt2):
        """
        Calculate NTXent Loss

        :param: h_img1: batch of image hashes #1 (original)
        :param: h_img2: batch of image hashes #2 (augmented)
        :param: h_txt1: batch of text hashes #1 (original)
        :param: h_txt2: batch of text hashes #2 (augmented)

        :returns: NTXent Loss
        """
        loss_ntxent_inter = self.contr_loss(h_img1, h_txt1, type='orig') * self.cfg.contrastive_weights[0]
        loss_ntxent_intra_img = self.contr_loss(h_img1, h_img2, type='orig') * self.cfg.contrastive_weights[1]
        loss_ntxent_intra_txt = self.contr_loss(h_txt1, h_txt2, type='orig') * self.cfg.contrastive_weights[2]
        

        loss_ntxent = loss_ntxent_inter +loss_ntxent_intra_txt+loss_ntxent_intra_img

        return loss_ntxent

    def calc_adversarial_loss(self, h_img1, h_img2):
        """
        Calculate Adversarial Loss

        :param: h_img1: batch of image hashes #1 (original)
        :param: h_img2: batch of image hashes #2 (augmented)

        :returns: Adversarial Loss
        """
        loss_adv_h1 = -torch.log(torch.sigmoid(self.model.discriminate_hash(h_img1))).sum()
        loss_adv_h2 = -torch.log(torch.sigmoid(self.model.discriminate_hash(h_img2))).sum()
        loss_adv_h = (loss_adv_h1 + loss_adv_h2)
        return loss_adv_h

    def calc_quantization_loss(self, h_img1, h_img2, h_txt1, h_txt2, ind):
        """
        Calculate Quantization Loss

        :param: h_img1: batch of image hashes #1 (original)
        :param: h_img2: batch of image hashes #2 (augmented)
        :param: h_txt1: batch of text hashes #1 (original)
        :param: h_txt2: batch of text hashes #2 (augmented)

        :returns: Quantization Loss
        """
        loss_quant_img1 = torch.sum(torch.pow(self.B[ind, :] - h_img1, 2))
        loss_quant_img2 = torch.sum(torch.pow(self.B[ind, :] - h_img2, 2))
        loss_quant_txt1 = torch.sum(torch.pow(self.B[ind, :] - h_txt1, 2))
        loss_quant_txt2 = torch.sum(torch.pow(self.B[ind, :] - h_txt2, 2))
        loss_quant = (loss_quant_img1 + loss_quant_img2 + loss_quant_txt1 + loss_quant_txt2)
        return loss_quant

    def calc_bit_balance_loss(self, h_img1, h_img2, h_txt1, h_txt2):
        """
        Calculate Bit Balance Loss

        :param: h_img1: batch of image hashes #1 (original)
        :param: h_img2: batch of image hashes #2 (augmented)
        :param: h_txt1: batch of text hashes #1 (original)
        :param: h_txt2: batch of text hashes #2 (augmented)

        :returns: Bit Balance Loss
        """

        loss_bb_img1 = torch.sum(torch.pow(torch.sum(h_img1, dim=1), 2))
        loss_bb_img2 = torch.sum(torch.pow(torch.sum(h_img2, dim=1), 2))
        loss_bb_txt1 = torch.sum(torch.pow(torch.sum(h_txt1, dim=1), 2))
        loss_bb_txt2 = torch.sum(torch.pow(torch.sum(h_txt2, dim=1), 2))
        loss_bb = (loss_bb_img1 + loss_bb_img2 + loss_bb_txt1 + loss_bb_txt2)
        return loss_bb

    @staticmethod
    def get_batch_losses_dict(loss_adv_h, loss_ntxent, loss_quant, loss_bb):
        """
        Get batch losses dict

        :param: loss_adv_h: Adversarial Loss
        :param: loss_ntxent: NTXent Loss
        :param: loss_quant: Quantization Loss
        :param: loss_bb: Bit Balance Loss

        :returns: batch losses dict
        """
        b_losses_dict = {'adv': 0., 'bb': 0., 'quant': 0., 'ntxent': 0.}
        b_losses_dict['adv'] += loss_adv_h.detach().cpu().numpy()
        b_losses_dict['ntxent'] += loss_ntxent.detach().cpu().numpy()
        b_losses_dict['quant'] += loss_quant.detach().cpu().numpy()
        b_losses_dict['bb'] += loss_bb.detach().cpu().numpy()
        b_losses_dict['sum'] = sum(b_losses_dict.values())
        return b_losses_dict

    @staticmethod
    def update_epoch_losses_dict(e_losses_dict, b_losses_dict):
        """
        Update epoch losses dictionary

        :param: e_losses_dict: epoch losses dictionary
        :param: b_losses_dict: batch losses dictionary

        :returns: updated dictionary with epoch losses
        """
        for k, v in b_losses_dict.items():
            e_losses_dict[k] += v
        return e_losses_dict

    def update_optimizer_params(self, epoch):
        """
        Update parameters of optimizers

        :param: epoch: current epoch
        """
        if epoch % 50 == 0:
            for params in self.optimizer_gen.param_groups:
                params['lr'] = max(params['lr'] * 0.8, 1e-6)

    def eval(self, epoch):
        """
        Evaluate model. Calculate MAPs for current epoch
        Save model and hashes if current epoch is the best

        :param: epoch: current epoch
        """
        self.model.eval()

        qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, indexes = self.get_codes_labels_indexes()
        # ketu
  


        mapi2t, mapt2i, mapi2i, mapt2t, mapavg = self.calc_maps_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY,
                                                                  self.cfg.retrieval_map_k)

        map_k_5 = (mapi2t, mapt2i, mapi2i, mapt2t, mapavg)
        map_k_10 = self.calc_maps_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, 10)
        map_k_20 = self.calc_maps_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, 20)
        map_r = self.calc_maps_rad(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, [0, 1, 2, 3, 4, 5])
        p_at_k = self.calc_p_top_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY)
        maps_eval = (map_k_5, map_k_10, map_k_20, map_r, p_at_k)

        self.update_maps_dict(mapi2t, mapt2i, mapi2i, mapt2t, mapavg)

        if mapavg > self.maps_max['avg']:
            self.update_max_maps_dict(mapi2t, mapt2i, mapi2i, mapt2t, mapavg)

            self.save_model()
            self.save_hash_codes()

        self.save_model('last')
        write_pickle(os.path.join(self.path, 'maps_eval.pkl'), maps_eval)

        self.model.train()

    def calc_maps_k(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, k):
        """
        Calculate MAPs, in regards to K

        :param: qBX: query hashes, modality X
        :param: qBY: query hashes, modality Y
        :param: rBX: response hashes, modality X
        :param: rBY: response hashes, modality Y
        :param: qLX: query labels, modality X
        :param: qLY: query labels, modality Y
        :param: rLX: response labels, modality X
        :param: rLY: response labels, modality Y
        :param: k: k

        :returns: MAPs
        """
        mapi2t = calc_map_k(qBX, rBY, qLX, rLY, k)
        mapt2i = calc_map_k(qBY, rBX, qLY, rLX, k)
        mapi2i = calc_map_k(qBX, rBX, qLX, rLX, k)
        mapt2t = calc_map_k(qBY, rBY, qLY, rLY, k)

        avg = (mapi2t.item() + mapt2i.item() + mapi2i.item() + mapt2t.item()) * 0.25

        mapi2t, mapt2i, mapi2i, mapt2t, mapavg = mapi2t.item(), mapt2i.item(), mapi2i.item(), mapt2t.item(), avg

        s = 'Valid: mAP@{:2d}, avg: {:3.3f}, i->t: {:3.3f}, t->i: {:3.3f}, i->i: {:3.3f}, t->t: {:3.3f}'
        self.logger.info(s.format(k, mapavg, mapi2t, mapt2i, mapi2i, mapt2t))

        return mapi2t, mapt2i, mapi2i, mapt2t, mapavg

    def calc_maps_rad(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, rs):
        """
        Calculate MAPs, in regard to Hamming radius

        :param: qBX: query hashes, modality X
        :param: qBY: query hashes, modality Y
        :param: rBX: response hashes, modality X
        :param: rBY: response hashes, modality Y
        :param: qLX: query labels, modality X
        :param: qLY: query labels, modality Y
        :param: rLX: response labels, modality X
        :param: rLY: response labels, modality Y
        :param: rs: hamming radiuses to output

        :returns: MAPs
        """
        mapsi2t = calc_map_rad(qBX, rBY, qLX, rLY)
        mapst2i = calc_map_rad(qBY, rBX, qLY, rLX)
        mapsi2i = calc_map_rad(qBX, rBX, qLX, rLX)
        mapst2t = calc_map_rad(qBY, rBY, qLY, rLY)

        mapsi2t, mapst2i, mapsi2i, mapst2t = mapsi2t.numpy(), mapst2i.numpy(), mapsi2i.numpy(), mapst2t.numpy()

        s = 'Valid: mAP HR{}, i->t: {:3.3f}, t->i: {:3.3f}, i->i: {:3.3f}, t->t: {:3.3f}'
        for r in rs:
            self.logger.info(s.format(r, mapsi2t[r], mapst2i[r], mapsi2i[r], mapst2t[r]))

        return mapsi2t, mapst2i, mapsi2i, mapst2t

    def get_codes_labels_indexes(self):
        """
        Generate binary codes from duplet dataloaders for query and response

        :param: remove_replications: remove replications from dataset

        :returns: hash codes and labels for query and response, sample indexes
        """
        # hash X, hash Y, labels X/Y, image replication factor, indexes X, indexes Y
        qBX, qBY, qLXY, irf_q, (qIX, qIY) = self.generate_codes_from_dataloader(self.dataloader_q)
        # hash X, hash Y, labels X/Y, image replication factor, indexes X, indexes Y
        rBX, rBY, rLXY, irf_db, (rIX, rIY) = self.generate_codes_from_dataloader(self.dataloader_db)

        # get Y Labels
        qLY = qLXY
        rLY = rLXY

        # X modality sometimes contains replicated samples (see datasets), remove them by selecting each nth element
        # remove replications for hash codes
        qBX = self.get_each_nth_element(qBX, irf_q)
        rBX = self.get_each_nth_element(rBX, irf_db)
        # remove replications for labels
        qLX = self.get_each_nth_element(qLXY, irf_q)
        rLX = self.get_each_nth_element(rLXY, irf_db)
        # remove replications for indexes
        qIX = self.get_each_nth_element(qIX, irf_q)
        rIX = self.get_each_nth_element(rIX, irf_db)
        # import ipdb; ipdb.set_trace()

        return qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, (qIX, qIY, rIX, rIY)

    @staticmethod
    def get_each_nth_element(arr, n):
        """
        intentionally ugly solution, needed to avoid query replications during test/validation

        :return: array
        """
        return arr[::n]

    def generate_codes_from_dataloader(self, dataloader):
        """
        Generate binary codes from duplet dataloader

        :param: dataloader: duplet dataloader

        :returns: hash codes for given duplet dataloader, image replication factor of dataset
        """
        num = len(dataloader.dataset)
        # print ()


        irf = dataloader.dataset.image_replication_factor
        # print (irf)
        # import ipdb; ipdb.set_trace()

        Bi = torch.zeros(num, self.cfg.hash_dim).to(self.device)
        Bt = torch.zeros(num, self.cfg.hash_dim).to(self.device)
        L = torch.zeros(num, self.cfg.label_dim).to(self.device)

        dataloader_idxs = []

        # for i, input_data in tqdm(enumerate(test_dataloader)):
        for i, (idx, sample_idxs, img, txt, label) in enumerate(dataloader):
          
            dataloader_idxs = self.stack_idxs(dataloader_idxs, sample_idxs)
            img = img.to(self.device)
            txt = txt.to(self.device)
            if len(label.shape) == 1:
                label = one_hot(label, num_classes=self.cfg.label_dim).to(self.device)
            else:
                label.to(self.device)
            bi = self.model.generate_img_code(img)
            bt = self.model.generate_txt_code(txt)
            idx_end = min(num, (i + 1) * self.cfg.batch_size)
            Bi[i * self.cfg.batch_size: idx_end, :] = bi.data
            Bt[i * self.cfg.batch_size: idx_end, :] = bt.data
            L[i * self.cfg.batch_size: idx_end, :] = label.data

        Bi = torch.sign(Bi)
        Bt = torch.sign(Bt)
        return Bi, Bt, L, irf, dataloader_idxs

    @staticmethod
    def stack_idxs(idxs, idxs_batch):
        if len(idxs) == 0:
            return [ib for ib in idxs_batch]
        else:
            return [torch.hstack(i).detach() for i in zip(idxs, idxs_batch)]

    def update_maps_dict(self, mapi2t, mapt2i, mapi2i, mapt2t, mapavg):
        """
        Update MAPs dictionary (append new values)

        :param: mapi2t: I-to-T MAP
        :param: mapt2i: T-to-I MAP
        :param: mapi2i: I-to-I MAP
        :param: mapt2t: T-to-T MAP
        :param: mapavg: average MAP
        """
        self.maps['i2t'].append(mapi2t)
        self.maps['t2i'].append(mapt2i)
        self.maps['i2i'].append(mapi2i)
        self.maps['t2t'].append(mapt2t)
        self.maps['avg'].append(mapavg)

    def update_max_maps_dict(self, mapi2t, mapt2i, mapi2i, mapt2t, mapavg):
        """
        Update max MAPs dictionary (replace values)

        :param: mapi2t: I-to-T MAP
        :param: mapt2i: T-to-I MAP
        :param: mapi2i: I-to-I MAP
        :param: mapt2t: T-to-T MAP
        :param: mapavg: average MAP
        """
        self.maps_max['i2t'] = mapi2t
        self.maps_max['t2i'] = mapt2i
        self.maps_max['i2i'] = mapi2i
        self.maps_max['t2t'] = mapt2t
        self.maps_max['avg'] = mapavg

    def save_hash_codes(self):
        """
        Save hash codes on a disk
        """
        with torch.cuda.device(self.device):
            torch.save([self.Hi1, self.Hi2, self.Ht1, self.Ht2], os.path.join(self.path, 'hash_codes_i_t.pth'))
        with torch.cuda.device(self.device):
            torch.save(self.B, os.path.join(self.path, 'hash_code.pth'))

    def training_complete(self):
        """
        Output training summary: time and best results
        """
        self.save_train_results_dict()

        current = time.time()
        delta = current - self.since
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(delta // 60, delta % 60))
        s = 'Max Avg MAP: {avg:3.3f}, Max MAPs: i->t: {i2t:3.3f}, t->i: {t2i:3.3f}, i->i: {i2i:3.3f}, t->t: {t2t:3.3f}'
        self.logger.info(s.format(**self.maps_max))

    def save_train_results_dict(self):
        """
        Save training history dictionary
        """
        res = self.maps
        res['losses'] = self.losses
        write_pickle(os.path.join(self.path, 'train_res_dict.pkl'), res)

    def test(self):
        """
        Test model. Calculate MAPs, PR-curves and P@K values.
        """
        self.model.to(self.device).train()

        self.model.eval()

        qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, indexes = self.get_codes_labels_indexes()
        # import ipdb; ipdb.set_trace() #ketu
        # print (indexes)
        

        maps = self.calc_maps_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, self.cfg.retrieval_map_k)
        map_dict = self.make_maps_dict(*maps)
        pr_dict = self.calc_pr_curves(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY)
        pk_dict = self.calc_p_top_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY)
        self.save_test_results_dicts(map_dict, pr_dict, pk_dict)

        self.model.train()

        current = time.time()
        delta = current - self.since
        self.logger.info('Test complete in {:.0f}m {:.0f}s'.format(delta // 60, delta % 60))

    def make_maps_dict(self, mapi2t, mapt2i, mapi2i, mapt2t, mapavg):
        """
        Make MAP dict from MAP values

        :param: mapi2t: I-to-T MAP
        :param: mapt2i: T-to-I MAP
        :param: mapi2i: I-to-I MAP
        :param: mapt2t: T-to-T MAP
        :param: mapavg: Average MAP

        :returns: MAPs dictionary
        """

        map_dict = {'mapi2t': mapi2t, 'mapt2i': mapt2i, 'mapi2i': mapi2i, 'mapt2t': mapt2t, 'mapavg': mapavg}

        s = 'Avg MAP: {:3.3f}, MAPs: i->t: {:3.3f}, t->i: {:3.3f}, i->i: {:3.3f}, t->t: {:3.3f}'
        self.logger.info(s.format(mapavg, mapi2t, mapt2i, mapi2i, mapt2t))

        return map_dict

    def calc_pr_curves(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY):
        """
        Calculate PR-curves

        :param: qBX: query hashes, modality X
        :param: qBY: query hashes, modality Y
        :param: rBX: response hashes, modality X
        :param: rBY: response hashes, modality Y
        :param: qLX: query labels, modality X
        :param: qLY: query labels, modality Y
        :param: rLX: response labels, modality X
        :param: rLY: response labels, modality Y

        :returns: PR-curves dictionary
        """
        p_i2t, r_i2t = pr_curve(qBX, rBY, qLX, rLY, tqdm_label='I2T')
        p_t2i, r_t2i = pr_curve(qBY, rBX, qLY, rLX, tqdm_label='T2I')
        p_i2i, r_i2i = pr_curve(qBX, rBX, qLX, rLX, tqdm_label='I2I')
        p_t2t, r_t2t = pr_curve(qBY, rBY, qLY, rLY, tqdm_label='T2T')

        pr_dict = {'pi2t': p_i2t, 'ri2t': r_i2t,
                   'pt2i': p_t2i, 'rt2i': r_t2i,
                   'pi2i': p_i2i, 'ri2i': r_i2i,
                   'pt2t': p_t2t, 'rt2t': r_t2t}

        self.logger.info('Precision-recall values: {}'.format(pr_dict))

        return pr_dict

    def calc_p_top_k(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY):
        """
        Calculate P@K values

        :param: qBX: query hashes, modality X
        :param: qBY: query hashes, modality Y
        :param: rBX: response hashes, modality X
        :param: rBY: response hashes, modality Y
        :param: qLX: query labels, modality X
        :param: qLY: query labels, modality Y
        :param: rLX: response labels, modality X
        :param: rLY: response labels, modality Y

        :returns: P@K values
        """
        k = [1, 5, 10, 20, 50] + list(range(100, 1001, 100))

        pk_i2t = p_top_k(qBX, rBY, qLX, rLY, k, tqdm_label='I2T')
        pk_t2i = p_top_k(qBY, rBX, qLY, rLX, k, tqdm_label='T2I')
        pk_i2i = p_top_k(qBX, rBX, qLX, rLX, k, tqdm_label='I2I')
        pk_t2t = p_top_k(qBY, rBY, qLY, rLY, k, tqdm_label='T2T')

        pk_dict = {'k': k,
                   'pki2t': pk_i2t,
                   'pkt2i': pk_t2i,
                   'pki2i': pk_i2i,
                   'pkt2t': pk_t2t}

        self.logger.info('P@K values: {}'.format(pk_dict))

        return pk_dict

    # def save_test_results_dicts(self, map_dict, pr_dict, pk_dict):
    #     """
    #     Save test results dictionary

    #     :param: map_dict: MAPs dictionary
    #     :param: pr_dict: PR-curves dictionary
    #     :param: pk_dict: P@K values dictionary
    #     """
    #     write_pickle(os.path.join(self.path, 'map_dict.pkl'), map_dict)
    #     write_pickle(os.path.join(self.path, 'pr_dict.pkl'), pr_dict)
    #     write_pickle(os.path.join(self.path, 'pk_dict.pkl'), pk_dict)

    def save_test_results_dicts(self, map_dict, pr_dict, pk_dict): #,query_index,retrieval_index
        """
        Save test results dictionary

        :param: map_dict: MAPs dictionary
        :param: pr_dict: PR-curves dictionary
        :param: pk_dict: P@K values dictionary
        """
        write_pickle(os.path.join(self.path, 'map_dict.pkl'), map_dict)
        write_pickle(os.path.join(self.path, 'pr_dict.pkl'), pr_dict)
        write_pickle(os.path.join(self.path, 'pk_dict.pkl'), pk_dict)
        


    def load_model(self, tag='best'):
        """
        Load model from the disk

        :param: tag: name tag
        """
        self.model.load(os.path.join(self.path, self.model.module_name + '_' + str(tag) + '.pth'))

    def save_model(self, tag='best'):
        """
        Save model on the disk

        :param: tag: name tag
        """
        self.model.save(self.model.module_name + '_' + str(tag) + '.pth', self.path, cuda_device=self.device)

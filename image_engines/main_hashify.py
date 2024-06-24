import socket
from omegaconf import DictConfig, OmegaConf
from src.vit_cmmae import CMMAEBackbone

import torch
from torch.utils.data import DataLoader
from src.bigearthnet_dataset.BEN_DataModule_LMDB_Encoder import BENDataSet

import numpy as np
from src.augmentations import (
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
)

import tqdm
import os
import wandb


def calc_bit_balance_loss_vessel(feats_s2):
    """
    Calculate Bit Balance Loss

    :param: h_img1: batch of image hashes #1 (original)
    :param: h_img2: batch of image hashes #2 (augmented)
    :param: h_txt1: batch of text hashes #1 (original)
    :param: h_txt2: batch of text hashes #2 (augmented)

    :returns: Bit Balance Loss
    """
    return torch.sum(torch.pow(torch.sum(feats_s2, dim=1), 2))


def calc_quantization_loss_vessel(feats_s2):
    """
    Calculate Quantization Loss

    :param: h_img1: batch of image hashes #1 (original)
    :param: h_img2: batch of image hashes #2 (augmented)
    :param: h_txt1: batch of text hashes #1 (original)
    :param: h_txt2: batch of text hashes #2 (augmented)

    :returns: Quantization Loss
    """

    B = feats_s2.detach().sign()

    return torch.sum(torch.pow(B - feats_s2, 2))


def calc_bit_balance_loss(feats_s1, feats_s2):
    """
    Calculate Bit Balance Loss

    :param: h_img1: batch of image hashes #1 (original)
    :param: h_img2: batch of image hashes #2 (augmented)
    :param: h_txt1: batch of text hashes #1 (original)
    :param: h_txt2: batch of text hashes #2 (augmented)

    :returns: Bit Balance Loss
    """
    loss_bb_img1 = torch.sum(torch.pow(torch.sum(feats_s1, dim=1), 2))
    loss_bb_img2 = torch.sum(torch.pow(torch.sum(feats_s2, dim=1), 2))
    loss_bb = loss_bb_img1 + loss_bb_img2
    return loss_bb


def calc_quantization_loss(feats_s1, feats_s2):
    """
    Calculate Quantization Loss

    :param: h_img1: batch of image hashes #1 (original)
    :param: h_img2: batch of image hashes #2 (augmented)
    :param: h_txt1: batch of text hashes #1 (original)
    :param: h_txt2: batch of text hashes #2 (augmented)

    :returns: Quantization Loss
    """

    B = (0.5 * (feats_s1.detach() + feats_s2.detach())).sign()

    loss_quant_img1 = torch.sum(torch.pow(B - feats_s1, 2))
    loss_quant_img2 = torch.sum(torch.pow(B - feats_s2, 2))
    loss_quant = loss_quant_img1 + loss_quant_img2
    return loss_quant


def cmmae_ntxent_loss_func(feats_s1, feats_s2, temp: float = 0.07, key: str = "feats"):
    feats_s1_norm = torch.norm(feats_s1, dim=1).reshape(-1, 1)
    feats_s2_norm = torch.norm(feats_s2, dim=1).reshape(-1, 1)

    feats_s1_norm = torch.div(feats_s1, feats_s1_norm)
    feats_s2_norm = torch.div(feats_s2, feats_s2_norm)
    feats_s12 = torch.cat((feats_s1_norm, feats_s2_norm))
    feats_s21 = torch.cat((feats_s2_norm, feats_s1_norm))

    similarities = torch.mm(feats_s12, feats_s12.T)
    sim_by_tau = torch.div(similarities, temp)
    exp_sim_by_tau = torch.exp(sim_by_tau)
    sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
    exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)

    numerators = torch.exp(
        torch.div(torch.nn.CosineSimilarity()(feats_s12, feats_s21), temp)
    )
    denominators = sum_of_rows - exp_sim_by_tau_diag
    num_by_den = torch.div(numerators, denominators)
    neglog_num_by_den = -torch.log(num_by_den)
    return torch.mean(neglog_num_by_den)


def calc_zerone_loss(feats_s2, labels):

    categorical = torch.argmax(labels, dim=1)
    zerone = categorical.unsqueeze(1).repeat(1, 128).to(feats_s2.device)
    zerone[zerone == 0] = -1

    diff = torch.absolute(zerone - feats_s2)

    return diff.sum(axis=-1).mean()


def main(run_path, ckpt_path):

    MAX_EPOCHS = 25
    BETA = 0.001
    GAMMA = 0.01
    CONTRASTIVE_TEMP = 0.5

    cfg = OmegaConf.load(f"{run_path}/args.yaml")
    OmegaConf.set_struct(cfg, False)

    gpu = cfg.devices[0]
    dev = f"cuda:{gpu}"

    ### PREPARE MODEL
    backbone = CMMAEBackbone(**cfg.backbone.kwargs)

    # load ckpt weights into backbone
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    ret = backbone.load_state_dict(state, strict=False)
    print("Loading return", ret)

    # we only train the hashing module
    backbone.to(dev).train()
    total_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print("Total params", total_params)

    for n, p in backbone.named_parameters():
        p.requires_grad = False
        if n.startswith("g.") or n.startswith("f."):
            p.requires_grad = True
            print("Train", n)

    total_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print("Hashing params", total_params)

    ### DATA LOADING
    # Build data augmentation pipeline
    pipelines = []
    for aug_cfg in cfg.augmentations:
        pipelines.append(
            NCropAugmentation(
                build_transform_pipeline(cfg.data.dataset, aug_cfg, cfg),
                aug_cfg.num_crops,
            )
        )
    transform = FullTransformPipeline(pipelines)

    # Prepare dataset
    train_dataset = BENDataSet(
        transform=transform,
        root_dir=cfg.data.root_dir,
        split_dir=cfg.data.split_dir,
        split="train",
        max_img_idx=cfg.data.get("max_img_idx", None),
        img_size=(cfg.data.num_bands, cfg.data.img_size, cfg.data.img_size),
    )

    # Prepare dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.optimizer.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    ### LOGGING SETUP
    path, file_name = os.path.split(ckpt_path)
    new_file_name = f"hash_{file_name}"
    new_ckpt_path = f"{path}/{new_file_name}"
    assert (
        os.path.exists(new_ckpt_path) == False
    ), f"Path {new_ckpt_path} already exists!"

    os.environ["WANDB_DIR"] = "./"
    stop = file_name.find("=") - 3 - 9
    cfg.name = file_name[:stop] + "_hash"
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.name,
        tags=["hash"],
        config=cfg,
    )

    ### START TRAINING
    lr = 0.0001
    optimizer = torch.optim.AdamW(lr=lr, params=backbone.parameters())
    metrics_dict = {}

    for e in range(MAX_EPOCHS):

        print(f"[{e+1}/{MAX_EPOCHS}]")

        for step, (idx, imgs, label) in enumerate(tqdm.tqdm(train_loader)):

            optimizer.zero_grad()
            imgs = imgs[0].to(dev)
            out = backbone(imgs)

            assert isinstance(out, dict)

            ### select [1] => features after projection f()/g()
            feats_s1 = out["s1"]
            feats_s2 = out["s2"]

            # Calc all losses
            loss_ntxent = cmmae_ntxent_loss_func(feats_s1, feats_s2, CONTRASTIVE_TEMP)
            loss_q = BETA * calc_quantization_loss(feats_s1, feats_s2)
            loss_bb = GAMMA * calc_bit_balance_loss(feats_s1, feats_s2)
            loss = loss_ntxent + loss_bb + loss_q

            metrics_dict["train_nt_xent_loss"] = loss_ntxent.detach().cpu().item()
            metrics_dict["train_quant_loss"] = loss_q.detach().cpu().item()
            metrics_dict["train_bb_loss"] = loss_bb.detach().cpu().item()
            metrics_dict["lr"] = lr

            loss.backward()
            optimizer.step()

        run.log(metrics_dict)
        print(metrics_dict)

        torch.save(
            {
                "trained_epochs": e,
                "state_dict": backbone.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            new_ckpt_path[:-5] + f".ep{e}" + new_ckpt_path[-5:],
        )


if __name__ == "__main__":

    run_path = "<TODO-./trained_models/{...}>"
    ckpt_path = f"<TODO-{run_path}/cmmae-vit_base_15-avg-10multi_enc-2x_enc-50_masking-lr0.0001-4rec_loss-on_masked-w_MIM0.5avg-wo_MDE-kc0kuiot-ep=0.ckpt>"

    main(run_path, ckpt_path)

from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
from src.loss import mim_loss_func
from src.pos_encoding import *
import numpy as np
from typing import Callable, Tuple
import pytorch_lightning as pl

from src.vit_duch import DUCHBackbone

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class DUCH(pl.LightningModule):
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
    ):

        super().__init__()

        # backbone related
        self.backbone_args: Dict[str, Any] = cfg.backbone.kwargs
        self.base_model: Callable = DUCHBackbone
        self.backbone_name: str = cfg.backbone.name

        kwargs = self.backbone_args.copy()
        self.backbone: nn.Module = self.base_model(**kwargs)
        self.features_dim: int = self.backbone.num_features

        # gather backbone info from timm
        self._vit_embed_dim: int = self.backbone.pos_embed.size(-1)
        self._vit_patch_size: int = self.backbone_args.get("patch_size", 15)
        self._vit_num_patches: int = self.backbone.patch_embed.num_patches

        # training related
        self.max_epochs: int = cfg.max_epochs

        # optimizer related
        self.batch_size: int = cfg.optimizer.batch_size
        self.lr: float = cfg.optimizer.lr
        self.weight_decay: float = cfg.optimizer.weight_decay
        self.extra_optimizer_args: Dict[str, Any] = cfg.optimizer.kwargs

        # scheduler related
        self.scheduler: str = cfg.scheduler.name
        self.min_lr: float = cfg.scheduler.min_lr
        self.warmup_start_lr: float = cfg.scheduler.warmup_start_lr
        self.warmup_epochs: int = cfg.scheduler.warmup_epochs
        self.scheduler_interval: str = cfg.scheduler.interval

        # gather backbone info from timm
        self._vit_embed_dim: int = self.backbone.pos_embed.size(-1)
        self._vit_patch_size: int = self.backbone_args.patch_size
        self._vit_num_patches: int = self.backbone.patch_embed.num_patches

        self.contrastive_temp = cfg.method_kwargs.contrastive_temp

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(DUCH, DUCH).add_and_assert_specific_cfg(cfg)
        return cfg

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        learnable_params = self.parameters()

        optimizer = torch.optim.AdamW(
            learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        max_warmup_steps = (
            self.warmup_epochs
            * (self.trainer.estimated_stepping_batches / self.max_epochs)
            if self.scheduler_interval == "step"
            else self.warmup_epochs
        )
        max_scheduler_steps = (
            self.trainer.estimated_stepping_batches
            if self.scheduler_interval == "step"
            else self.max_epochs
        )
        scheduler = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=max_warmup_steps,
                max_epochs=max_scheduler_steps,
                warmup_start_lr=(
                    self.warmup_start_lr if self.warmup_epochs > 0 else self.lr
                ),
                eta_min=self.min_lr,
            ),
            "interval": self.scheduler_interval,
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        # modified base forward
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)

        # S1/S2 passed through encoder
        forward_dicts = self.backbone(X)
        out_s1, feats_s1 = forward_dicts["s1"]
        out_s2, feats_s2 = forward_dicts["s2"]

        # out_ == immediate output (class token) of encoders
        # feats_ == immediate projected class
        out = {
            "out_s1": out_s1,
            "out_s2": out_s2,
            "feats_s1": feats_s1,
            "feats_s2": feats_s2,
        }

        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for MAE reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss
        """

        out = super().training_step(batch, batch_idx)

        metrics = {}

        # Add MDE and MSP loss to outs of encoder
        out_s1 = out["out_s1"][0]
        out_s2 = out["out_s2"][0]

        out_s1_norm = torch.norm(out_s1, dim=1).reshape(-1, 1)
        out_s2_norm = torch.norm(out_s2, dim=1).reshape(-1, 1)

        out_s1_norm = torch.div(out_s1, out_s1_norm)
        out_s2_norm = torch.div(out_s2, out_s2_norm)

        pairwise_sims = torch.diag(torch.mm(out_s1_norm, out_s2_norm.T))
        loss_MDE = torch.log(1 + torch.exp(pairwise_sims)).mean()

        metrics["train_mde_loss"] = loss_MDE

        # Add MSP loss to outs of encoder
        def msp(feats):
            distances = torch.cdist(feats, feats, p=2)
            inf_diag = torch.zeros(
                distances.shape[1], distances.shape[1], device=feats.device
            ).fill_diagonal_(torch.inf)
            distances += inf_diag

            # feats.shape[1] == embed dim
            min_indices = torch.argmin(distances, dim=1, keepdim=True)
            min_indices = min_indices.repeat(1, feats.shape[1])

            min_distance_feats = torch.gather(feats, dim=0, index=min_indices)
            return torch.nn.CosineSimilarity()(feats, min_distance_feats)

        loss_MSP = 0.5 * (msp(out_s1) + msp(out_s2)).mean()
        metrics["train_msp_loss"] = loss_MSP

        # Add NT-Xent loss to projections
        loss_ntxent = mim_loss_func(out, self.contrastive_temp)
        metrics["train_nt_xent_loss"] = loss_ntxent

        # Coefficients for weighting of MSP and MDE losses
        weights = np.logspace(np.log(10**-4), np.log(1), 150, base=np.exp(1))
        alpha = weights[self.current_epoch]
        beta = weights[self.current_epoch]
        metrics["alpha"] = alpha

        self.log_dict(metrics, on_step=True, sync_dist=True)
        return loss_ntxent - beta * loss_MDE + alpha * loss_MSP

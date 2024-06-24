from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
from src.loss import mae_loss_func
import pytorch_lightning as pl

from typing import Callable, Tuple
from src.utils import omegaconf_select

from src.pos_encoding import *
from timm.models.vision_transformer import Block

from src.vit_cmmae_vessel import CMMAEBackboneVessel
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class CMMAEVesselDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        embed_dim,
        depth,
        num_heads,
        num_patches,
        patch_size,
        mlp_ratio=4.0,
        decoder_in_chans=10,
    ) -> None:
        super().__init__()

        self.num_patches = num_patches

        self.decoder_embed = nn.Linear(in_dim, embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.Sequential(
            *[
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(
            embed_dim, patch_size**2 * decoder_in_chans, bias=True
        )

        # init all weights according to MAE's repo
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        decoder_pos_embed = generate_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # unshuffle
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


class CMMAEVessel(pl.LightningModule):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements MAE.

        Extra cfg settings:
            method_kwargs:
                mask_ratio (float): percentage of image to mask.
                decoder_embed_dim (int): number of dimensions for the embedding in the decoder
                decoder_depth (int) depth of the decoder
                decoder_num_heads (int) number of heads for the decoder
                norm_pix_loss (bool): whether to normalize the pixels of each patch with their
                    respective mean and std for the loss. Defaults to False.
        """

        super().__init__()

        # backbone related
        self.backbone_args: Dict[str, Any] = cfg.backbone.kwargs
        self.base_model: Callable = CMMAEBackboneVessel
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

        # masking/reconstruction related
        self.mask_ratio: float = cfg.method_kwargs.mask_ratio
        self.norm_pix_loss: bool = cfg.method_kwargs.norm_pix_loss
        self.supervised_loss: bool = cfg.method_kwargs.supervised_loss
        self.supervised_loss_factor: float = cfg.method_kwargs.supervised_loss_factor
        self.supervised_weight = torch.tensor(0.53, device="cuda:0")

        # gather backbone info from timm
        self._vit_embed_dim: int = self.backbone.pos_embed.size(-1)
        self._vit_patch_size: int = self.backbone_args.patch_size
        self._vit_num_patches: int = self.backbone.patch_embed.num_patches

        decoder_embed_dim: int = cfg.method_kwargs.decoder_embed_dim
        decoder_depth: int = cfg.method_kwargs.decoder_depth
        decoder_num_heads: int = cfg.method_kwargs.decoder_num_heads
        decoder_in_chans: int = cfg.backbone.kwargs.in_chans

        # decoder
        self.decoder = CMMAEVesselDecoder(
            in_dim=self.features_dim,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            num_patches=self._vit_num_patches,
            patch_size=self._vit_patch_size,
            mlp_ratio=4.0,
            decoder_in_chans=decoder_in_chans,
        )

        # w/o Hashing Module
        # self.f = nn.Sequential(
        #     nn.Linear(self._vit_embed_dim, self._vit_embed_dim),
        #     nn.GELU(),
        #     nn.Linear(self._vit_embed_dim, 128),
        #     nn.Tanh()
        # )

        if self.supervised_loss:
            self.clf_head = nn.Linear(self._vit_embed_dim, 1)

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(CMMAEVessel, CMMAEVessel).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(
            cfg, "method_kwargs.decoder_embed_dim"
        )
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.decoder_depth")
        assert not omegaconf.OmegaConf.is_missing(
            cfg, "method_kwargs.decoder_num_heads"
        )

        cfg.method_kwargs.mask_ratio = omegaconf_select(
            cfg, "method_kwargs.mask_ratio", 0.75
        )
        cfg.method_kwargs.norm_pix_loss = omegaconf_select(
            cfg,
            "method_kwargs.norm_pix_loss",
            False,
        )

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

        out = {}
        if self.training:
            feats, patch_feats, mask, ids_restore, feats_unmasked = self.backbone(
                X, self.mask_ratio
            )
            pred = self.decoder(patch_feats, ids_restore)

            out.update({"mask": mask, "pred": pred, "feats_unmasked": feats_unmasked})
        else:
            feats = self.backbone(X)

        if self.supervised_loss:
            vessel_pred = self.clf_head(feats_unmasked)
            out.update({"vessel_pred": vessel_pred})

        out.update({"feats": feats})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for MAE reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a list of batches [X], where [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss
        """

        _, _, ohe_targets = batch

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        metrics = {}

        # Reconstruction Loss
        patch_size = self._vit_patch_size
        imgs = batch[1]
        reconstruction_loss = 0
        for i in range(self.num_large_crops):
            reconstruction_loss += mae_loss_func(
                imgs[i],
                out["pred"][i],
                out["mask"][i],
                patch_size,
                norm_pix_loss=self.norm_pix_loss,
                num_channels=4,
            )
        reconstruction_loss /= self.num_large_crops

        clf_loss = torch.tensor(0, device=class_loss.device)
        if self.supervised_loss:
            ves_logits = out["vessel_pred"][0]
            cat_targets = torch.argmax(ohe_targets, dim=1)

            ves_logits = ves_logits.squeeze(1)
            clf_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                ves_logits.float(), cat_targets.float()
            )

        metrics.update(
            {
                "train_clf_loss": clf_loss,
                "train_reconstruction_loss": reconstruction_loss,
            }
        )
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return reconstruction_loss + self.supervised_loss_factor * clf_loss + class_loss

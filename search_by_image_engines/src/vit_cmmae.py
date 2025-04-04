from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, VisionTransformer
import numpy as np

from src.pos_encoding import *


class CMMAEBackbone(VisionTransformer):
    def __init__(
        self,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        img_size=120,
        patch_size=15,
        in_chans=12,
        multi_sensor_encoder_depth=10,
        cross_sensor_encoder_depth=2,
        global_pool="avg",
        fc_norm=None,
        num_classes=0,
        **kwargs,
    ):
        assert in_chans == 12
        assert img_size % patch_size == 0

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=cross_sensor_encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            global_pool=global_pool,
            fc_norm=fc_norm,
            num_classes=num_classes,
            norm_layer=norm_layer,
            **kwargs,
        )

        in_chans_s1 = 2
        in_chans_s2 = 10
        self.multi_sensor_encoder_depth = multi_sensor_encoder_depth
        self.mask_ratio = 0.0

        # --------------------------------------------------------------------------
        # CMMAE S1 encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans_s1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_patches = self.patch_embed.num_patches

        if multi_sensor_encoder_depth:
            self.blocks_s1 = nn.Sequential(
                *[
                    Block(
                        embed_dim,
                        num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                    )
                    for _ in range(multi_sensor_encoder_depth)
                ]
            )
            self.norm_s1 = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # CMMAE S2 encoder specifics
        self.patch_embed_s2 = PatchEmbed(img_size, patch_size, in_chans_s2, embed_dim)
        self.cls_token_s2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        assert self.patch_embed_s2.num_patches == self.num_patches

        if multi_sensor_encoder_depth:
            self.blocks_s2 = nn.Sequential(
                *[
                    Block(
                        embed_dim,
                        num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                    )
                    for _ in range(multi_sensor_encoder_depth)
                ]
            )
            self.norm_s2 = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # S1 (S2) encoder specifics
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        # --------------------------------------------------------------------------
        # CMMAE shared encoder specifics
        self.blocks = nn.Sequential(
            *[
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(cross_sensor_encoder_depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        ### Hashing module
        self.f = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 128),
            nn.Tanh(),
        )
        self.g = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 128),
            nn.Tanh(),
        )

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = generate_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_s2.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.cls_token_s2, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio, ids_shuffle, ids_restore):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder_s1(self, x, mask_ratio, ids_shuffle, ids_restore):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(
            x, mask_ratio, ids_shuffle, ids_restore
        )

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        if self.multi_sensor_encoder_depth:
            x = self.blocks_s1(x)
            x = self.norm_s1(x)

        return x, mask, ids_restore

    def forward_encoder_s2(self, x, mask_ratio, ids_shuffle, ids_restore):
        # embed patches
        x = self.patch_embed_s2(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(
            x, mask_ratio, ids_shuffle, ids_restore
        )

        # append cls token
        cls_token = self.cls_token_s2 + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        if self.multi_sensor_encoder_depth:
            x = self.blocks_s2(x)
            x = self.norm_s2(x)

        return x, mask, ids_restore

    def forward_shared_encoder(self, x):
        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, imgs, mask_ratio=0):

        N = imgs.shape[0]
        L = self.num_patches

        # sort noise for each sample
        noise = torch.rand(N, L, device=imgs.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        imgs_s1 = imgs[:, 10:, :, :]
        imgs_s2 = imgs[:, :10, :, :]

        # Pass S1 (S2) through corresponding encoder
        pre_feats_s1, mask_s1, ids_restore_s1 = self.forward_encoder_s1(
            imgs_s1, mask_ratio, ids_shuffle, ids_restore
        )

        noise = torch.rand(N, L, device=imgs.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        pre_feats_s2, mask_s2, ids_restore_s2 = self.forward_encoder_s2(
            imgs_s2,
            mask_ratio,
            ids_shuffle,
            ids_restore,
        )

        # Pass obtained features through shared encoder
        feats_s1 = self.forward_shared_encoder(pre_feats_s1)
        feats_s2 = self.forward_shared_encoder(pre_feats_s2)

        # Classification head of transformer
        out_s1 = self.forward_head(feats_s1)
        out_s2 = self.forward_head(feats_s2)

        if mask_ratio:
            return {
                "s1": (out_s1, feats_s1, mask_s1, ids_restore_s1),
                "s2": (out_s2, feats_s2, mask_s2, ids_restore_s2),
            }

        out_s1 = self.f(out_s1)
        out_s2 = self.g(out_s2)

        return {
            "s1": out_s1,
            "s2": out_s2,
        }

    def get_s1_hash(self, imgs_s1):
        pre_feats_s1 = self.forward_encoder_s1(imgs_s1)
        feats_s1 = self.forward_shared_encoder(pre_feats_s1)
        out_s1 = self.forward_head(feats_s1)

        # Apply deep hashing module
        out_s1 = self.f(out_s1)

        # Convert to binary numbers
        hash_codes = out_s1.sign()
        hash_codes[hash_codes == -1] = 0

        return hash_codes

    def get_s2_hash(self, imgs_s2):
        pre_feats_s2 = self.forward_encoder_s2(imgs_s2)
        feats_s2 = self.forward_shared_encoder(pre_feats_s2)
        out_s2 = self.forward_head(feats_s2)

        # Apply deep hashing module
        out_s2 = self.f(out_s2)

        # Convert to binary numbers
        hash_codes = out_s2.sign()
        hash_codes[hash_codes == -1] = 0

        return hash_codes

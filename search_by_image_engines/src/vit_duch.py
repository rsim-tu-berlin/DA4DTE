from functools import partial

import torch
import torch.nn as nn
from src.pos_encoding import generate_2d_sincos_pos_embed
from timm.models.vision_transformer import Block, PatchEmbed, VisionTransformer


class DUCHBackbone(VisionTransformer):
    """Masked Autoencoder with VisionTransformer backbone
    Adapted from https://github.com/facebookresearch/mae.
    """

    def __init__(
        self,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        img_size=120,
        patch_size=8,
        in_chans=12,
        pre_encoder_depth=12,
        global_pool="token",
        fc_norm=None,
        num_classes=0,
        identical_masking=True,
        disjoint_masking=False,
        **kwargs,
    ):
        assert in_chans == 12
        assert img_size % patch_size == 0
        assert depth == pre_encoder_depth

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=pre_encoder_depth,
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

        # --------------------------------------------------------------------------
        # S1 encoder specifics

        # we have to assign PatchEmbed to self.patch_embed since it is used parent ViT class,
        # otherwise these parameters do not receive gradients!
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans_s1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_patches = self.patch_embed.num_patches
        self.identical_masking = identical_masking
        self.disjoint_masking = disjoint_masking

        self.blocks = nn.Sequential(
            *[
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(pre_encoder_depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # S2 encoder specifics
        self.patch_embed_s2 = PatchEmbed(img_size, patch_size, in_chans_s2, embed_dim)
        self.cls_token_s2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        assert self.patch_embed_s2.num_patches == self.num_patches

        self.blocks_s2 = nn.Sequential(
            *[
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(pre_encoder_depth)
            ]
        )
        self.norm_s2 = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # Projection specifics, only operating on class tokens

        # in DUCH method, we always have an additional projection head which can be exploited as
        ### Deep Hashing Module
        self.g = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 128),
        )
        self.tanh = nn.Tanh()

        # --------------------------------------------------------------------------

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

    def forward_encoder_s1(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        return x

    def forward_encoder_s2(self, x):
        # embed patches
        x = self.patch_embed_s2(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token_s2 + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.blocks_s2(x)
        x = self.norm_s2(x)

        return x

    def forward(self, imgs):
        """Forward a batch of 12-channel images through separate encoders and
        apply projection layer.

        Args:
            imgs (torch.Tensor):

        Returns:
            Dict: Dict with keys 's1', 's2' with values (class_tokens, projection(class_tokens))
        """

        imgs_s1 = imgs[:, 10:, :, :]
        imgs_s2 = imgs[:, :10, :, :]

        encoded_s1 = self.forward_encoder_s1(imgs_s1)
        encoded_s2 = self.forward_encoder_s2(imgs_s2)

        out_s1 = self.forward_head(encoded_s1)
        out_s2 = self.forward_head(encoded_s2)

        feats_s1 = self.g(out_s1)
        feats_s2 = self.g(out_s2)

        feats_s1 = self.tanh(feats_s1)
        feats_s2 = self.tanh(feats_s2)

        return {"s1": (out_s1, feats_s1), "s2": (out_s2, feats_s2)}

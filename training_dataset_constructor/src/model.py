from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, VisionTransformer

import numpy as np

def generate_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Adapted from https://github.com/facebookresearch/mae.
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = generate_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def generate_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    # Adapted from https://github.com/facebookresearch/mae.

    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = generate_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = generate_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def generate_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Adapted from https://github.com/facebookresearch/mae.
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Network(VisionTransformer):
    def __init__(
        self,
        img_size=120,
        patch_size=15,
        in_chans=12,
        embed_dim=768,
        depth=12,
        pre_encoder_depth=10,
        shared_encoder_depth=2,
        num_heads=12,
        mlp_ratio=4.0,
        global_pool="avg",
        fc_norm=None,
        num_classes=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    ):
        assert in_chans == 12
        assert img_size % patch_size == 0

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=shared_encoder_depth,
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
        self.pre_encoder_depth = pre_encoder_depth
        self.mask_ratio = 0.0


        # --------------------------------------------------------------------------
        # CMMAE S1 encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans_s1, embed_dim) 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_patches = self.patch_embed.num_patches

        if pre_encoder_depth:
            self.blocks_s1 = nn.Sequential(
                *[
                    Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                    for _ in range(pre_encoder_depth)
                ]
            )
            self.norm_s1 = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # CMMAE S2 encoder specifics
        self.patch_embed_s2 = PatchEmbed(img_size, patch_size, in_chans_s2, embed_dim)
        self.cls_token_s2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        assert self.patch_embed_s2.num_patches == self.num_patches

        if pre_encoder_depth:
            self.blocks_s2 = nn.Sequential(
                *[
                    Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                    for _ in range(pre_encoder_depth)
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
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(shared_encoder_depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------


        self.f = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 128),
            nn.Tanh()
        )
        self.g = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 128),
            nn.Tanh()
        )

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = generate_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True
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
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        if self.pre_encoder_depth:
            x = self.blocks_s1(x)
            x = self.norm_s1(x)

        return x
    
    def forward_encoder_s2(self, x):
        x = self.patch_embed_s2(x)
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token_s2 + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        if self.pre_encoder_depth:
            x = self.blocks_s2(x)
            x = self.norm_s2(x)

        return x
    
    def forward_shared_encoder(self, x):
        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def get_s1_hash(self, imgs_s1):
        pre_feats_s1 = self.forward_encoder_s1(imgs_s1)
        feats_s1 = self.forward_shared_encoder(pre_feats_s1)
        out_s1 = self.forward_head(feats_s1)

        # Apply deep hashing module
        out_s1 = self.f(out_s1)

        # Convert to binary numbers
        hash_codes = out_s1.sign()
        hash_codes[hash_codes==-1] = 0

        return hash_codes
    
    
    def get_s2_hash(self, imgs_s2):
        pre_feats_s2 = self.forward_encoder_s2(imgs_s2)
        feats_s2 = self.forward_shared_encoder(pre_feats_s2)
        out_s2 = self.forward_head(feats_s2)

        # Apply deep hashing module
        out_s2 = self.f(out_s2)

        # Convert to binary numbers
        hash_codes = out_s2.sign()
        hash_codes[hash_codes==-1] = 0

        return hash_codes


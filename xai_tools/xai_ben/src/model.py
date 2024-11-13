import torch

from src.vit_LRP import Block, PatchEmbed, VisionTransformerLRP
from src.layers_ours import *


class CrossModalMaskedAutoencoderViT(VisionTransformerLRP):
    """CSMAE re-implementation using only module (layers) that are extended
    to be compatible with relevance propgation, to allow LRP.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=24,
        pre_encoder_depth=12,
        shared_encoder_depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=0,
        norm_layer=LayerNorm,
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
            num_classes=num_classes,
            **kwargs,
        )

        in_chans_s1 = 2
        in_chans_s2 = 10

        self.pre_encoder_depth = pre_encoder_depth
        self.patch_size = patch_size

        # --------------------------------------------------------------------------
        # CMMAE S1 encoder specifics

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans_s1, embed_dim)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_patches = self.patch_embed.num_patches

        if pre_encoder_depth:
            self.blocks_s1 = Sequential(
                *[
                    Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True)
                    for _ in range(pre_encoder_depth)
                ]
            )
            self.norm_s1 = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # CMMAE S2 encoder specifics
        self.patch_embed_s2 = PatchEmbed(img_size, patch_size, in_chans_s2, embed_dim)
        self.cls_token_s2 = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        assert self.patch_embed_s2.num_patches == self.num_patches

        if pre_encoder_depth:
            self.blocks_s2 = Sequential(
                *[
                    Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True)
                    for _ in range(pre_encoder_depth)
                ]
            )
            self.norm_s2 = LayerNorm(embed_dim)

        # --------------------------------------------------------------------------
        # S1 (S2) encoder specifics
        self.pos_embed = torch.nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        # --------------------------------------------------------------------------
        # CMMAE shared encoder specifics
        self.blocks = Sequential(
            *[
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True)
                for i in range(shared_encoder_depth)
            ]
        )
        self.norm = LayerNorm(embed_dim)
        # --------------------------------------------------------------------------

    def forward_encoders_s1(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = self.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])

        x.register_hook(self.save_inp_grad)

        for blk in self.blocks_s1:
            x = blk(x)
            x = self.norm_s1(x)

        for blk in self.blocks:
            x = blk(x)
            x = self.norm_s1(x)

        x = self.norm(x)

        return x

    def forward_encoders_s2(self, x):
        B = x.shape[0]
        x = self.patch_embed_s2(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = self.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])

        x.register_hook(self.save_inp_grad)

        for blk in self.blocks_s2:
            x = blk(x)
            x = self.norm_s2(x)

        for blk in self.blocks:
            x = blk(x)
            x = self.norm_s2(x)

        x = self.norm(x)

        return x

    def avgpool(self, x):
        # x.shape = N, L, D

        x2d = x.unsqueeze(1)  # N, 1, L, D
        x2d = self.fixpool(x2d)  # N, 1, L, D

        return x2d.squeeze(2)

    def forward_s1(self, imgs, mask_ratio=0):
        assert mask_ratio == 0

        imgs_s1 = imgs[:, 10:, :, :]
        feats_s1 = self.forward_encoders_s1(imgs_s1)

        out_s1 = self.pool(
            feats_s1, dim=1, indices=torch.tensor(0, device=feats_s1.device)
        )

        out_s1 = self.classifier1(out_s1)
        out_s1 = self.classifier_norm(out_s1)
        out_s1 = self.classifier_nonlin(out_s1)
        pred_s1 = self.classifier2(out_s1)

        return pred_s1

    def forward_s2(self, imgs, mask_ratio=0):
        assert mask_ratio == 0

        imgs_s2 = imgs[:, :10, :, :]
        feats_s2 = self.forward_encoders_s2(imgs_s2)

        ### cls_token
        out_s2 = self.pool(
            feats_s2, dim=1, indices=torch.tensor(0, device=feats_s2.device)
        )

        out_s2 = self.classifier1(out_s2)
        out_s2 = self.classifier_norm(out_s2)
        out_s2 = self.classifier_nonlin(out_s2)
        pred_s2 = self.classifier2(out_s2)

        return pred_s2


def vit_base(**kwargs):
    model = CrossModalMaskedAutoencoderViT(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=LayerNorm,
        **kwargs,
    )
    return model

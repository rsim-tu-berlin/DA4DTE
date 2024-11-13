import torch

from src.vit_BiLRP import Block, VisionTransformerBiLRP
from src.layers_ours import *


class MaskedAutoencoder4ViT(VisionTransformerBiLRP):
    """Masked Autoencoder with VisionTransformer backbone
    Adapted from https://github.com/facebookresearch/mae.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=4,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=0,
        norm_layer=LayerNorm,
        supervised_loss="",
        **kwargs,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            **kwargs,
        )

        self.supervised_loss = supervised_loss

        # --------------------------------------------------------------------------
        self.blocks = Sequential(
            *[
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True)
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        self.apply(self._init_weights)

    def forward_encoder(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])

        x.register_hook(self.save_inp_grad)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x

    def forward(self, imgs, mask_ratio=0):
        assert mask_ratio == 0
        feats = self.forward_encoder(imgs)

        out = self.pool(feats, dim=1, indices=torch.tensor(0, device=feats.device))
        out = self.f(out)

        return out


def vit_base(**kwargs):
    model = MaskedAutoencoder4ViT(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=LayerNorm,
        **kwargs,
    )
    return model

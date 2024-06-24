import omegaconf
from src.utils import omegaconf_select


def parse_cfg(cfg: omegaconf.DictConfig):
    """Add default values to config if not provided correct.

    Args:
        cfg (omegaconf.DictConfig): config

    Returns:
        omegaconf.DictConfig: config
    """

    # fix number of big/small crops
    cfg.data.num_large_crops = 1
    cfg.data.num_small_crops = 0

    # extra optimizer kwargs
    cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})
    cfg.optimizer.kwargs.betas = omegaconf_select(
        cfg, "optimizer.kwargs.betas", [0.9, 0.999]
    )

    if cfg.method == "cmmae":
        cfg.name = "-".join(
            [
                "cmmae",
                cfg.backbone.name + "_" + str(cfg.backbone.kwargs.patch_size),
                cfg.backbone.kwargs.global_pool,
                str(cfg.backbone.kwargs.multi_sensor_encoder_depth)
                + "multi_enc-"
                + str(cfg.backbone.kwargs.cross_sensor_encoder_depth)
                + "x_enc",
                str(int(cfg.method_kwargs.mask_ratio * 100)) + "_masking",
                "lr" + str(cfg.optimizer.lr),
                (
                    "4rec_loss"
                    if (
                        cfg.method_kwargs.apply_umr_loss
                        and cfg.method_kwargs.apply_cmr_loss
                    )
                    else (
                        "2rec_loss"
                        if cfg.method_kwargs.apply_cmr_loss
                        else "unimodalrec_loss"
                    )
                ),
                "on_masked",
                (
                    "w_MIM"
                    + str(cfg.method_kwargs.mim_temp)
                    + cfg.backbone.kwargs.global_pool
                    if cfg.method_kwargs.apply_mim_loss
                    else "wo_MIM"
                ),
                "w_MDE" if cfg.method_kwargs.apply_mde_loss else "wo_MDE",
            ]
        )
    else:
        cfg.name = "-".join(
            [cfg.method, cfg.backbone.name + "_" + str(cfg.backbone.kwargs.patch_size)]
        )

    return cfg

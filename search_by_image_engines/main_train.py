import inspect


# hack away pl-bolts UnderReviewWarnings
import warnings


warnings.simplefilter("ignore")
original_filterwarnings = warnings.filterwarnings


def _filterwarnings(*args, **kwargs):
    return original_filterwarnings(*args, **{**kwargs, "append": True})


warnings.filterwarnings = _filterwarnings


import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from torch.utils.data import DataLoader

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from src.bigearthnet_dataset.BEN_DataModule_LMDB_Encoder import BENDataSet
import os

from src.args import parse_cfg
from src.augmentations import (
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
)

from src.cmmae import CMMAE
from src.duch import DUCHBackbone

from src.vessel_dataset.vessel_ds import VesselDataset
from src.cmmae_vessel import CMMAEVessel

from src.utils import Checkpointer


@hydra.main(version_base="1.2", config_path="./", config_name="cfg.yaml")
def main(cfg: DictConfig):

    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)
    seed_everything(cfg.seed)

    # Instantiate CMMAE model
    model = CMMAE(cfg)
    model = model.to(memory_format=torch.channels_last)

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

    # Prepare dataset for pretraining. This can be replaced by an instance
    # of the VesselDataset class, something along the lines of the next code chunk
    train_dataset = BENDataSet(
        transform=transform,
        root_dir=cfg.data.root_dir,
        split_dir=cfg.data.split_dir,
        split="train",
        max_img_idx=cfg.data.get("max_img_idx", None),
        img_size=(cfg.data.num_bands, cfg.data.img_size, cfg.data.img_size),
    )
    # train_dataset = VesselDataset(
    #     root_dir=cfg.data.root_dir,
    #     split_dir=cfg.data.split_dir,
    #     split=cfg.data.split,
    #     transform=transform,
    #     img_size=(4, 128, 128),
    # )

    # Prepare dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.optimizer.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Build callbacks considering the train configuration
    callbacks = []

    ckpt = Checkpointer(
        cfg,
        logdir=cfg.checkpoint.dir,
    )
    callbacks.append(ckpt)

    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            name=cfg.name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
        )
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))

        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    # Run training
    trainer_kwargs = OmegaConf.to_container(cfg)
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {
        name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs
    }
    trainer_kwargs.update(
        {
            "logger": wandb_logger if cfg.wandb.enabled else None,
            "callbacks": callbacks,
            "enable_checkpointing": False,
            "strategy": DDPStrategy(find_unused_parameters=cfg.find_unused_parameters),
        }
    )

    trainer = Trainer(
        log_every_n_steps=10,
        **trainer_kwargs,
        num_sanity_val_steps=2,
    )
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()

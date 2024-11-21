from utils import logger
from controller_DUCH import ControllerDUCH
from dataset.data_handlers import get_dataloaders, DataHandlerAugmentedTxtImg
from dataset.datasets import DatasetQuadrupletAugmentedTxtImg, DatasetDuplet1
from configs.config import cfg
from images_augment import new_mapping
import os
import torch
from icecream import ic


def main():
    log = logger()

    mode = "TEST" if cfg.test else "TRAIN"
    print("which mode", mode)
    s = "Init ({}): {}, {}, {} bits, tag: {}, preset: {}"
    log.info(
        s.format(
            mode, cfg.model_type, cfg.dataset.upper(), cfg.hash_dim, cfg.tag, cfg.preset
        )
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = new_mapping["6:0"]
    device = torch.device(cfg.cuda_device if torch.cuda.is_available() else "cpu")
    print("device is", device)
    torch.set_num_threads(64)

    # Training with img-img_aug-txt-txt_aug
    data_handler_class = DataHandlerAugmentedTxtImg
    ds_train_class = DatasetQuadrupletAugmentedTxtImg
    ds_query_class = DatasetDuplet1
    ds_db_class = DatasetDuplet1
    controller_class = ControllerDUCH
    ic()
    dl_train, dl_q, dl_db = get_dataloaders(
        data_handler_class, ds_train_class, ds_query_class, ds_db_class
    )
    ic()
    controller = controller_class(log, cfg, (dl_train, dl_q, dl_db))

    if cfg.test:
        controller.load_model()
        controller.test()
    else:
        for epoch in range(cfg.max_epoch):
            controller.train_epoch(epoch)
            if ((epoch + 1) % cfg.valid_freq == 0) and cfg.valid:
                controller.eval(epoch)
            # save the model
            if epoch + 1 == cfg.max_epoch:
                controller.save_model("last")
                controller.training_complete()


if __name__ == "__main__":
    main()

import logging
import os

import hydra
import torch
from omegaconf import DictConfig

from src.data.dataloader import NiftiDataModule
from src.models import build_model
from src.training.trainer import Trainer
from src.training.training_utils import seed_everything

local_rank = int(os.environ.get("LOCAL_RANK", 0))
os.environ["MIOPEN_USER_DB_PATH"] = f"/tmp/miopen_cache_{local_rank}"
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = f"/tmp/miopen_cache_{local_rank}"
os.environ["WANDB__SERVICE_WAIT"] = "300"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S")

torch.backends.cudnn.benchmark = False


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    seed_everything(cfg.seed, local_rank=local_rank)
    model = build_model(cfg)
    dataloader = NiftiDataModule(cfg)
    trainer = Trainer(model, dataloader, cfg)
    # train on tuh
    trainer.fit()
    # eval on kits and kirc
    trainer.eval()


if __name__ == "__main__":
    main()

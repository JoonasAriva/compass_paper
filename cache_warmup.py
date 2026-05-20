# warmup_cache.py
import os
os.environ["MIOPEN_USER_DB_PATH"] = "/tmp/miopen_cache_warmup"
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = "/tmp/miopen_cache_warmup"

from hydra import initialize, compose
from src.data.dataloader import NiftiDataModule, validate_cache
from torch.utils.data import DataLoader

with initialize(config_path="conf", version_base=None):
    cfg = compose(config_name="config", overrides=[
        "distributed=false",
        "dataloader.train_workers=2",
        "dataloader.val_workers=2",
    ])

datamodule = NiftiDataModule(cfg)

print("Building train cache...")
loader = DataLoader(datamodule.train_dataset, batch_size=1, num_workers=8, persistent_workers=False, prefetch_factor=1)
print("Length:", len(datamodule.train_dataset))
for i, batch in enumerate(loader):
    if i % 50 == 0:
        print(f"  {i} done")

print("Building test cache...")
print("Length:", len(datamodule.test_dataset))
loader = DataLoader(datamodule.test_dataset, batch_size=1, num_workers=8, persistent_workers=False, prefetch_factor=1)
for i, batch in enumerate(loader):
    if i % 50 == 0:
        print(f"  {i} done")

print("Cache built successfully")
import os

os.environ["MIOPEN_USER_DB_PATH"] = "/tmp/miopen_cache_warmup"
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = "/tmp/miopen_cache_warmup"

import torch
import pandas as pd
from src.data.dataloader import NiftiDataModule
from torch.utils.data import DataLoader

from src.models.compass_2d_resnet import ResNetCompass

model = ResNetCompass(name='resnet18', norm_layer='group', framework="compass", pretrained=None)
sd = torch.load('/scratch/project_465002884/results/compass/resnet18/2d_slice/2026-06-10/16-44-14/checkpoints/best.pth',
                map_location='cuda:0', weights_only=True)
sd = sd["model"]
new_sd = {key.replace("module.", ""): value for key, value in sd.items()}
model.load_state_dict(state_dict=new_sd)
model.cuda()

from hydra import initialize, compose

with initialize(config_path="conf", version_base=None):
    cfg = compose(config_name="config", overrides=[
        "distributed=false",
        "dataloader.train_workers=2",
        "dataloader.val_workers=2",
    ])

datamodule = NiftiDataModule(cfg)

print("Calculating compass scores for training set...")
loader = DataLoader(datamodule.train_dataset, batch_size=1, num_workers=8, persistent_workers=False, prefetch_factor=1)
print("Length:", len(datamodule.train_dataset))

ctx = torch.set_grad_enabled(False)
model.eval()
# main_df = pd.DataFrame()
# with ctx:
#     for i, batch in enumerate(loader):
#         if i % 50 == 0:
#             print(f"  {i} done")
#
#         scan = torch.squeeze(batch["image"]).cuda()
#         scan = torch.permute(scan, (1, 0, 2, 3))  # C,D,H,W --> D,C,H,W
#         scan_end = batch["original_depth"]
#
#         with torch.autocast(device_type="cuda"):
#             preds = model(scan, scan_end=scan_end)["predictions"]
#
#             df = pd.DataFrame()
#             df["compass_scores"] = pd.Series(preds.cpu()[:, 0])
#             df["slice_nr"] = df.index
#             df["scan_id"] = batch["scan_path"][0]
#             df["scan_class"] = batch["class"][0]
#             df["normal_kidney_slices"] = batch["normal_kidney_slices"][0]
#             df["slice_classes"] = batch["slice_classes"][0]
#
#             main_df = pd.concat([main_df, df])
# main_df.to_csv("train_set_compass_scores_2d_slice_vol2.csv", index=False)
# print("Saved train scores")
print("Calculating compass scores for validation set...")
print("Length:", len(datamodule.test_dataset))
loader = DataLoader(datamodule.test_dataset, batch_size=1, num_workers=8, persistent_workers=False, prefetch_factor=1)

main_df = pd.DataFrame()
with ctx:
    for i, batch in enumerate(loader):
        if i % 50 == 0:
            print(f"  {i} done")

        scan = torch.squeeze(batch["image"]).cuda()
        scan = torch.permute(scan, (1, 0, 2, 3))  # C,D,H,W --> D,C,H,W
        scan_end = batch["original_depth"]

        with torch.autocast(device_type="cuda"):
            preds = model(scan, scan_end=scan_end)["predictions"]

            df = pd.DataFrame()
            df["compass_scores"] = pd.Series(preds.cpu()[:, 0])
            df["slice_nr"] = df.index
            df["scan_id"] = batch["scan_path"][0]
            df["scan_class"] = batch["class"][0]
            df["normal_kidney_slices"] = batch["normal_kidney_slices"][0]
            df["slice_classes"] = batch["slice_classes"][0]

            main_df = pd.concat([main_df, df])
main_df.to_csv("test_set_compass_scores_2d_slice_kits_and_kirc.csv", index=False)
print("Saved validation scores")




import os

os.environ["MIOPEN_USER_DB_PATH"] = "/tmp/miopen_cache_warmup"
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = "/tmp/miopen_cache_warmup"

import json
from pathlib import Path

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

# Where the cached feature tensors + manifests go. Adjust to your scratch layout.
FEATURE_ROOT = Path("/scratch/project_465002884/2d_slice_compass_features")


def save_scan_features(out_dir, idx, batch, preds, dataset):
    """
    Truncates to the scan's real (unpadded) depth, saves one .pt per scan,
    and returns a manifest row for it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_end = batch["original_depth"].reshape(-1)[0].item()
    # min() guards against either case: preds already truncated internally
    # by the model, or preds covering the full padded depth.
    valid_len = min(preds.shape[0], int(scan_end))

    feats = preds[:valid_len].float().cpu()
    slice_classes = batch["slice_classes"].reshape(-1)[:valid_len].cpu()
    normal_kidney_slices = batch["normal_kidney_slices"].reshape(-1)[:valid_len].cpu()

    scan_path = batch["scan_path"][0] if isinstance(batch["scan_path"], list) else batch["scan_path"]
    label = bool(dataset.labels[idx][0])

    stem = Path(scan_path).name.replace(".nii.gz", "").replace(".nii", "")
    cache_path = out_dir / f"{stem}.pt"

    torch.save({
        "features": feats,                     # (valid_len, feat_dim)
        "slice_classes": slice_classes,         # (valid_len,)
        "normal_kidney_slices": normal_kidney_slices,
        "scan_path": scan_path,
        "label": label,
    }, cache_path)

    return {
        "cache_path": str(cache_path),
        "scan_path": scan_path,
        "label": label,
        "num_instances": valid_len,
    }


print("Calculating compass scores for training set...")
loader = DataLoader(datamodule.train_dataset, batch_size=1, num_workers=8, persistent_workers=False, prefetch_factor=1)
print("Length:", len(datamodule.train_dataset))

ctx = torch.set_grad_enabled(False)
model.eval()

train_out_dir = FEATURE_ROOT / "train"
train_manifest = []

with ctx:
    for i, batch in enumerate(loader):
        if i % 50 == 0:
            print(f"  {i} done")

        scan = torch.squeeze(batch["image"]).cuda()
        scan = torch.permute(scan, (1, 0, 2, 3))  # C,D,H,W --> D,C,H,W
        scan_end = batch["original_depth"]

        with torch.autocast(device_type="cuda"):
            preds = model(scan, scan_end=scan_end)["feature_vector"]

            row = save_scan_features(train_out_dir, i, batch, preds, datamodule.train_dataset)
            train_manifest.append(row)

with open(train_out_dir / "manifest.json", "w") as f:
    json.dump(train_manifest, f, indent=2)

print("Saved train vectors")
print("Calculating vectors for validation set...")
print("Length:", len(datamodule.test_dataset))
loader = DataLoader(datamodule.test_dataset, batch_size=1, num_workers=8, persistent_workers=False, prefetch_factor=1)

test_out_dir = FEATURE_ROOT / "test"
test_manifest = []

with ctx:
    for i, batch in enumerate(loader):
        if i % 50 == 0:
            print(f"  {i} done")

        scan = torch.squeeze(batch["image"]).cuda()
        scan = torch.permute(scan, (1, 0, 2, 3))  # C,D,H,W --> D,C,H,W
        scan_end = batch["original_depth"]

        with torch.autocast(device_type="cuda"):
            preds = model(scan, scan_end=scan_end)["feature_vector"]

            row = save_scan_features(test_out_dir, i, batch, preds, datamodule.test_dataset)
            test_manifest.append(row)

with open(test_out_dir / "manifest.json", "w") as f:
    json.dump(test_manifest, f, indent=2)

print("Saved validation vectors")